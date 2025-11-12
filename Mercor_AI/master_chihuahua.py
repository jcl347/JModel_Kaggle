# ============================================
# Kaggle env fix — pin conflicting dependencies
# Run this cell FIRST, before importing anything else.
# ============================================
import sys, subprocess

def pip_install(pkgs):
    # -qU keeps logs quiet but upgrades/downgrades as needed
    cmd = [sys.executable, "-m", "pip", "install", "-qU"] + pkgs
    print("PIP:", " ".join(pkgs))
    subprocess.check_call(cmd)

# Version choices that satisfy all conflicts you saw:
# - protobuf 5.x satisfies: opentelemetry-proto (>=5,<7), a2a-sdk (>=5.29.5), ydf (>=5.29.1), grpcio-status (>=5.26,<6)
# - rich <14 for bigframes
# - click != 8.3.0 for ray
# - cryptography <44 and pyOpenSSL <= 24.2.1 for pydrive2
# - gcsfs requires fsspec==2025.3.0
# - bigframes also wants google-cloud-bigquery-storage >=2.30,<3
fixes = [
    "protobuf==5.29.2",
    "google-cloud-bigquery-storage>=2.30,<3.0",
    "rich==13.7.1",
    "click==8.1.7",
    "cryptography<44",
    "pyOpenSSL<=24.2.1",
    "fsspec==2025.3.0",
    "gcsfs==2025.3.0",
]

try:
    pip_install(fixes)
except Exception as e:
    print("pip fix failed:", e)

# (Optional) if you *never* use these in your notebook, you can uninstall to avoid future churn:
# subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "ray", "bigframes", "pydrive2"])

# quick sanity print
import pkg_resources as pr
wanted = ["protobuf","google-cloud-bigquery-storage","rich","click","cryptography","pyOpenSSL","fsspec","gcsfs"]
print({d.project_name: d.version for d in pr.working_set if d.project_name in wanted})
# --------------------------------------------------

# ============================================
# Mercor AI Text Detection — OOF-first 2×Transformer, 5-seed, SGKFold-by-topic
#   Models:
#     - microsoft/deberta-v3-small
#     - fakespot-ai/roberta-base-ai-text-detection-v1
# Pipeline:
#   1) Same StratifiedGroupKFold by `topic` for BOTH models
#   2) For each fold: train 5 seeds per model -> average fold preds (OOF + test)
#   3) Collect full OOF per model; compute per-fold & mean CV AUC
#   4) Blends:
#        a) Equal probability mean of model means (diagnostic)
#        b) LOGIT (odds) blend with weight w∈[0,1] tuned on OOF to maximize AUC
#   5) Save:
#        - oof_deberta.csv, oof_fakespot.csv
#        - test_deberta.csv, test_fakespot.csv
#        - submission_deberta_meanseed.csv
#        - submission_fakespot_meanseed.csv
#        - submission_prob_mean_equal.csv
#        - submission_logit_weighted_w{w}.csv
#        - cv_scores.csv (per-method)
#        - run_hparams.json (all tuned/used hyperparams incl. best w)
# ============================================

import os, gc, json, math, random, warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Prefer StratifiedGroupKFold for leakage-safe topic-aware CV
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGF = True
except Exception:
    HAS_SGF = False

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

warnings.filterwarnings("ignore")

# --------------------------
# Config
# --------------------------
DATA_PATH = "/kaggle/input/mercor-ai-detection"
TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
TEST_CSV  = os.path.join(DATA_PATH, "test.csv")

MODELS = [
    "microsoft/deberta-v3-small",
    "fakespot-ai/roberta-base-ai-text-detection-v1",
]

N_SPLITS   = 5
SEEDS      = [13, 21, 42, 87, 123]   # 5-seed rotation
MAX_LEN    = 512
LR         = 2e-5
EPOCHS     = 7
BATCH      = 16
FP16       = torch.cuda.is_available()
OUT_DIR    = "./oof_blend_runs"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------
# Helpers
# --------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_text(topic, answer) -> str:
    t = "" if pd.isna(topic) else str(topic).strip()
    a = "" if pd.isna(answer) else str(answer).strip()
    # Simple concat works well here; you can switch to "TOPIC: ... ANSWER: ..." if preferred.
    return f"{t} {a}".strip()

def tokenize_fn(tokenizer, max_len):
    def _fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )
    return _fn

def make_folds(train_df: pd.DataFrame, n_splits: int, random_state: int = 777) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Same folds for both models and all seeds."""
    y = train_df["is_cheating"].values
    if HAS_SGF and "topic" in train_df.columns:
        groups = train_df["topic"].astype(str).values
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(splitter.split(train_df, y, groups))
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(splitter.split(train_df, y))
    return splits

def compute_auc(y_true, y_prob) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return 0.5

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def logit(p):  return np.log((p + 1e-12) / (1 - p + 1e-12))

# --------------------------
# Data
# --------------------------
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

assert {"id", "topic", "answer", "is_cheating"}.issubset(train.columns), "train.csv missing required columns"
assert {"id", "topic", "answer"}.issubset(test.columns), "test.csv missing required columns"

train["text"] = [make_text(t, a) for t, a in zip(train["topic"], train["answer"])]
test["text"]  = [make_text(t, a) for t, a in zip(test["topic"], test["answer"])]
y = train["is_cheating"].values.astype(int)

folds = make_folds(train, N_SPLITS, random_state=777)
print(f"Using {'StratifiedGroupKFold' if HAS_SGF and 'topic' in train.columns else 'StratifiedKFold'} with {N_SPLITS} folds.")

# --------------------------
# Core training per model
# --------------------------
def train_model_get_oof_and_test(model_name: str) -> Dict:
    """
    For a given model:
      - For each fold:
          - For each seed in SEEDS:
              * Train on train_idx, validate on val_idx
              * Predict val → record fold AUC
              * Predict test
          - Average seed predictions for this fold’s val rows → write into final OOF
      - Average seed predictions across all folds for test → final test probs
    Returns:
      dict with keys:
        "oof"  : np.ndarray shape (n_train,)
        "test" : np.ndarray shape (n_test,)
        "seed_fold_aucs": list of dict rows
    """
    print(f"\n=== {model_name} ===")
    oof = np.zeros(len(train), dtype=np.float32)

    # to average test across seeds: collect per-seed test predictions across folds
    # we'll collect test per seed (accumulated over folds, averaged by N_SPLITS later)
    test_seed_accum = {s: np.zeros(len(test), dtype=np.float32) for s in SEEDS}

    seed_fold_rows = []  # diagnostics rows

    # Prepare tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    for f, (tr_idx, va_idx) in enumerate(folds, start=1):
        print(f"[{model_name}] Fold {f}/{N_SPLITS} (val size={len(va_idx)})")

        # fold-level containers for seed averaging on val
        val_seed_preds = []

        # Build datasets for this fold ONCE (the text is fixed)
        tr_df = train.loc[tr_idx, ["text", "is_cheating"]].rename(columns={"is_cheating": "labels"}).reset_index(drop=True)
        va_df = train.loc[va_idx, ["text", "is_cheating"]].rename(columns={"is_cheating": "labels"}).reset_index(drop=True)
        te_df = test[["id", "text"]].copy()

        ds_tr = Dataset.from_pandas(tr_df)
        ds_va = Dataset.from_pandas(va_df)
        ds_te = Dataset.from_pandas(te_df)

        tok = tokenize_fn(tokenizer, MAX_LEN)
        ds_tr = ds_tr.map(tok, batched=True, load_from_cache_file=False)
        ds_va = ds_va.map(tok, batched=True, load_from_cache_file=False)
        ds_te = ds_te.map(tok, batched=True, load_from_cache_file=False)

        ds_tr.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        ds_va.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        ds_te.set_format(type="torch", columns=["input_ids", "attention_mask"])

        for seed in SEEDS:
            set_all_seeds(seed)
            # fresh model per seed
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            try:
                model.config.use_cache = False
                if hasattr(model, "model") and hasattr(model.model, "config"):
                    model.model.config.use_cache = False
            except Exception:
                pass

            out_dir_seed = os.path.join(OUT_DIR, f"tmp_{model_name.split('/')[-1]}_f{f}_s{seed}")
            args = TrainingArguments(
                output_dir=out_dir_seed,
                eval_strategy="epoch",
                save_strategy="no",             # no checkpoints
                load_best_model_at_end=False,   # because we don't save
                metric_for_best_model="roc_auc",
                learning_rate=LR,
                per_device_train_batch_size=BATCH,
                per_device_eval_batch_size=BATCH,
                num_train_epochs=EPOCHS,
                fp16=FP16,
                logging_strategy="epoch",
                report_to="none",
                seed=seed,
            )

            def compute_metrics(p):
                logits, labels = p
                probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
                return {"roc_auc": roc_auc_score(labels, probs)}

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=ds_tr,
                eval_dataset=ds_va,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
                tokenizer=tokenizer
            )

            # Train → predict val & test
            trainer.train()

            # VAL
            val_pred = trainer.predict(ds_va)
            logits = val_pred.predictions
            val_probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
            val_auc = compute_auc(va_df["labels"].values, val_probs)
            seed_fold_rows.append({
                "model": model_name,
                "fold": f,
                "seed": seed,
                "val_auc": val_auc
            })
            print(f"  - seed {seed}: fold AUC={val_auc:.6f}")
            val_seed_preds.append(val_probs)

            # TEST
            test_logits = trainer.predict(ds_te).predictions
            test_probs  = torch.softmax(torch.tensor(test_logits), dim=1)[:, 1].numpy()
            test_seed_accum[seed] += (test_probs / N_SPLITS)  # average over folds

            # cleanup per seed
            try:
                shutil.rmtree(out_dir_seed, ignore_errors=True)
            except Exception:
                pass
            del trainer, model
            torch.cuda.empty_cache(); gc.collect()

        # seed-mean for this fold's val indices
        val_seed_preds = np.vstack(val_seed_preds)   # (n_seeds, n_val)
        oof[va_idx] = val_seed_preds.mean(axis=0).astype(np.float32)
        print(f"  => Fold {f} (seed-mean) AUC = {compute_auc(va_df['labels'].values, oof[va_idx]):.6f}")

        # cleanup fold datasets
        del ds_tr, ds_va, ds_te, tr_df, va_df, te_df, val_seed_preds
        torch.cuda.empty_cache(); gc.collect()

        # Optional: prune HF cache between folds if you’re tight on space
        try:
            shutil.rmtree("/kaggle/temp/hf", ignore_errors=True)
        except Exception:
            pass

    # Average test across seeds (already averaged over folds above)
    test_probs_by_seed = np.vstack([test_seed_accum[s] for s in SEEDS])  # (n_seeds, n_test)
    test_probs_final   = test_probs_by_seed.mean(axis=0).astype(np.float32)

    return {"oof": oof, "test": test_probs_final, "seed_fold_aucs": seed_fold_rows}


# --------------------------
# Run for all models
# --------------------------
all_results = {}
for m in MODELS:
    res = train_model_get_oof_and_test(m)
    all_results[m] = res

    # Save per-model artifacts
    pd.DataFrame(
        {"id": train["id"], "oof": res["oof"], "y": y}
    ).to_csv(os.path.join(OUT_DIR, f"oof_{m.split('/')[-1]}.csv"), index=False)

    pd.DataFrame(
        {"id": test["id"], "is_cheating": res["test"]}
    ).to_csv(os.path.join(OUT_DIR, f"test_{m.split('/')[-1]}.csv"), index=False)

# --------------------------
# Blends on OOF (to score) + Test (to submit)
# --------------------------
m1, m2 = MODELS
oof_m1, oof_m2 = all_results[m1]["oof"], all_results[m2]["oof"]
tst_m1, tst_m2 = all_results[m1]["test"], all_results[m2]["test"]

# 1) Mean in probability space
oof_mean = 0.5 * (oof_m1 + oof_m2)
tst_mean = 0.5 * (tst_m1 + tst_m2)

# 2) Logit-average (equal weights)
oof_logit_equal = sigmoid(0.5 * (logit(oof_m1) + logit(oof_m2)))
tst_logit_equal = sigmoid(0.5 * (logit(tst_m1) + logit(tst_m2)))

# 3) Logit-average with OOF-tuned weight w in [0,1]
grid = np.linspace(0.0, 1.0, 101)
best_w, best_auc = 0.5, -1
for w in grid:
    oof_blend = sigmoid(w * logit(oof_m1) + (1 - w) * logit(oof_m2))
    auc = compute_auc(y, oof_blend)
    if auc > best_auc:
        best_auc, best_w = auc, float(w)
oof_logit_w = sigmoid(best_w * logit(oof_m1) + (1 - best_w) * logit(oof_m2))
tst_logit_w = sigmoid(best_w * logit(tst_m1) + (1 - best_w) * logit(tst_m2))

# 4) Stacking Logistic Regression (fit on OOF; predict test)
X_oof = np.vstack([oof_m1, oof_m2]).T
X_tst = np.vstack([tst_m1, tst_m2]).T
stack_lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
stack_lr.fit(X_oof, y)
oof_stack_lr = stack_lr.predict_proba(X_oof)[:, 1]
tst_stack_lr = stack_lr.predict_proba(X_tst)[:, 1]

# --------------------------
# Save submissions
# --------------------------
sub_mean = pd.DataFrame({"id": test["id"], "is_cheating": tst_mean})
sub_mean.to_csv("submission_mean.csv", index=False)

sub_leq = pd.DataFrame({"id": test["id"], "is_cheating": tst_logit_equal})
sub_leq.to_csv("submission_logit_equal.csv", index=False)

sub_lw  = pd.DataFrame({"id": test["id"], "is_cheating": tst_logit_w})
sub_lw.to_csv("submission_logit_wbest.csv", index=False)

sub_stk = pd.DataFrame({"id": test["id"], "is_cheating": tst_stack_lr})
sub_stk.to_csv("submission_stack_lr.csv", index=False)

print("\nSaved: submission_mean.csv, submission_logit_equal.csv, submission_logit_wbest.csv, submission_stack_lr.csv")

# --------------------------
# Score OOF and log diagnostics
# --------------------------
scores = []
scores.append({"blend": "mean_prob",        "oof_auc": compute_auc(y, oof_mean)})
scores.append({"blend": "logit_equal",      "oof_auc": compute_auc(y, oof_logit_equal)})
scores.append({"blend": f"logit_wbest({best_w:.2f})", "oof_auc": compute_auc(y, oof_logit_w)})
scores.append({"blend": "stack_lr",         "oof_auc": compute_auc(y, oof_stack_lr)})

cv_scores_df = pd.DataFrame(scores).sort_values("oof_auc", ascending=False)
cv_scores_df.to_csv("cv_scores.csv", index=False)
print("\nOOF AUCs:\n", cv_scores_df)

# Per-seed fold diagnostics
seed_rows = []
for m in MODELS:
    seed_rows += all_results[m]["seed_fold_aucs"]
pd.DataFrame(seed_rows).to_csv("seed_fold_aucs.csv", index=False)
print("Saved: seed_fold_aucs.csv")

# --------------------------
# Save run hyperparameters
# --------------------------
run_hparams = {
    "models": MODELS,
    "n_splits": N_SPLITS,
    "seeds": SEEDS,
    "max_len": MAX_LEN,
    "epochs": EPOCHS,
    "learning_rate": LR,
    "batch_size": BATCH,
    "fp16": FP16,
    "folds_random_state": 777,
    "blend_methods": [
        "mean_prob",
        "logit_equal",
        "logit_wbest_grid_0.01",
        "stacking_logistic_regression"
    ],
    "logit_wbest": best_w
}
with open("run_hparams.json", "w") as f:
    json.dump(run_hparams, f, indent=2)
print("Saved: run_hparams.json")

# --------------------------
# Final note
# --------------------------
print("\nDone. You can pick any submission*.csv to submit. "
      "Prefer the one with the best OOF AUC in cv_scores.csv.")
