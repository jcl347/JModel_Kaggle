# ============================================================
# Cell 0: Protobuf compatibility patch for Kaggle / papermill
# ============================================================
try:
    from google.protobuf import message_factory as _message_factory

    # Newer protobuf versions removed MessageFactory.GetPrototype.
    # Some libs in the environment still call it, so we alias it
    # to the newer GetMessageClass implementation.
    if hasattr(_message_factory, "MessageFactory") and \
       not hasattr(_message_factory.MessageFactory, "GetPrototype"):

        def _get_prototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _get_prototype
        print("Patched google.protobuf.MessageFactory.GetPrototype -> GetMessageClass")

except Exception as e:
    # If anything goes wrong, just print and continue; your pipeline doesn't depend on this.
    print("Protobuf patch not applied:", repr(e))

import os
import gc
import random
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
)

from tqdm.auto import tqdm

# ============================================================
# Global config (tuned)
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_SPLITS = 5

MAX_LEN = 512

# 🔧 Tuned LRs
LR_ROBERTA = 2e-5          # lower for the big model
LR_FAKESPOT = 3e-5         # slightly higher for the smaller model

NUM_EPOCHS = 25            # early stopping will usually stop sooner
WARMUP_RATIO = 0.1         # ~10% of total steps as warmup
PATIENCE = 5
BATCH_TRAIN = 8
BATCH_VAL = 16
WEIGHT_DECAY = 0.01        # L2 regularization
MAX_GRAD_NORM = 1.0        # gradient clipping

ROBERTA_ID = "FacebookAI/roberta-large"
FAKESPOT_ID = "fakespot-ai/roberta-base-ai-text-detection-v1"

print(f"DEVICE: {DEVICE}")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ============================================================
# 1. Load data & build RoBERTa-style prompt
# ============================================================

train_df = pd.read_csv("/kaggle/input/mercor-ai-detection/train.csv")
test_df  = pd.read_csv("/kaggle/input/mercor-ai-detection/test.csv")

print("Train shape:", train_df.shape)
print("Test shape: ", test_df.shape)
print("\nClass distribution:")
print(train_df["is_cheating"].value_counts(normalize=True))

def prompt(topic: str, answer: str) -> str:
    return (
        "Predict if AI generated text was used:\n"
        f"Topic:{topic}\n"
        f"Answer:{answer}"
    )

train_df["input"] = train_df.apply(
    lambda row: prompt(row["topic"], row["answer"]), axis=1
)
test_df["input"] = test_df.apply(
    lambda row: prompt(row["topic"], row["answer"]), axis=1
)

train_df["is_cheating"] = train_df["is_cheating"].astype(int)
y_train = train_df["is_cheating"].values

# ============================================================
# 2. Dataset classes & folds
# ============================================================

class AIDetectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": label,
        }


class AIDetectTestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }


def get_stratified_folds(df: pd.DataFrame, n_splits: int, target: str = "is_cheating"):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    y = df[target].values
    idx = np.arange(len(df))
    return list(skf.split(idx, y))


folds = get_stratified_folds(train_df, N_SPLITS, target="is_cheating")

# ============================================================
# 3. Train / eval loops
# ============================================================

def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scheduler,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device).long()

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)

        loss.backward()
        # 🔧 gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_one_epoch(
    model,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
):
    """
    Returns:
      avg_val_loss, accuracy, auc_roc, val_probs (N_val, 2)
    """
    model.eval()
    total_loss = 0.0

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).long()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            all_preds.extend(preds)

    avg_loss = total_loss / len(loader)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    except Exception:
        auc = 0.5

    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, auc, all_probs

# ============================================================
# 4. Generic CV trainer for any transformer model
# ============================================================

def train_cv_transformer(model_id: str, model_prefix: str, lr: float):
    """
    Train a transformer (RoBERTa-large or Fakespot RoBERTa) with:
      - 5-fold StratifiedKFold
      - class-weighted CE loss
      - cosine warmup scheduler
      - early stopping on val AUC ROC

    Returns:
      tokenizer, checkpoint_paths, fold_best_aucs, oof_probs
    """
    print("\n" + "=" * 70)
    print(f"TRAINING CV MODEL: {model_id}  ({model_prefix})")
    print("=" * 70 + "\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    fold_best_aucs: List[float] = []
    checkpoint_paths: List[str] = []

    # OOF P(class=1)
    oof_probs = np.zeros(len(train_df), dtype=np.float32)

    for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
        print(f"\n--- {model_prefix} | Fold {fold_idx}/{N_SPLITS} ---")

        df_tr = train_df.iloc[tr_idx].reset_index(drop=True)
        df_va = train_df.iloc[va_idx].reset_index(drop=True)

        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=df_tr["is_cheating"].values,
        )
        weights_tensor = torch.tensor(weights, dtype=torch.float, device=DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)

        train_dataset = AIDetectDataset(
            df_tr["input"], df_tr["is_cheating"], tokenizer, max_len=MAX_LEN
        )
        val_dataset = AIDetectDataset(
            df_va["input"], df_va["is_cheating"], tokenizer, max_len=MAX_LEN
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_TRAIN,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_VAL,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=2,
            pin_memory=True,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=2
        )
        model.to(DEVICE)

        total_steps = len(train_loader) * NUM_EPOCHS
        num_warmup_steps = max(1, int(WARMUP_RATIO * total_steps))

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )

        best_auc = -np.inf
        best_path = f"{model_prefix}_fold{fold_idx}.pth"
        trigger = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, loss_fn, scheduler, DEVICE
            )
            val_loss, val_acc, val_auc, val_probs = evaluate_one_epoch(
                model, val_loader, loss_fn, DEVICE
            )

            print(
                f"{model_prefix} | Fold {fold_idx} | "
                f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
                f"TrainLoss={train_loss:.4f} | "
                f"ValLoss={val_loss:.4f} | "
                f"AUC={val_auc:.6f} | "
                f"Acc={val_acc:.4f}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                trigger = 0
                torch.save(model.state_dict(), best_path)
                oof_probs[va_idx] = val_probs[:, 1]
                print(f"  -> New best model saved: {best_path} (AUC={best_auc:.6f})")
            else:
                trigger += 1

            if trigger >= PATIENCE or np.isclose(val_auc, 1.0):
                print("  -> EARLY STOPPING")
                break

        fold_best_aucs.append(best_auc)
        checkpoint_paths.append(best_path)

        del model, train_loader, val_loader, train_dataset, val_dataset
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{model_prefix} fold best AUCs: {fold_best_aucs}")
    print(f"{model_prefix} mean AUC:       {np.mean(fold_best_aucs):.6f}")

    return tokenizer, checkpoint_paths, fold_best_aucs, oof_probs

# ============================================================
# 5. Train both models with tuned hyperparameters
# ============================================================

set_seed(SEED)

roberta_tokenizer, roberta_ckpts, roberta_fold_aucs, roberta_oof = train_cv_transformer(
    ROBERTA_ID, model_prefix="roberta_large", lr=LR_ROBERTA
)

fakespot_tokenizer, fakespot_ckpts, fakespot_fold_aucs, fakespot_oof = train_cv_transformer(
    FAKESPOT_ID, model_prefix="fakespot_roberta", lr=LR_FAKESPOT
)

# ============================================================
# 6. OOF AUCs + simple blend + logistic stacker
# ============================================================

y = train_df["is_cheating"].values

auc_roberta_oof = roc_auc_score(y, roberta_oof)
auc_fakespot_oof = roc_auc_score(y, fakespot_oof)

print("\n" + "=" * 70)
print(f"RoBERTa-large OOF AUC: {auc_roberta_oof:.6f}")
print(f"Fakespot OOF AUC:      {auc_fakespot_oof:.6f}")

# Grid-search linear blend
best_w = None
best_auc = -1.0
for w in np.linspace(0.0, 1.0, 101):
    blended = w * roberta_oof + (1.0 - w) * fakespot_oof
    auc = roc_auc_score(y, blended)
    if auc > best_auc:
        best_auc = auc
        best_w = w

print(f"\nBest linear OOF blend:")
print(f"  w_roberta = {best_w:.2f}, w_fakespot = {1-best_w:.2f}")
print(f"  Ensemble OOF AUC = {best_auc:.6f}")

# 🔧 Logistic regression stacker on top of OOF preds
X_stack = np.column_stack([roberta_oof, fakespot_oof])
stacker = LogisticRegression(solver="lbfgs")
stacker.fit(X_stack, y)
stack_oof = stacker.predict_proba(X_stack)[:, 1]
auc_stack = roc_auc_score(y, stack_oof)
print(f"\nLogistic stacker OOF AUC: {auc_stack:.6f}")

use_stacker = auc_stack > best_auc
print(f"\nUsing stacker for test preds? {use_stacker}")

print("=" * 70 + "\n")

# ============================================================
# 7. Test-time CV inference for each family
# ============================================================

def predict_test_cv(model_id, tokenizer, checkpoint_paths, test_df, model_prefix=""):
    print("\n" + "=" * 70)
    print(f"TEST PREDICTION: {model_id} ({model_prefix})")
    print("=" * 70 + "\n")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_dataset = AIDetectTestDataset(test_df["input"], tokenizer, max_len=MAX_LEN)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_VAL,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True,
    )

    all_fold_probs = []

    for fold_idx, ckpt_path in enumerate(checkpoint_paths, start=1):
        print(f"  -> Inference with fold {fold_idx} checkpoint: {ckpt_path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=2
        )
        state_dict = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        fold_probs = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                fold_probs.append(probs[:, 1])

        fold_probs = np.concatenate(fold_probs, axis=0)
        all_fold_probs.append(fold_probs)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    all_fold_probs = np.stack(all_fold_probs, axis=0)  # (n_folds, n_test)
    mean_probs = all_fold_probs.mean(axis=0)
    print(f"\nDone. Shape of mean_probs: {mean_probs.shape}")
    return mean_probs


roberta_test = predict_test_cv(
    ROBERTA_ID, roberta_tokenizer, roberta_ckpts, test_df, model_prefix="roberta_large"
)
fakespot_test = predict_test_cv(
    FAKESPOT_ID, fakespot_tokenizer, fakespot_ckpts, test_df, model_prefix="fakespot_roberta"
)

# ============================================================
# 8. Final ensemble + submissions
# ============================================================

if use_stacker:
    # non-linear stacking in probability space
    X_test_stack = np.column_stack([roberta_test, fakespot_test])
    final_test = stacker.predict_proba(X_test_stack)[:, 1]
else:
    # fallback to best linear blend
    w_roberta = best_w
    w_fakespot = 1.0 - best_w
    final_test = w_roberta * roberta_test + w_fakespot * fakespot_test

sub_ensemble = pd.DataFrame(
    {
        "id": test_df["id"],
        "is_cheating": final_test,
    }
)
sub_ensemble.to_csv("submission_ensemble_roberta_fakespot.csv", index=False)
print("\n✅ Wrote ensemble submission: submission_ensemble_roberta_fakespot.csv")

sub_roberta = pd.DataFrame(
    {
        "id": test_df["id"],
        "is_cheating": roberta_test,
    }
)
sub_roberta.to_csv("roberta_large_only.csv", index=False)

sub_fakespot = pd.DataFrame(
    {
        "id": test_df["id"],
        "is_cheating": fakespot_test,
    }
)
sub_fakespot.to_csv("fakespot_only.csv", index=False)

print("Also wrote:")
print("  roberta_large_only.csv")
print("  fakespot_only.csv")

print("\n🎉 5-fold RoBERTa-large + Fakespot ensemble (tuned) complete!")
