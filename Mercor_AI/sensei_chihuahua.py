# mercor_tde_sdae_stack.py
# Tiny-data, low-leakage text pipeline with stacking and optional transformer embeddings export.
# - Per-fold TF-IDF/SVD fit on TRAIN SPLIT only
# - StratifiedGroupKFold by topic (if available)
# - LightGBM tuned via TDE (variance-penalized inner CV) with safe fallback
# - Base learners: LR (std), LGBM (TDE), SDAE, XGBoost (if available), RandomForest
# - Per-fold Mutual Information feature screening on train only
# - Stack with CV-calibrated logistic regression on OOF base preds
# - Aggregated LGBM feature importance (FIXED printing)
# - Silenced LightGBM warnings
# - Random seed rotation (printed)
# - Optional transformer embedding computation & CSV export, or load precomputed embeddings
# - Final output: submission.csv with is_cheating strictly {0,1}; submission_probs.csv for diagnostics

# FINAL SCORE: .89 AUC

import os, gc, re, io, sys, json, time, math, argparse, random, contextlib, warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGF = True
except Exception:
    HAS_SGF = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# General helpers
# =========================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

@contextlib.contextmanager
def silence_lightgbm(patterns=None):
    """Filter noisy LightGBM messages from stdout/stderr."""
    if patterns is None:
        patterns = [
            "No further splits with positive gain",
            "Stopped training because there are no more leaves that meet the split requirements",
        ]
    class FilterIO:
        def __init__(self, orig): self._orig = orig
        def write(self, s):
            try:
                if any(p in s for p in patterns): return len(s)
            except Exception:
                pass
            return self._orig.write(s)
        def flush(self):
            try: return self._orig.flush()
            except Exception: pass
    out0, err0 = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = FilterIO(out0), FilterIO(err0)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = out0, err0

def quick_eda(df: pd.DataFrame) -> Dict:
    lengths = df["answer"].fillna("").str.len().values
    qu = np.quantile(lengths, [0, .01, .05, .25, .5, .75, .95, .99, 1.0])
    return dict(
        n_rows=int(len(df)),
        pos_rate=float(df["is_cheating"].mean()) if "is_cheating" in df.columns else None,
        length_quantiles={ "q00": float(qu[0]), "q01": float(qu[1]), "q05": float(qu[2]),
                           "q25": float(qu[3]), "q50": float(qu[4]), "q75": float(qu[5]),
                           "q95": float(qu[6]), "q99": float(qu[7]), "q100": float(qu[8])},
        n_topics=int(df["topic"].nunique()) if "topic" in df.columns else None
    )

_CONNECTORS = [
    'in conclusion','in summary','furthermore','moreover','additionally','however',
    'therefore','thus','consequently','as a result','on the other hand',
    'for instance','for example','it is important to note','it is worth noting','that being said'
]

def lexical_diversity(text: str) -> float:
    toks = str(text).lower().split()
    return (len(set(toks)) / len(toks)) if toks else 0.0

def shannon_char_entropy(text: str) -> float:
    s = str(text)
    if not s: return 0.0
    vals, cnts = np.unique(list(s), return_counts=True)
    p = cnts / cnts.sum()
    return float(-(p * np.log2(p + 1e-12)).sum())

def _split_sents(text: str) -> List[str]:
    return [seg for seg in re.split(r'[.!?]+', text) if seg is not None]

def sentence_len_variance(text: str) -> float:
    sents = [s for s in _split_sents(str(text)) if s.strip()]
    if len(sents) <= 1: return 0.0
    lens = [len(s.split()) for s in sents]
    return float(np.var(lens))

def connector_density(text: str) -> float:
    low = str(text).lower()
    wc = max(1, len(low.split()))
    return sum(1 for ph in _CONNECTORS if ph in low) / wc

def topic_coherence_shared(topic: str, answer: str, vec) -> float:
    if not str(topic).strip() or not str(answer).strip():
        return 0.0
    v1 = vec.transform([topic])
    v2 = vec.transform([answer])
    return float(cosine_similarity(v1, v2)[0, 0])

# =========================
# Optional external embeddings: loader
# =========================
def load_external_embeddings(train_df: pd.DataFrame,
                             test_df: pd.DataFrame,
                             path_one: Optional[str] = None,
                             path_train: Optional[str] = None,
                             path_test: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    try:
        if path_one and os.path.exists(path_one):
            ext = pd.read_csv(path_one)
            assert "id" in ext.columns
            ext_cols = [c for c in ext.columns if c != "id"]
            ext = ext.set_index("id")
            tr = ext.loc[train_df["id"]].reset_index(drop=True)
            te = ext.loc[test_df["id"]].reset_index(drop=True)
            names = [f"ext_{c}" for c in ext_cols]
            tr.columns = names; te.columns = names
            return tr.values.astype(np.float32), te.values.astype(np.float32), names
        elif path_train and path_test and os.path.exists(path_train) and os.path.exists(path_test):
            tr = pd.read_csv(path_train); te = pd.read_csv(path_test)
            assert "id" in tr.columns and "id" in te.columns
            tr = tr.set_index("id").loc[train_df["id"]].reset_index(drop=True)
            te = te.set_index("id").loc[test_df["id"]].reset_index(drop=True)
            names = [f"ext_{i}" for i in range(tr.shape[1])]
            return tr.values.astype(np.float32), te.values.astype(np.float32), names
        else:
            print("[ExtEmb] No external embeddings found. Skipping.")
            return None, None, []
    except Exception as e:
        print(f"[ExtEmb] Skipped due to error: {e}")
        return None, None, []

# =========================
# Optional transformer embedding computation & export
# =========================
def _try_import_transformers():
    try:
        from transformers import AutoTokenizer, AutoModel
        return AutoTokenizer, AutoModel
    except Exception:
        return None, None

@torch.no_grad()
def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def compute_and_save_embeddings(train_df: pd.DataFrame,
                                test_df: pd.DataFrame,
                                model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                                out_all_in_one: Optional[str] = None,
                                out_train: Optional[str] = None,
                                out_test: Optional[str] = None,
                                batch_size: int = 16,
                                max_length: int = 512,
                                use_fp16: bool = True) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    AutoTokenizer, AutoModel = _try_import_transformers()
    if AutoTokenizer is None:
        print("[Emb] transformers not available; skipping embedding computation.")
        return None, None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        print(f"[Emb] Could not load model '{model_name}': {e}. Skipping embeddings.")
        return None, None, None

    model.eval()
    model.to(DEVICE)
    if use_fp16 and DEVICE == "cuda":
        model.half()

    @torch.no_grad()
    def _embed_texts(texts: List[str]) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=max_length,
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(DEVICE)
            attn = enc["attention_mask"].to(DEVICE)
            if "token_type_ids" in enc:
                enc.pop("token_type_ids", None)

            outputs = model(input_ids=input_ids, attention_mask=attn)
            last_hidden = outputs.last_hidden_state
            pooled = _mean_pool(last_hidden, attn)
            vecs.append(pooled.detach().float().cpu().numpy())
        return np.vstack(vecs)

    print(f"[Emb] Computing embeddings with '{model_name}' on {DEVICE}...")
    tr_vec = _embed_texts(train_df["answer"].fillna("").astype(str).tolist())
    te_vec = _embed_texts(test_df["answer"].fillna("").astype(str).tolist())
    dim = tr_vec.shape[1]

    cols = [f"embed_{i}" for i in range(dim)]
    tr_out = pd.DataFrame({"id": train_df["id"].values})
    te_out = pd.DataFrame({"id": test_df["id"].values})
    for i, c in enumerate(cols):
        tr_out[c] = tr_vec[:, i]
        te_out[c] = te_vec[:, i]

    written_all = written_tr = written_te = None

    if out_all_in_one:
        all_df = pd.concat([tr_out, te_out], ignore_index=True)
        all_df.to_csv(out_all_in_one, index=False)
        written_all = out_all_in_one
        print(f"[Emb] Wrote combined embeddings: {out_all_in_one}")
    else:
        if out_train:
            tr_out.to_csv(out_train, index=False)
            written_tr = out_train
            print(f"[Emb] Wrote train embeddings: {out_train}")
        if out_test:
            te_out.to_csv(out_test, index=False)
            written_te = out_test
            print(f"[Emb] Wrote test embeddings: {out_test}")

    return written_all, written_tr, written_te

# =========================
# Per-fold Feature Maker (fit on TRAIN SPLIT ONLY)
# =========================
class FeatureMakerFold:
    def __init__(self, svd_word=384, svd_char=192, word_max=3000, char_max=2000):
        self.word_vec = TfidfVectorizer(
            max_features=word_max, ngram_range=(1,3),
            min_df=2, max_df=0.95, sublinear_tf=True, stop_words='english'
        )
        self.char_vec = TfidfVectorizer(
            max_features=char_max, analyzer='char_wb',
            ngram_range=(3,5), min_df=2, sublinear_tf=True
        )
        self.svd_word = TruncatedSVD(n_components=svd_word, random_state=42)
        self.svd_char = TruncatedSVD(n_components=svd_char, random_state=42)
        self.tfidf_shared = TfidfVectorizer(max_features=2000, ngram_range=(1,2), stop_words='english')
        self.scaler_meta = StandardScaler()
        self.meta_names  = ["len","wc","lex_div","char_entropy","sent_var","conn_density","topic_coherence"]
        self.word_names  = [f"wsvd_{i}" for i in range(svd_word)]
        self.char_names  = [f"csvd_{i}" for i in range(svd_char)]
        self._fitted = False

    def _build_meta(self, df: pd.DataFrame) -> np.ndarray:
        a = df["answer"].fillna("").astype(str)
        meta = pd.DataFrame({
            "len": a.str.len().astype(float),
            "wc":  a.str.split().apply(len).astype(float),
            "lex_div": a.apply(lexical_diversity).astype(float),
            "char_entropy": a.apply(shannon_char_entropy).astype(float),
            "sent_var": a.apply(sentence_len_variance).astype(float),
            "conn_density": a.apply(connector_density).astype(float)
        })
        topics = df["topic"].fillna("").astype(str).values if "topic" in df.columns else np.array([""]*len(df))
        sims = [topic_coherence_shared(t, ans, self.tfidf_shared) for t, ans in zip(topics, a.values)]
        meta["topic_coherence"] = np.array(sims, dtype=float)
        return meta.values

    def fit(self, df_train: pd.DataFrame):
        text = df_train["answer"].fillna("").astype(str).values
        W = self.word_vec.fit_transform(text)
        C = self.char_vec.fit_transform(text)
        self.svd_word.fit(W)
        self.svd_char.fit(C)
        shared_corpus = list(df_train["answer"].fillna("").astype(str).values)
        if "topic" in df_train.columns:
            shared_corpus += list(df_train["topic"].fillna("").astype(str).values)
        self.tfidf_shared.fit(shared_corpus)
        meta_tr = self._build_meta(df_train)
        self.scaler_meta.fit(meta_tr)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        assert self._fitted
        text = df["answer"].fillna("").astype(str).values
        W = self.svd_word.transform(self.word_vec.transform(text))
        C = self.svd_char.transform(self.char_vec.transform(text))
        M = self.scaler_meta.transform(self._build_meta(df))
        X = np.hstack([W, C, M])
        feats = self.word_names + self.char_names + self.meta_names
        return X, feats

# =========================
# MI Screener
# =========================
class MIScreener:
    def __init__(self, keep_ratio: float = 0.7, min_keep: int = 128, max_keep: Optional[int] = None):
        self.keep_ratio = keep_ratio
        self.min_keep = min_keep
        self.max_keep = max_keep
        self.keep_idx_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        mi = mutual_info_classif(X, y, discrete_features=False, random_state=0)
        order = np.argsort(-mi)
        k = max(self.min_keep, int(self.keep_ratio * X.shape[1]))
        if self.max_keep is not None:
            k = min(k, self.max_keep)
        self.keep_idx_ = order[:k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.keep_idx_ is not None
        return X[:, self.keep_idx_]

# =========================
# SDAE
# =========================
class SDAEDataset(Dataset):
    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class SDAE(nn.Module):
    def __init__(self, in_dim, enc_dim=192, mid_dim=128, p_drop=0.35, label_smoothing=0.04):
        super().__init__()
        self.sigma = 0.15
        self.label_smoothing = label_smoothing
        self.bn0 = nn.BatchNorm1d(in_dim)
        self.enc = nn.Sequential(
            nn.Linear(in_dim, enc_dim),
            nn.BatchNorm1d(enc_dim),
            nn.SiLU(),
            nn.Dropout(p_drop),
        )
        self.dec = nn.Linear(enc_dim, in_dim)
        self.cls = nn.Sequential(
            nn.Linear(in_dim + enc_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.SiLU(),
            nn.Dropout(p_drop),
            nn.Linear(mid_dim, 1)
        )
    def forward(self, x, train=True):
        x0 = self.bn0(x)
        x_noise = x0 + self.sigma * torch.randn_like(x0) if train and self.sigma > 0 else x0
        h = self.enc(x_noise)
        recon = self.dec(h)
        logits = self.cls(torch.cat([x0, h], dim=1)).squeeze(1)
        return recon, logits
    def bce_ls(self, logits, targets):
        eps = self.label_smoothing
        targets = targets * (1 - eps) + 0.5 * eps
        return nn.BCEWithLogitsLoss()(logits, targets)

def train_sdae_fold(X_tr, y_tr, X_va, y_va, max_epochs=300, batch=64, lr=1e-3, es_patience=30):
    in_dim = X_tr.shape[1]
    model = SDAE(in_dim=in_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    dl_tr = DataLoader(SDAEDataset(X_tr, y_tr), batch_size=batch, shuffle=True, num_workers=0, pin_memory=True)
    dl_va = DataLoader(SDAEDataset(X_va, y_va), batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

    best_auc, best_state, patience = -1.0, None, 0
    for _epoch in range(1, max_epochs+1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            recon, logits = model(xb, train=True)
            loss = 0.35*nn.MSELoss()(recon, xb) + 0.65*model.bce_ls(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = []
            for xb, _ in dl_va:
                xb = xb.to(DEVICE)
                _, lg = model(xb, train=False)
                pv.append(torch.sigmoid(lg).cpu().numpy())
        p_va = np.concatenate(pv, axis=0).ravel()
        auc = roc_auc_score(y_va, p_va)
        if auc > best_auc + 1e-4:
            best_auc = auc; best_state = {k: v.cpu() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
        if patience >= es_patience: break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        pv = []
        for xb, _ in DataLoader(SDAEDataset(X_va, y_va), batch_size=512, shuffle=False):
            xb = xb.to(DEVICE)
            _, lg = model(xb, train=False)
            pv.append(torch.sigmoid(lg).cpu().numpy())
    return model, np.concatenate(pv, axis=0).ravel(), float(best_auc)

def predict_sdae(model, X):
    dl = DataLoader(SDAEDataset(X, np.zeros(len(X))), batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    model.eval(); preds=[]
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(DEVICE)
            _, lg = model(xb, train=False)
            preds.append(torch.sigmoid(lg).cpu().numpy())
    return np.concatenate(preds, axis=0).ravel()

# =========================
# TDE tuner (variance-penalized LGBM)
# =========================
try:
    from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
    HAS_HYPEROPT = True
except Exception:
    HAS_HYPEROPT = False

@dataclass
class TDEConfig:
    trials: int = 40
    n_splits: int = 5
    var_lambda: float = 0.5
    random_state: int = 99991

def tde_optimize_lgb(train_df: pd.DataFrame, y: np.ndarray, groups: Optional[np.ndarray], cfg: TDEConfig) -> Dict:
    if not HAS_HYPEROPT:
        print("[TDE] hyperopt not available, using safe defaults.")
        return dict(
            n_estimators=900, learning_rate=0.03, max_depth=4, num_leaves=31,
            min_child_samples=45, min_gain_to_split=1e-3, subsample=0.85, colsample_bytree=0.7,
            reg_alpha=0.2, reg_lambda=0.4
        )

    if HAS_SGF and groups is not None:
        splitter = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
        split_iter = list(splitter.split(train_df, y, groups))
    else:
        splitter = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
        split_iter = list(splitter.split(train_df, y))

    space = {
        "n_estimators": hp.quniform("n_estimators", 500, 1600, 50),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.015), np.log(0.08)),
        "max_depth": hp.quniform("max_depth", 3, 6, 1),
        "num_leaves": hp.quniform("num_leaves", 15, 63, 2),
        "min_child_samples": hp.quniform("min_child_samples", 20, 90, 5),
        "min_gain_to_split": hp.loguniform("min_gain_to_split", np.log(1e-4), np.log(5e-2)),
        "subsample": hp.uniform("subsample", 0.6, 0.95),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 0.9),
        "reg_alpha": hp.uniform("reg_alpha", 0.0, 1.0),
        "reg_lambda": hp.uniform("reg_lambda", 0.0, 1.0),
    }

    def obj(params):
        params = {k: (int(v) if k in ("n_estimators","max_depth","num_leaves","min_child_samples") else float(v))
                  for k, v in params.items()}
        fold_aucs = []
        for (tr, va) in split_iter:
            with silence_lightgbm():
                fm = FeatureMakerFold()
                fm.fit(train_df.iloc[tr])
                X_tr, _ = fm.transform(train_df.iloc[tr])
                X_va, _ = fm.transform(train_df.iloc[va])

                lgbm = lgb.LGBMClassifier(
                    **params,
                    random_state=cfg.random_state,
                    device="gpu" if getattr(lgb, "__version__", "0") >= "4.0.0" and torch.cuda.is_available() else "cpu",
                    verbosity=-1
                )
                lgbm.fit(X_tr, y[tr], eval_set=[(X_va, y[va])],
                         callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
                p = lgbm.predict_proba(X_va)[:,1]
                fold_aucs.append(roc_auc_score(y[va], p))
        mean_auc = float(np.mean(fold_aucs))
        std_auc  = float(np.std(fold_aucs))
        score = mean_auc - cfg.var_lambda * std_auc
        return {"loss": -score, "status": STATUS_OK,
                "eval": {"mean_auc": mean_auc, "std_auc": std_auc, "score": score}}

    trials = Trials()
    best = fmin(fn=obj, space=space, algo=tpe.suggest, max_evals=cfg.trials,
                trials=trials, rstate=np.random.default_rng(cfg.random_state))
    best = {k: (int(v) if k in ("n_estimators","max_depth","num_leaves","min_child_samples") else float(v)) for k, v in best.items()}
    best_trial = min(trials.results, key=lambda r: r["loss"])
    ev = best_trial.get("eval", {})
    print(f"[TDE] Best params={best} | mean_auc={ev.get('mean_auc'):.6f} | std={ev.get('std_auc'):.6f} | score={ev.get('score'):.6f}")
    return best

# =========================
# Main training
# =========================
def run(data_dir: str, n_splits: int, n_seed_draws: int, tde_trials: int,
        # Embedding config
        compute_emb: int = 0,
        emb_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        emb_batch: int = 16,
        emb_max_len: int = 512,
        emb_fp16: int = 1,
        emb_out_all_in_one: str = "",
        emb_out_train: str = "",
        emb_out_test: str = "",
        # Precomputed embedding paths (if not computing)
        ext_all_in_one: Optional[str] = None,
        ext_train_path: Optional[str] = None,
        ext_test_path: Optional[str] = None):

    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    assert "id" in train.columns and "id" in test.columns
    assert "is_cheating" in train.columns
    train["answer"] = train["answer"].fillna("")
    test["answer"]  = test["answer"].fillna("")

    print(f"Train: {train.shape} | Test: {test.shape} | Pos rate: {train['is_cheating'].mean():.4f}")
    print("Using StratifiedGroupKFold by topic" if HAS_SGF and "topic" in train.columns else "Using StratifiedKFold")
    print("Length quantiles:", quick_eda(train)["length_quantiles"])

    y_full = train["is_cheating"].values.astype(float)
    groups = train["topic"].values if (HAS_SGF and "topic" in train.columns) else None

    # ----- Embeddings: compute OR load -----
    ext_tr = ext_te = None
    ext_names: List[str] = []

    if int(compute_emb) == 1:
        all_path, tr_path, te_path = compute_and_save_embeddings(
            train, test,
            model_name=emb_model_name,
            out_all_in_one=emb_out_all_in_one or None,
            out_train=emb_out_train or None,
            out_test=emb_out_test or None,
            batch_size=int(emb_batch),
            max_length=int(emb_max_len),
            use_fp16=bool(int(emb_fp16))
        )
        load_one = all_path if all_path else None
        load_tr  = tr_path if tr_path else None
        load_te  = te_path if te_path else None
        ext_tr, ext_te, ext_names = load_external_embeddings(
            train, test,
            path_one=load_one,
            path_train=load_tr,
            path_test=load_te
        )
    else:
        ext_tr, ext_te, ext_names = load_external_embeddings(
            train, test,
            path_one=ext_all_in_one,
            path_train=ext_train_path,
            path_test=ext_test_path
        )
    if ext_tr is not None:
        print(f"[Emb] Using external embeddings with shape train={ext_tr.shape}, test={ext_te.shape}")

    # ----- TDE optimization for LightGBM -----
    tde_cfg = TDEConfig(trials=tde_trials)
    best_lgb_params = tde_optimize_lgb(train, y_full, groups, tde_cfg)

    # ----- Seed rotation -----
    rng = np.random.default_rng(int(time.time()) % (2**31 - 1))
    seeds = list(rng.integers(low=1, high=10**9, size=n_seed_draws))
    print(f"[Seeds] {seeds}")

    all_seed_oof = []   # meta-calibrated oof per seed
    all_seed_te  = []   # meta-calibrated test per seed
    fi_rows = []        # LGBM feature importances

    for seed in seeds:
        print(f"\n=== Seed {int(seed)} ===")
        seed_everything(int(seed))

        if HAS_SGF and groups is not None:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
            split_iter = splitter.split(train, y_full, groups)
        else:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
            split_iter = splitter.split(train, y_full)

        n = len(train); m = len(test)
        oof_LR = np.zeros(n); te_LR = np.zeros(m)
        oof_LGB = np.zeros(n); te_LGB = np.zeros(m)
        oof_SDAE = np.zeros(n); te_SDAE = np.zeros(m)
        oof_XGB = np.zeros(n); te_XGB = np.zeros(m)
        oof_RF  = np.zeros(n); te_RF  = np.zeros(m)

        for f, (tr_idx, va_idx) in enumerate(split_iter, start=1):
            print(f"  Fold {f}/{n_splits}")

            fm = FeatureMakerFold()
            fm.fit(train.iloc[tr_idx])
            X_tr, feat_names = fm.transform(train.iloc[tr_idx])
            X_va, _          = fm.transform(train.iloc[va_idx])
            X_te, _          = fm.transform(test)

            if ext_tr is not None and ext_te is not None:
                X_tr = np.hstack([X_tr, ext_tr[tr_idx]])
                X_va = np.hstack([X_va, ext_tr[va_idx]])
                X_te_full = np.hstack([X_te, ext_te])
                feat_names = feat_names + ext_names
            else:
                X_te_full = X_te

            y_tr, y_va = y_full[tr_idx], y_full[va_idx]

            # MI screening
            mi_screener = MIScreener(keep_ratio=0.7, min_keep=128, max_keep=None)
            mi_screener.fit(X_tr, y_tr)
            keep_idx = mi_screener.keep_idx_
            X_tr = X_tr[:, keep_idx]
            X_va = X_va[:, keep_idx]
            X_te_fold = X_te_full[:, keep_idx]
            feat_names_keep = [feat_names[i] for i in keep_idx]
            print(f"    [MI] kept {X_tr.shape[1]} / {len(feat_names)} features")

            # LR (std)
            scaler = StandardScaler(with_mean=True, with_std=True)
            X_tr_sc = scaler.fit_transform(X_tr)
            X_va_sc = scaler.transform(X_va)
            X_te_sc = scaler.transform(X_te_fold)
            lr = LogisticRegression(C=0.5, solver="saga", max_iter=2000, penalty="l2",
                                    random_state=int(seed), n_jobs=-1)
            lr.fit(X_tr_sc, y_tr)
            oof_LR[va_idx] = lr.predict_proba(X_va_sc)[:,1]
            te_LR += lr.predict_proba(X_te_sc)[:,1] / n_splits

            # LightGBM (TDE)
            with silence_lightgbm():
                lgbm = lgb.LGBMClassifier(
                    **best_lgb_params,
                    random_state=int(seed),
                    device="gpu" if getattr(lgb, "__version__", "0") >= "4.0.0" and torch.cuda.is_available() else "cpu",
                    verbosity=-1
                )
                lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                         callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            oof_LGB[va_idx] = lgbm.predict_proba(X_va)[:,1]
            te_LGB += lgbm.predict_proba(X_te_fold)[:,1] / n_splits

            # FI rows
            try:
                gain = lgbm.booster_.feature_importance(importance_type="gain")
                fi_rows.append(pd.DataFrame({
                    "feature": feat_names_keep,
                    "gain": gain,
                    "seed": int(seed), "fold": f
                }))
            except Exception:
                pass

            # XGB
            if HAS_XGB:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.6,
                    reg_alpha=0.4, reg_lambda=0.8,
                    min_child_weight=3,
                    random_state=int(seed),
                    eval_metric="logloss",
                    tree_method="hist",
                )
                xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                oof_XGB[va_idx] = xgb_model.predict_proba(X_va)[:,1]
                te_XGB += xgb_model.predict_proba(X_te_fold)[:,1] / n_splits
            else:
                oof_XGB[va_idx] = 0.0
                te_XGB += 0.0

            # RF
            rf = RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=3, min_samples_split=8,
                max_features=0.6, n_jobs=-1, random_state=int(seed), bootstrap=True
            )
            rf.fit(X_tr, y_tr)
            oof_RF[va_idx] = rf.predict_proba(X_va)[:,1]
            te_RF += rf.predict_proba(X_te_fold)[:,1] / n_splits

            # SDAE
            sdae, p_va, _best = train_sdae_fold(X_tr, y_tr, X_va, y_va, max_epochs=300, batch=64, lr=1e-3, es_patience=30)
            oof_SDAE[va_idx] = p_va
            te_SDAE += predict_sdae(sdae, X_te_fold) / n_splits
            del sdae
            gc.collect(); torch.cuda.empty_cache()

            fold_auc = roc_auc_score(y_va, (oof_LR[va_idx] + oof_LGB[va_idx] + oof_SDAE[va_idx]) / 3.0)
            print(f"    Fold Ens AUC (LR+LGB+SDAE eq.wt) = {fold_auc:.6f}")

        # Stacking + calibration
        base_oof = np.vstack([oof_LR, oof_LGB, oof_SDAE, oof_XGB, oof_RF]).T
        base_te  = np.vstack([te_LR, te_LGB, te_SDAE, te_XGB, te_RF]).T

        try:
            calibrator = CalibratedClassifierCV(
                base_estimator=LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
                method="sigmoid", cv=5
            )
            calibrator.fit(base_oof, y_full)
            oof_cal = calibrator.predict_proba(base_oof)[:,1]
            te_cal  = calibrator.predict_proba(base_te)[:,1]
            print(f"  Seed {int(seed)} OOF AUC (stack-calibrated): {roc_auc_score(y_full, oof_cal):.6f}")
            all_seed_oof.append(oof_cal)
            all_seed_te.append(te_cal)
        except Exception as e:
            print(f"  [Stack calib fallback: {e}] using LR meta without calibration.")
            meta = LogisticRegression(C=0.5, solver="lbfgs", max_iter=1000)
            meta.fit(base_oof, y_full)
            oof_raw = meta.predict_proba(base_oof)[:,1]
            te_raw  = meta.predict_proba(base_te)[:,1]
            print(f"  Seed {int(seed)} OOF AUC (stack-raw): {roc_auc_score(y_full, oof_raw):.6f}")
            all_seed_oof.append(oof_raw)
            all_seed_te.append(te_raw)

    # Aggregate across seeds
    oof_stack = np.vstack(all_seed_oof)
    te_stack  = np.vstack(all_seed_te)
    oof_mean  = oof_stack.mean(axis=0)
    te_mean   = te_stack.mean(axis=0)
    final_oof_auc = roc_auc_score(y_full, oof_mean)

    print("\n=== Seed Rotation Summary ===")
    for i, s in enumerate(seeds):
        print(f"Seed {int(s)}: OOF AUC={roc_auc_score(y_full, oof_stack[i]):.6f}")
    print(f"Multi-seed OOF AUC (mean): {final_oof_auc:.6f}")

    # ---------------------------
    # Feature importances (FIXED printing)
    # ---------------------------
    try:
        if len(fi_rows):
            fi = pd.concat(fi_rows, ignore_index=True)

            # Coerce types safely
            fi["feature"] = fi["feature"].astype(str)
            fi["gain"] = pd.to_numeric(fi["gain"], errors="coerce").fillna(0.0)

            # Use named aggregations to avoid MultiIndex columns
            agg = (
                fi.groupby("feature", as_index=False)
                  .agg(mean=("gain", "mean"),
                       median=("gain", "median"),
                       sum=("gain", "sum"),
                       count=("gain", "count"))
            ).sort_values("mean", ascending=False)

            top = agg.head(25)

            print("\nTop 25 LightGBM features by mean gain:")
            for rank, row in enumerate(top.itertuples(index=False), start=1):
                # itertuples gives attributes: feature, mean, median, sum, count
                feat = str(row.feature)
                mean_gain = float(row.mean)
                used_in = int(row.count)
                print(f"  {rank:>2}. {feat:<24s} mean_gain={mean_gain:.2f} used_in={used_in}")

            # Optional: highlight meta features
            try:
                meta_names = FeatureMakerFold().meta_names
                meta_in_top = [f for f in top["feature"].tolist() if f in meta_names]
                if meta_in_top:
                    print("\nMeta features appearing in TOP-25:", meta_in_top)
            except Exception:
                pass
        else:
            print("[FI] No feature importance collected (unexpected).")
    except Exception as e:
        print(f"[FI] Skipped feature-importance print due to: {e}")

    # ---------------------------
    # Probability diagnostics
    # ---------------------------
    probs_path = "submission_probs.csv"
    pd.DataFrame(
        {"id": test["id"], "is_cheating": te_mean.clip(1e-6, 1-1e-6)}
    ).to_csv(probs_path, index=False, float_format="%.10f")
    print(f"\n[Info] Wrote diagnostic probabilities: {probs_path}")

    # ---------------------------
    # Robust thresholding from OOF (blend Youden + prevalence) + shrink toward 0.5
    # ---------------------------
    pos_rate = float(train["is_cheating"].mean())
    fpr, tpr, thr = roc_curve(y_full, oof_mean)
    youden_idx = int(np.argmax(tpr - fpr))
    t_youden = float(thr[youden_idx])
    t_prev   = float(np.quantile(oof_mean, 1.0 - pos_rate))
    t_blend  = 0.45*t_youden + 0.45*t_prev + 0.10*0.5
    t_final  = float(np.clip(0.75*t_blend + 0.25*0.5, 0.2, 0.8))
    print(f"[Thresholds] youden={t_youden:.4f}  prev={t_prev:.4f}  final(shrunk)={t_final:.4f}")

    # ---------------------------
    # FINAL binary submission (0/1)
    # ---------------------------
    assert te_mean.shape[0] == len(test), "Pred length must equal test rows"
    test_binary = (te_mean >= t_final).astype(np.int32)

    submission = pd.DataFrame({
        "id": test["id"],          # keep as Series; pandas will handle it
        "is_cheating": test_binary # already ndarray -> no .values
    })

    # Final validations
    assert len(submission) == len(test), "Submission rows must match test.csv"
    uniq_vals = set(np.unique(submission["is_cheating"]))
    assert uniq_vals <= {0, 1}, f"is_cheating must be 0/1 only, got {uniq_vals}"

    final_path = "submission.csv"
    submission.to_csv(final_path, index=False)
    print(f"[OK] Wrote FINAL binary submission: {final_path}")
    print(f"[Stats] positives={submission['is_cheating'].sum()} "
          f"({submission['is_cheating'].mean():.3f} of {len(submission)})")

    # ---------------------------
    # OOF diagnostics
    # ---------------------------
    pd.DataFrame({"id": train["id"], "oof": oof_mean, "y": y_full}).to_csv("oof_stack.csv", index=False)

    print("Sanity: 'id' used only for outputs; never in features.")
    return final_oof_auc
# =========================
# CLI
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="/kaggle/input/mercor-ai-detection")
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=8, help="number of random seeds to draw")
    ap.add_argument("--tde_trials", type=int, default=40, help="hyperopt trials for LightGBM")

    # Embedding controls
    ap.add_argument("--compute_emb", type=int, default=1, help="1=compute embeddings & save CSV(s); 0=skip")
    ap.add_argument("--emb_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--emb_batch", type=int, default=16)
    ap.add_argument("--emb_max_len", type=int, default=512)
    ap.add_argument("--emb_fp16", type=int, default=1, help="1=fp16 on CUDA, else fp32")
    ap.add_argument("--emb_out_all_in_one", type=str, default="combined_embedding.csv", help="path to write combined embeddings CSV")
    ap.add_argument("--emb_out_train", type=str, default="train_emb.csv", help="path to write train embeddings CSV")
    ap.add_argument("--emb_out_test", type=str, default="test_embed.csv", help="path to write test embeddings CSV")

    # Precomputed embedding paths
    ap.add_argument("--ext_all_in_one", type=str, default="", help="CSV with id + embeddings (train+test)")
    ap.add_argument("--ext_train_path", type=str, default="", help="CSV with train embeddings (id-aligned)")
    ap.add_argument("--ext_test_path", type=str, default="", help="CSV with test embeddings (id-aligned)")

    args, _unknown = ap.parse_known_args()

    auc = run(
        data_dir=args.data_dir,
        n_splits=args.splits,
        n_seed_draws=args.seeds,
        tde_trials=args.tde_trials,
        compute_emb=args.compute_emb,
        emb_model_name=args.emb_model_name,
        emb_batch=args.emb_batch,
        emb_max_len=args.emb_max_len,
        emb_fp16=args.emb_fp16,
        emb_out_all_in_one=args.emb_out_all_in_one,
        emb_out_train=args.emb_out_train,
        emb_out_test=args.emb_out_test,
        ext_all_in_one=(args.ext_all_in_one or None),
        ext_train_path=(args.ext_train_path or None),
        ext_test_path=(args.ext_test_path or None),
    )
    print(f"\nFinal Multi-seed OOF AUC: {auc:.6f}")