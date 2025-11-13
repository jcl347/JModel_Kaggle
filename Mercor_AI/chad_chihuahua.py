# ============================================
# Kaggle env fix — pin conflicting dependencies
# Run this cell FIRST, before importing anything else.
# ============================================
import sys, subprocess

def pip_install(pkgs):
    cmd = [sys.executable, "-m", "pip", "install", "-qU"] + pkgs
    print("PIP:", " ".join(pkgs))
    subprocess.check_call(cmd)

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

import pkg_resources as pr
wanted = ["protobuf","google-cloud-bigquery-storage","rich","click","cryptography","pyOpenSSL","fsspec","gcsfs"]
print({d.project_name: d.version for d in pr.working_set if d.project_name in wanted})
# --------------------------------------------------


# ============================================================================
# 1. SETUP AND IMPORTS
# ============================================================================
import os
import gc
import json
import math
import random
import shutil
import inspect
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import re
import warnings
warnings.filterwarnings('ignore')

# For transformers
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
        AutoConfig,
        GPT2LMHeadModel,
        GPT2TokenizerFast,
    )
    from torch.utils.data import Dataset as TorchDataset
    TRANSFORMERS_AVAILABLE = True
    print(f"✓ Transformers available | GPU: {torch.cuda.is_available()}")
except Exception as e:
    TRANSFORMERS_AVAILABLE = False
    print("⚠ Transformers not available:", e)

# HF datasets for Trainer
try:
    from datasets import Dataset as HFDataset
    HF_DATASETS_AVAILABLE = True
except Exception as e:
    HF_DATASETS_AVAILABLE = False
    print("⚠ datasets library not available:", e)

# For potential group k-folds (topic leakage-safe)
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGF = True
except Exception:
    HAS_SGF = False

# For text analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except:
    TEXTBLOB_AVAILABLE = False

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# ============================================================================
# 2. LOAD DATA
# ============================================================================
print("\n📊 Loading data...")
train_df = pd.read_csv('/kaggle/input/gpt-oss-20b-using-llama-cpp-benford-s-law-lb-1-0/train_valid_merged.csv')
test_df = pd.read_csv('/kaggle/input/mercor-ai-detection/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"\nClass distribution:")
print(train_df['is_cheating'].value_counts(normalize=True))
print(f"\nTopic distribution:")
print(train_df['topic'].value_counts())

y_train = train_df['is_cheating'].values.astype(int)

# Text field for transformers (topic + answer)
def make_text(topic, answer) -> str:
    t = "" if pd.isna(topic) else str(topic).strip()
    a = "" if pd.isna(answer) else str(answer).strip()
    return f"{t} {a}".strip()

train_df["text"] = [make_text(t, a) for t, a in zip(train_df["topic"], train_df["answer"])]
test_df["text"]  = [make_text(t, a) for t, a in zip(test_df["topic"], test_df["answer"])]


# ============================================================================
# 3. BENFORD'S LAW FEATURE EXTRACTOR
# ============================================================================
class BenfordAnalyzer:
    """Benford's Law based statistical anomaly detection for AI text"""

    def __init__(self):
        # Benford's Law theoretical distribution for digits 1-9
        self.benford_dist = np.array([np.log10(1 + 1/d) for d in range(1, 10)])

    @staticmethod
    def extract_first_digit(value):
        """Extract first digit from a number"""
        if value <= 0:
            return 1
        return int(str(int(abs(value)))[0])

    def calculate_benford_metrics(self, sequence):
        """
        Calculate Benford's Law deviation metrics.
        Returns:
        - chi_square
        - kl_div
        - entropy
        """
        if len(sequence) == 0:
            return 0, 0, 0

        # Limit sequence size for efficiency
        if len(sequence) > 2000:
            sequence = np.random.choice(sequence, 2000, replace=False)

        # Extract first digits
        first_digits = np.array([self.extract_first_digit(x) for x in sequence if x > 0])

        if len(first_digits) == 0:
            return 0, 0, 0

        # Observed distribution
        observed_dist = np.zeros(9)
        for digit in range(1, 10):
            observed_dist[digit - 1] = np.sum(first_digits == digit) / len(first_digits)

        # Chi-square statistic
        expected_counts = self.benford_dist * len(first_digits)
        observed_counts = observed_dist * len(first_digits)
        chi_square = np.sum((observed_counts - expected_counts) ** 2 / (expected_counts + 1e-8))

        # KL divergence
        kl_div = np.sum(observed_dist * np.log(observed_dist / (self.benford_dist + 1e-8) + 1e-8))

        # Entropy
        entropy = -np.sum(observed_dist * np.log(observed_dist + 1e-8))

        return chi_square, kl_div, entropy

    def extract_benford_features(self, df):
        """Extract all Benford's Law-based features"""
        benford_data = []

        print("Computing Benford's Law features...")
        for text in df['answer']:
            text_str = str(text)
            words = text_str.split()

            # Word length sequence
            word_lengths = np.array([len(w) for w in words])

            # Character counts per word
            char_counts = np.array([len(w) for w in words])

            # Sentence lengths
            sentences = [s.strip() for s in re.split(r'[.!?]+', text_str) if s.strip()]
            sentence_lengths = np.array([len(s.split()) for s in sentences])

            # Punctuation positions
            punct_positions = np.array([i for i, c in enumerate(text_str) if c in '.,;:!?'])

            chi_word, kl_word, ent_word = self.calculate_benford_metrics(word_lengths)
            chi_char, kl_char, ent_char = self.calculate_benford_metrics(char_counts)
            chi_sent, kl_sent, ent_sent = self.calculate_benford_metrics(sentence_lengths)
            chi_punct, kl_punct, ent_punct = self.calculate_benford_metrics(punct_positions)

            benford_data.append({
                'benford_chi_word': chi_word,
                'benford_kl_word': kl_word,
                'benford_entropy_word': ent_word,
                'benford_chi_char': chi_char,
                'benford_kl_char': kl_char,
                'benford_entropy_char': ent_char,
                'benford_chi_sent': chi_sent,
                'benford_kl_sent': kl_sent,
                'benford_entropy_sent': ent_sent,
                'benford_chi_punct': chi_punct,
                'benford_kl_punct': kl_punct,
                'benford_entropy_punct': ent_punct,
            })

        benford_df = pd.DataFrame(benford_data)
        return benford_df


# ============================================================================
# 4. ENHANCED FEATURE ENGINEERING WITH AI-SPECIFIC PATTERNS + BENFORD
# ============================================================================
def enhanced_ai_features(df):
    """Specialized AI text detection features"""
    features = pd.DataFrame()

    # === BASIC METRICS ===
    features['text_length'] = df['answer'].str.len()
    features['word_count'] = df['answer'].str.split().str.len()
    features['avg_word_length'] = features['text_length'] / (features['word_count'] + 1)
    features['char_count'] = df['answer'].apply(lambda x: len(str(x)))

    # === SENTENCE ANALYSIS ===
    features['sentence_count'] = df['answer'].str.count(r'[.!?]+')
    features['avg_sentence_length'] = features['word_count'] / (features['sentence_count'] + 1)
    features['sentence_length_variance'] = df['answer'].apply(
        lambda x: np.var([len(s.split()) for s in re.split(r'[.!?]+', str(x)) if s.strip()]) 
        if len(re.split(r'[.!?]+', str(x))) > 1 else 0
    )
    features['max_sentence_length'] = df['answer'].apply(
        lambda x: max([len(s.split()) for s in re.split(r'[.!?]+', str(x)) if s.strip()] + [0])
    )
    features['min_sentence_length'] = df['answer'].apply(
        lambda x: min([len(s.split()) for s in re.split(r'[.!?]+', str(x)) if s.strip()] + [999])
    )

    # === PUNCTUATION PATTERNS ===
    features['comma_count'] = df['answer'].str.count(',')
    features['semicolon_count'] = df['answer'].str.count(';')
    features['colon_count'] = df['answer'].str.count(':')
    features['exclamation_count'] = df['answer'].str.count('!')
    features['question_count'] = df['answer'].str.count(r'\?')
    features['period_count'] = df['answer'].str.count(r'\.')
    features['quote_count'] = df['answer'].str.count('"')
    features['apostrophe_count'] = df['answer'].str.count("'")
    features['dash_count'] = df['answer'].str.count('-')
    features['ellipsis_count'] = df['answer'].str.count(r'\.\.\.')
    features['parentheses_count'] = df['answer'].str.count(r'[\(\)]')

    features['total_punctuation'] = (features['comma_count'] + features['semicolon_count'] +
                                     features['colon_count'] + features['exclamation_count'] +
                                     features['question_count'] + features['period_count'])
    features['punctuation_ratio'] = features['total_punctuation'] / (features['text_length'] + 1)
    features['punctuation_diversity'] = df['answer'].apply(
        lambda x: len(set([c for c in str(x) if c in '.,;:!?"\'\'-()[]{}']))
    )

    # === VOCABULARY RICHNESS ===
    features['unique_words'] = df['answer'].apply(lambda x: len(set(str(x).lower().split())))
    features['ttr'] = features['unique_words'] / (features['word_count'] + 1)
    features['unique_word_ratio'] = features['unique_words'] / (features['word_count'] + 1)

    features['yules_k'] = df['answer'].apply(
        lambda x: 10000 * (sum([freq**2 for freq in pd.Series(str(x).lower().split()).value_counts().values]) -
                           len(str(x).split())) /
        (len(str(x).split())**2) if len(str(x).split()) > 0 else 0
    )

    # Hapax / dis legomena
    features['hapax_legomena'] = df['answer'].apply(
        lambda x: sum(1 for _, count in pd.Series(str(x).lower().split()).value_counts().items() if count == 1)
    )
    features['dis_legomena'] = df['answer'].apply(
        lambda x: sum(1 for _, count in pd.Series(str(x).lower().split()).value_counts().items() if count == 2)
    )
    features['hapax_ratio'] = features['hapax_legomena'] / (features['word_count'] + 1)

    # === AI-SPECIFIC PATTERNS ===
    ai_connectors = [
        'in conclusion', 'in summary', 'furthermore', 'moreover',
        'additionally', 'however', 'therefore', 'thus', 'consequently',
        'as a result', 'on the other hand', 'for instance', 'for example',
        'it is important to note', 'it is worth noting', 'that being said',
        'in other words', 'specifically', 'namely'
    ]
    features['ai_connector_density'] = df['answer'].apply(
        lambda x: sum(1 for phrase in ai_connectors if phrase in str(x).lower()) /
        (len(str(x).split()) + 1)
    )

    formal_words = ['utilize', 'facilitate', 'implement', 'methodology', 'paradigm',
                    'leverage', 'robust', 'optimal', 'enhance', 'demonstrate']
    features['formal_word_ratio'] = df['answer'].apply(
        lambda x: sum(1 for word in formal_words if word in str(x).lower()) /
        (len(str(x).split()) + 1)
    )

    passive_indicators = ['is made', 'was made', 'is given', 'was given', 'is shown',
                          'was shown', 'is considered', 'was considered', 'by the']
    features['passive_voice_ratio'] = df['answer'].apply(
        lambda x: sum(1 for phrase in passive_indicators if phrase in str(x).lower()) /
        (len(str(x).split()) + 1)
    )

    features['repetitive_starts'] = df['answer'].apply(
        lambda x: len(set([s.split()[0].lower() if s.split() else ''
                           for s in re.split(r'[.!?]+', str(x)) if s.strip()])) /
        (len([s for s in re.split(r'[.!?]+', str(x)) if s.strip()]) + 1)
    )

    features['hapax_dis_ratio'] = df['answer'].apply(
        lambda x: (sum(1 for c in pd.Series(str(x).lower().split()).value_counts().values if c == 1) +
                   sum(1 for c in pd.Series(str(x).lower().split()).value_counts().values if c == 2)) /
        len(str(x).split()) if len(str(x).split()) > 0 else 0
    )

    features['subordinate_ratio'] = df['answer'].str.count(
        r'\b(that|which|who|when|where|while|although|because|if)\b') / (features['word_count'] + 1)

    # Sentiment
    if TEXTBLOB_AVAILABLE:
        features['sentiment_polarity'] = df['answer'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        features['sentiment_subjectivity'] = df['answer'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    else:
        features['sentiment_polarity'] = 0
        features['sentiment_subjectivity'] = 0

    # Reading difficulty
    def flesch_reading_ease(text):
        sentences = len([s for s in re.split(r'[.!?]+', str(text)) if s.strip()])
        words = len(str(text).split())
        syllables = sum([len(re.findall(r'[aeiouy]+', word.lower())) for word in str(text).split()])

        if sentences > 0 and words > 0:
            return 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        return 0

    features['flesch_reading_ease'] = df['answer'].apply(flesch_reading_ease)

    # === WORD LENGTH DISTRIBUTION ===
    features['max_word_length'] = df['answer'].apply(lambda x: max([len(w) for w in str(x).split()] + [0]))
    features['min_word_length'] = df['answer'].apply(lambda x: min([len(w) for w in str(x).split()] + [999]))
    features['word_length_std'] = df['answer'].apply(
        lambda x: np.std([len(w) for w in str(x).split()]) if len(str(x).split()) > 1 else 0
    )

    features['very_short_words'] = df['answer'].apply(lambda x: sum(1 for w in str(x).split() if len(w) <= 2))
    features['short_words'] = df['answer'].apply(lambda x: sum(1 for w in str(x).split() if 3 <= len(w) <= 4))
    features['medium_words'] = df['answer'].apply(lambda x: sum(1 for w in str(x).split() if 5 <= len(w) <= 7))
    features['long_words'] = df['answer'].apply(lambda x: sum(1 for w in str(x).split() if 8 <= len(w) <= 10))
    features['very_long_words'] = df['answer'].apply(lambda x: sum(1 for w in str(x).split() if len(w) > 10))

    features['short_word_ratio'] = features['short_words'] / (features['word_count'] + 1)
    features['long_word_ratio'] = (features['long_words'] + features['very_long_words']) / (features['word_count'] + 1)

    # === CAPITALIZATION ===
    features['capital_letters'] = df['answer'].apply(lambda x: sum(1 for c in str(x) if c.isupper()))
    features['capital_ratio'] = features['capital_letters'] / (features['text_length'] + 1)
    features['all_caps_words'] = df['answer'].apply(
        lambda x: sum(1 for w in str(x).split() if w.isupper() and len(w) > 1)
    )
    features['title_case_words'] = df['answer'].apply(
        lambda x: sum(1 for w in str(x).split() if w.istitle())
    )
    features['title_case_ratio'] = features['title_case_words'] / (features['word_count'] + 1)

    # === SPECIAL CHARACTERS ===
    features['digit_count'] = df['answer'].str.count(r'\d')
    features['digit_ratio'] = features['digit_count'] / (features['text_length'] + 1)
    features['special_char_count'] = df['answer'].apply(
        lambda x: sum(1 for c in str(x) if not c.isalnum() and not c.isspace())
    )

    # === PARAGRAPH STRUCTURE ===
    features['paragraph_count'] = df['answer'].apply(
        lambda x: len([p for p in str(x).split('\n\n') if p.strip()])
    )
    features['avg_paragraph_length'] = features['word_count'] / (features['paragraph_count'] + 1)

    # === REPETITION PATTERNS ===
    features['repeated_words'] = df['answer'].apply(
        lambda x: len([w for w, c in pd.Series(str(x).lower().split()).value_counts().items() if c > 1])
    )
    features['max_word_repetition'] = df['answer'].apply(
        lambda x: pd.Series(str(x).lower().split()).value_counts().max() if len(str(x).split()) > 0 else 0
    )
    features['consecutive_duplicates'] = df['answer'].apply(
        lambda x: sum(1 for i in range(len(str(x).split()) - 1)
                      if str(x).split()[i].lower() == str(x).split()[i+1].lower())
    )

    # === UNIFORMITY METRICS ===
    features['word_length_uniformity'] = df['answer'].apply(
        lambda x: 1 / (np.std([len(w) for w in str(x).split()]) + 0.1)
        if len(str(x).split()) > 1 else 0
    )
    features['sentence_length_uniformity'] = df['answer'].apply(
        lambda x: 1 / (np.std([len(s.split()) for s in re.split(r'[.!?]+', str(x)) if s.strip()]) + 0.1)
        if len([s for s in re.split(r'[.!?]+', str(x)) if s.strip()]) > 1 else 0
    )

    # === BURSTINESS ===
    features['burstiness'] = features['sentence_length_variance'] / (features['avg_sentence_length'] + 1)

    # === TOPIC FEATURES ===
    topic_dummies = pd.get_dummies(df['topic'], prefix='topic')

    return pd.concat([features, topic_dummies], axis=1)


print("\n🔧 Engineering enhanced AI-specific features...")
train_features = enhanced_ai_features(train_df)
test_features = enhanced_ai_features(test_df)

print("\n📊 Adding Benford's Law features...")
benford_analyzer = BenfordAnalyzer()
train_benford = benford_analyzer.extract_benford_features(train_df)
test_benford = benford_analyzer.extract_benford_features(test_df)

train_features = pd.concat([train_features, train_benford], axis=1)
test_features = pd.concat([test_features, test_benford], axis=1)

# Align columns
test_features = test_features.reindex(columns=train_features.columns, fill_value=0)
print(f"✓ Total engineered features: {train_features.shape[1]} (including Benford metrics)")


# ============================================================================
# 5. ENHANCED TEXT FEATURE EXTRACTION
# ============================================================================
print("\n📝 Creating enhanced text representations...")

def enhanced_text_features(train_df, test_df):
    """Improved text feature extraction"""

    # Character-level n-gram
    char_vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=2000,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )

    # Word-level n-gram
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=3000,
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
        stop_words='english'
    )

    # AI pattern vocabulary
    ai_pattern_words = [
        'conclusion', 'summary', 'however', 'therefore', 'moreover',
        'furthermore', 'additionally', 'importantly', 'notably'
    ]
    custom_vectorizer = TfidfVectorizer(
        vocabulary=ai_pattern_words,
        binary=True
    )

    train_char = char_vectorizer.fit_transform(train_df['answer'])
    test_char = char_vectorizer.transform(test_df['answer'])

    train_word = word_vectorizer.fit_transform(train_df['answer'])
    test_word = word_vectorizer.transform(test_df['answer'])

    train_custom = custom_vectorizer.fit_transform(train_df['answer'])
    test_custom = custom_vectorizer.transform(test_df['answer'])

    return (train_char, test_char, train_word, test_word, train_custom, test_custom)

train_char, test_char, train_word, test_word, train_custom, test_custom = enhanced_text_features(train_df, test_df)

X_train_full = np.hstack([
    train_char.toarray(),
    train_word.toarray(),
    train_custom.toarray(),
    train_features.values
])

X_test_full = np.hstack([
    test_char.toarray(),
    test_word.toarray(),
    test_custom.toarray(),
    test_features.values
])

print(f"✓ Total combined features: {X_train_full.shape[1]}")


# ============================================================================
# 6. HELPER FUNCTIONS (AUC, SIGMOID, FOLDS)
# ============================================================================
def compute_auc(y_true, y_prob) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return 0.5

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def logit(p):
    return np.log((p + 1e-12) / (1 - p + 1e-12))

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_folds(df: pd.DataFrame, n_splits: int, random_state: int = 777) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Topic-aware folds if StratifiedGroupKFold is available."""
    y = df["is_cheating"].values
    if HAS_SGF and "topic" in df.columns:
        groups = df["topic"].astype(str).values
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(splitter.split(df, y, groups))
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(splitter.split(df, y))
    return splits


# ============================================================================
# 7. TRANSFORMERS TRAINING (2×MODELS, BEST-EPOCH, SEED-TRIMMING)
# ============================================================================
TRANSFORMER_MODELS = [
    "microsoft/deberta-v3-small",
    "fakespot-ai/roberta-base-ai-text-detection-v1",
]

N_SPLITS_TRANS = 5
TRANSFORMER_SEEDS = [13, 21, 42, 87, 123]  # 5 seeds / fold
MAX_LEN = 512
LR_TR = 2e-5
EPOCHS_TR = 7
BATCH_TR = 16
OUT_DIR_TR = "./oof_transformers"
os.makedirs(OUT_DIR_TR, exist_ok=True)

if TRANSFORMERS_AVAILABLE and HF_DATASETS_AVAILABLE:
    FP16_TR = torch.cuda.is_available()
else:
    FP16_TR = False

def tokenize_fn(tokenizer, max_len):
    def _fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )
    return _fn

# Hugging Face Trainer-compatible metric
def trainer_compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    return {"roc_auc": roc_auc_score(labels, probs)}

def create_training_args(output_dir: str, seed: int):
    """
    HF-compatible TrainingArguments:
      - Handles both older `eval_strategy` and newer `evaluation_strategy` names.
      - Keeps save/eval strategies in sync so load_best_model_at_end works.
    """
    base_kwargs = dict(
        output_dir=output_dir,
        save_strategy="epoch",          # we'll match eval to this
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        learning_rate=LR_TR,
        per_device_train_batch_size=BATCH_TR,
        per_device_eval_batch_size=BATCH_TR,
        num_train_epochs=EPOCHS_TR,
        fp16=FP16_TR,
        logging_strategy="epoch",
        report_to="none",
        seed=seed,
        save_total_limit=1,
    )

    sig = inspect.signature(TrainingArguments)
    valid_kwargs = {}

    # Simple keys that exist in this HF version
    for k, v in base_kwargs.items():
        if k in sig.parameters:
            valid_kwargs[k] = v

    # Handle the eval/evaluation strategy naming difference
    if "evaluation_strategy" in sig.parameters:
        valid_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in sig.parameters:
        valid_kwargs["eval_strategy"] = "epoch"

    # Ensure save_strategy exists if supported
    if "save_strategy" in sig.parameters:
        valid_kwargs["save_strategy"] = "epoch"

    return TrainingArguments(**valid_kwargs)

def train_transformer_model_get_oof_and_test(model_name: str) -> Dict:
    """
    Multi-seed, multi-fold training with:
      - best-epoch checkpoint (ROC AUC)
      - EarlyStoppingCallback(patience=2)
      - seed trimming: drop worst seed per fold before averaging
    Returns:
      {
        "oof": np.ndarray (n_train,),
        "test": np.ndarray (n_test,),
        "seed_fold_aucs": list of dicts
      }
    """
    if not (TRANSFORMERS_AVAILABLE and HF_DATASETS_AVAILABLE):
        print(f"Skipping {model_name}: transformers or datasets not available")
        return {
            "oof": np.zeros(len(train_df), dtype=np.float32),
            "test": np.zeros(len(test_df), dtype=np.float32),
            "seed_fold_aucs": [],
        }

    print(f"\n=== Transformer model: {model_name} ===")
    folds = make_folds(train_df, N_SPLITS_TRANS, random_state=777)
    print(f"Using {'StratifiedGroupKFold' if HAS_SGF and 'topic' in train_df.columns else 'StratifiedKFold'} with {N_SPLITS_TRANS} folds.")

    oof = np.zeros(len(train_df), dtype=np.float32)
    seed_fold_rows = []

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # test dataset (fixed)
    te_df = test_df[["id", "text"]].copy()
    ds_te = HFDataset.from_pandas(te_df)
    ds_te = ds_te.map(tokenize_fn(tokenizer, MAX_LEN), batched=True, load_from_cache_file=False)
    ds_te.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # global test accumulator (after seed-trimming per fold)
    test_probs_global = np.zeros(len(test_df), dtype=np.float32)

    for f, (tr_idx, va_idx) in enumerate(folds, start=1):
        print(f"[{model_name}] Fold {f}/{N_SPLITS_TRANS} (val size={len(va_idx)})")

        tr_df = train_df.loc[tr_idx, ["text", "is_cheating"]].rename(columns={"is_cheating": "labels"}).reset_index(drop=True)
        va_df = train_df.loc[va_idx, ["text", "is_cheating"]].rename(columns={"is_cheating": "labels"}).reset_index(drop=True)

        ds_tr = HFDataset.from_pandas(tr_df)
        ds_va = HFDataset.from_pandas(va_df)

        tok = tokenize_fn(tokenizer, MAX_LEN)
        ds_tr = ds_tr.map(tok, batched=True, load_from_cache_file=False)
        ds_va = ds_va.map(tok, batched=True, load_from_cache_file=False)

        ds_tr.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        ds_va.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # Store per-seed predictions/metrics so we can trim the worst seed
        fold_seed_info = []

        for seed in TRANSFORMER_SEEDS:
            set_all_seeds(seed)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            try:
                model.config.use_cache = False
                if hasattr(model, "model") and hasattr(model.model, "config"):
                    model.model.config.use_cache = False
            except Exception:
                pass

            out_dir_seed = os.path.join(OUT_DIR_TR, f"tmp_{model_name.split('/')[-1]}_f{f}_s{seed}")
            args = create_training_args(out_dir_seed, seed=seed)

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=ds_tr,
                eval_dataset=ds_va,
                compute_metrics=trainer_compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
                tokenizer=tokenizer,
            )

            trainer.train()

            # VAL
            val_pred = trainer.predict(ds_va)
            logits_val = val_pred.predictions
            val_probs = torch.softmax(torch.tensor(logits_val), dim=1)[:, 1].numpy()
            val_auc = compute_auc(va_df["labels"].values, val_probs)
            print(f"  - seed {seed}: fold {f} AUC={val_auc:.6f}")

            # TEST (with best checkpoint loaded at end)
            test_logits = trainer.predict(ds_te).predictions
            test_probs = torch.softmax(torch.tensor(test_logits), dim=1)[:, 1].numpy()

            fold_seed_info.append({
                "seed": seed,
                "val_auc": val_auc,
                "val_probs": val_probs,
                "test_probs": test_probs,
            })
            seed_fold_rows.append({
                "model": model_name,
                "fold": f,
                "seed": seed,
                "val_auc": val_auc,
            })

            # cleanup per seed
            try:
                shutil.rmtree(out_dir_seed, ignore_errors=True)
            except Exception:
                pass
            del trainer, model
            torch.cuda.empty_cache()
            gc.collect()

        # ---- Seed trimming: drop worst seed on this fold ----
        fold_seed_info_sorted = sorted(fold_seed_info, key=lambda d: d["val_auc"], reverse=True)
        if len(fold_seed_info_sorted) > 1:
            kept = fold_seed_info_sorted[:-1]  # drop worst
            dropped = fold_seed_info_sorted[-1]
            print(f"  => Dropping worst seed on fold {f}: seed {dropped['seed']} (AUC={dropped['val_auc']:.6f})")
        else:
            kept = fold_seed_info_sorted

        # Average kept seeds on VAL + TEST
        val_stack = np.vstack([d["val_probs"] for d in kept])  # (n_seeds_kept, n_val)
        val_mean = val_stack.mean(axis=0)
        oof[va_idx] = val_mean.astype(np.float32)

        test_stack = np.vstack([d["test_probs"] for d in kept])  # (n_seeds_kept, n_test)
        test_mean_fold = test_stack.mean(axis=0)
        test_probs_global += test_mean_fold / N_SPLITS_TRANS

        fold_auc_kept = compute_auc(va_df["labels"].values, val_mean)
        print(f"  => Fold {f} (after seed-trim) AUC = {fold_auc_kept:.6f}")

        # cleanup fold datasets
        del ds_tr, ds_va, tr_df, va_df, val_stack, test_stack, val_mean, test_mean_fold
        torch.cuda.empty_cache()
        gc.collect()

        # Optional: prune HF cache between folds
        try:
            shutil.rmtree("/kaggle/temp/hf", ignore_errors=True)
        except Exception:
            pass

    return {
        "oof": oof,
        "test": test_probs_global.astype(np.float32),
        "seed_fold_aucs": seed_fold_rows,
    }


transformer_results = {}
if TRANSFORMERS_AVAILABLE and HF_DATASETS_AVAILABLE:
    for m in TRANSFORMER_MODELS:
        transformer_results[m] = train_transformer_model_get_oof_and_test(m)

    # Save transformer-only OOF/test artifacts (optional)
    for m in TRANSFORMER_MODELS:
        pd.DataFrame({
            "id": train_df["id"],
            "oof": transformer_results[m]["oof"],
            "y": y_train,
        }).to_csv(os.path.join(OUT_DIR_TR, f"oof_{m.split('/')[-1]}.csv"), index=False)

        pd.DataFrame({
            "id": test_df["id"],
            "is_cheating": transformer_results[m]["test"],
        }).to_csv(os.path.join(OUT_DIR_TR, f"test_{m.split('/')[-1]}.csv"), index=False)

    # Transformer internal blend (logit-weighted)
    if len(TRANSFORMER_MODELS) == 2:
        m1, m2 = TRANSFORMER_MODELS
        oof_t1, oof_t2 = transformer_results[m1]["oof"], transformer_results[m2]["oof"]
        tst_t1, tst_t2 = transformer_results[m1]["test"], transformer_results[m2]["test"]

        grid = np.linspace(0.0, 1.0, 101)
        best_w_tr, best_auc_tr = 0.5, -1
        for w in grid:
            oof_blend_tr = sigmoid(w * logit(oof_t1) + (1 - w) * logit(oof_t2))
            auc_tr = compute_auc(y_train, oof_blend_tr)
            if auc_tr > best_auc_tr:
                best_auc_tr, best_w_tr = auc_tr, float(w)

        oof_transformer_blend = sigmoid(best_w_tr * logit(oof_t1) + (1 - best_w_tr) * logit(oof_t2))
        tst_transformer_blend = sigmoid(best_w_tr * logit(tst_t1) + (1 - best_w_tr) * logit(tst_t2))

        print(f"\nTransformer-only logit blend: best_w={best_w_tr:.2f}, OOF AUC={best_auc_tr:.6f}")
    else:
        oof_transformer_blend = None
        tst_transformer_blend = None
else:
    oof_transformer_blend = None
    tst_transformer_blend = None
    print("\nSkipping transformer training and blending.")


# ============================================================================
# 8. ADVANCED ENSEMBLE WITH OPTIMIZED WEIGHTS (BENFORD SIDE)
# ============================================================================
print("\n" + "="*70)
print("🚀 TRAINING BENFORD-ENHANCED OPTIMIZED ENSEMBLE")
print("="*70)

def improved_ensemble_predictions(predictions_dict, y_train=None, method='optimized_weights'):
    """Optimized-weight ensemble (Benford models)."""
    if method == 'optimized_weights' and y_train is not None:
        from scipy.optimize import minimize

        keys = list(predictions_dict.keys())
        preds = [predictions_dict[k] for k in keys]

        def objective(weights):
            combined = sum(w * p for w, p in zip(weights, preds))
            return -roc_auc_score(y_train, combined)

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(len(preds))]
        x0 = np.ones(len(preds)) / len(preds)

        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            optimized_weights = result.x
            print("Optimized weights:", dict(zip(keys, optimized_weights)))
            combined = sum(w * p for w, p in zip(optimized_weights, preds))
            return combined, optimized_weights
        else:
            print("Weight optimization failed, using equal weights")

    # fallback equal weights
    preds = list(predictions_dict.values())
    combined = np.mean(preds, axis=0)
    return combined, np.ones(len(preds)) / len(preds)


n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

# OOF / test predictions for Benford models
oof_preds = {
    'lgb': np.zeros(len(train_df)),
    'xgb': np.zeros(len(train_df)),
    'cat': np.zeros(len(train_df)),
    'lr':  np.zeros(len(train_df)),
}
test_preds = {
    'lgb': np.zeros(len(test_df)),
    'xgb': np.zeros(len(test_df)),
    'cat': np.zeros(len(test_df)),
    'lr':  np.zeros(len(test_df)),
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train)):
    print(f"\n📊 Fold {fold + 1}/{n_splits}")

    X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.008,
        max_depth=7,
        num_leaves=63,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_split_gain=0.01,
        random_state=SEED + fold,
        verbose=-1,
        n_jobs=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
    )
    oof_preds['lgb'][val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    test_preds['lgb'] += lgb_model.predict_proba(X_test_full)[:, 1] / n_splits

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.008,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        gamma=0,
        random_state=SEED + fold,
        eval_metric='auc',
        tree_method='hist',
        early_stopping_rounds=150,
        n_jobs=-1
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    oof_preds['xgb'][val_idx] = xgb_model.predict_proba(X_val)[:, 1]
    test_preds['xgb'] += xgb_model.predict_proba(X_test_full)[:, 1] / n_splits

    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=3,
        random_seed=SEED + fold,
        verbose=0,
        early_stopping_rounds=150
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    oof_preds['cat'][val_idx] = cat_model.predict_proba(X_val)[:, 1]
    test_preds['cat'] += cat_model.predict_proba(X_test_full)[:, 1] / n_splits

    # Logistic Regression
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    lr_model = LogisticRegression(
        C=0.3,
        max_iter=1000,
        random_state=SEED + fold,
        n_jobs=-1,
        solver='saga'
    )
    lr_model.fit(X_tr_scaled, y_tr)
    oof_preds['lr'][val_idx] = lr_model.predict_proba(X_val_scaled)[:, 1]

    X_test_scaled = scaler.transform(X_test_full)
    test_preds['lr'] += lr_model.predict_proba(X_test_scaled)[:, 1] / n_splits

    lgb_score = roc_auc_score(y_val, oof_preds['lgb'][val_idx])
    xgb_score = roc_auc_score(y_val, oof_preds['xgb'][val_idx])
    cat_score = roc_auc_score(y_val, oof_preds['cat'][val_idx])
    lr_score  = roc_auc_score(y_val, oof_preds['lr'][val_idx])

    print(f"  LGB: {lgb_score:.5f} | XGB: {xgb_score:.5f} | CAT: {cat_score:.5f} | LR: {lr_score:.5f}")

print("\n🔄 Applying optimized weight ensemble (Benford models only)...")
oof_benford_ensemble, benford_weights = improved_ensemble_predictions(oof_preds, y_train, method='optimized_weights')
test_benford_ensemble = np.mean(list(test_preds.values()), axis=0)  # keep original behaviour


# ============================================================================
# 9. META-ENSEMBLE: BENFORD ENSEMBLE + 2×TRANSFORMER BLEND
# ============================================================================
print("\n" + "="*70)
print("🧠 META-ENSEMBLING BENFORD + TRANSFORMERS")
print("="*70)

if oof_transformer_blend is not None:
    print(f"Benford-only OOF AUC:     {compute_auc(y_train, oof_benford_ensemble):.6f}")
    print(f"Transformer-only OOF AUC: {compute_auc(y_train, oof_transformer_blend):.6f}")

    grid = np.linspace(0.0, 1.0, 101)
    best_w_meta, best_auc_meta = 0.5, -1
    for w in grid:
        oof_meta = sigmoid(w * logit(oof_benford_ensemble) + (1 - w) * logit(oof_transformer_blend))
        auc_meta = compute_auc(y_train, oof_meta)
        if auc_meta > best_auc_meta:
            best_auc_meta, best_w_meta = auc_meta, float(w)

    oof_blend_lvl2 = sigmoid(best_w_meta * logit(oof_benford_ensemble) +
                             (1 - best_w_meta) * logit(oof_transformer_blend))
    test_blend_lvl2 = sigmoid(best_w_meta * logit(test_benford_ensemble) +
                              (1 - best_w_meta) * logit(tst_transformer_blend))
    print(f"Meta logit-blend w={best_w_meta:.2f}, OOF AUC={best_auc_meta:.6f}")
else:
    print("Transformers unavailable, using Benford ensemble only.")
    oof_blend_lvl2 = oof_benford_ensemble
    test_blend_lvl2 = test_benford_ensemble

print(f"Final pre-PL OOF AUC (level-2 blend): {compute_auc(y_train, oof_blend_lvl2):.6f}")


# ============================================================================
# 10. SMART PSEUDO-LABELING (USING META-ENSEMBLE PREDICTIONS)
# ============================================================================
print("\n" + "="*70)
print("🎯 APPLYING SMART PSEUDO-LABELING (LEVEL-2 BLEND)")
print("="*70)

def smart_pseudo_labeling(train_features, test_features, y_train, 
                          initial_predictions, 
                          confidence_threshold_high=0.98, 
                          confidence_threshold_low=0.02):
    """Smart pseudo-labeling strategy"""
    high_conf_mask = (initial_predictions >= confidence_threshold_high) | (initial_predictions <= confidence_threshold_low)

    if high_conf_mask.sum() < len(test_features) * 0.3:
        confidence_threshold_high = 0.95
        confidence_threshold_low = 0.05
        high_conf_mask = (initial_predictions >= confidence_threshold_high) | (initial_predictions <= confidence_threshold_low)
        print(f"  Adjusted thresholds to: high={confidence_threshold_high}, low={confidence_threshold_low}")

    high_conf_indices = np.where(high_conf_mask)[0]
    print(f"  High confidence samples: {len(high_conf_indices)} / {len(test_features)} "
          f"({100*len(high_conf_indices)/len(test_features):.1f}%)")

    if len(high_conf_indices) > 0:
        pseudo_X = test_features[high_conf_indices]
        pseudo_y = (initial_predictions[high_conf_indices] > 0.5).astype(int)

        confidence_weights = np.where(
            initial_predictions[high_conf_indices] > 0.5,
            initial_predictions[high_conf_indices],
            1 - initial_predictions[high_conf_indices]
        )

        X_combined = np.vstack([train_features, pseudo_X])
        y_combined = np.concatenate([y_train, pseudo_y])
        sample_weights = np.concatenate([
            np.ones(len(y_train)),
            confidence_weights * 0.5
        ])

        return X_combined, y_combined, sample_weights, len(high_conf_indices)

    return train_features, y_train, np.ones(len(y_train)), 0

test_ensemble = test_blend_lvl2  # seed for PL
X_combined, y_combined, sample_weights, n_pseudo = smart_pseudo_labeling(
    X_train_full, X_test_full, y_train, test_ensemble
)

if n_pseudo > 0:
    print(f"✓ Using {n_pseudo} pseudo-labeled samples for enhanced training")

    lgb_pseudo = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.01,
        max_depth=7,
        num_leaves=63,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=SEED,
        verbose=-1,
        n_jobs=-1
    )
    lgb_pseudo.fit(X_combined, y_combined, sample_weight=sample_weights)
    test_pseudo = lgb_pseudo.predict_proba(X_test_full)[:, 1]

    blend_ratio = min(0.3, n_pseudo / len(test_df))
    test_ensemble_enhanced = test_ensemble * (1 - blend_ratio) + test_pseudo * blend_ratio

    print(f"✓ Pseudo-label blend ratio: {blend_ratio:.3f}")
    test_ensemble = test_ensemble_enhanced


# ============================================================================
# 11. TOPIC-SPECIFIC CALIBRATION + ISOTONIC CALIBRATION
# ============================================================================
print("\n" + "="*70)
print("🎨 APPLYING TOPIC-SPECIFIC CALIBRATION")
print("="*70)

def topic_specific_calibration(predictions, test_df, train_df):
    """Topic-specific calibration"""
    topic_stats = train_df.groupby('topic')['is_cheating'].agg(['mean', 'count']).reset_index()
    topic_stats = topic_stats[topic_stats['count'] > 5]

    calibrated_predictions = predictions.copy()
    adjustment_count = 0

    for _, row in topic_stats.iterrows():
        topic = row['topic']
        topic_mean = row['mean']
        topic_mask = test_df['topic'] == topic

        if topic_mask.sum() > 0:
            adjustment_strength = 0.1
            topic_predictions = predictions[topic_mask]

            if topic_mean > 0.7:
                calibrated_predictions[topic_mask] = (
                    topic_predictions * (1 - adjustment_strength) +
                    np.clip(topic_predictions + 0.1, 0, 1) * adjustment_strength
                )
                adjustment_count += topic_mask.sum()
            elif topic_mean < 0.3:
                calibrated_predictions[topic_mask] = (
                    topic_predictions * (1 - adjustment_strength) +
                    np.clip(topic_predictions - 0.1, 0, 1) * adjustment_strength
                )
                adjustment_count += topic_mask.sum()

    print(f"✓ Applied topic-specific calibration to {adjustment_count} samples")
    return calibrated_predictions

test_calibrated = topic_specific_calibration(test_ensemble, test_df, train_df)

print("\n" + "="*70)
print("⚙️ APPLYING ADVANCED CALIBRATION (ISOTONIC)")
print("="*70)

from sklearn.isotonic import IsotonicRegression

iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(oof_blend_lvl2, y_train)  # calibrate on meta-ensemble OOF
test_final_calibrated = iso_reg.transform(test_calibrated)

print("✓ Isotonic calibration applied")

test_final_calibrated = np.clip(test_final_calibrated, 0.001, 0.999)
test_final = 0.97 * test_final_calibrated + 0.03 * test_ensemble

print("✓ Final smoothing applied")


# ============================================================================
# 12. CREATE SUBMISSIONS
# ============================================================================
print("\n" + "="*70)
print("💾 CREATING BENFORD + TRANSFORMER META-ENSEMBLE SUBMISSIONS")
print("="*70)

submission_main = pd.DataFrame({
    'id': test_df['id'],
    'is_cheating': test_final
})
submission_main.to_csv('submission.csv', index=False)

submission_conservative = pd.DataFrame({
    'id': test_df['id'],
    'is_cheating': test_final_calibrated
})
submission_conservative.to_csv('benford_transformer_conservative.csv', index=False)

submission_aggressive = pd.DataFrame({
    'id': test_df['id'],
    'is_cheating': np.power(test_final, 0.9)
})
submission_aggressive.to_csv('benford_transformer_aggressive.csv', index=False)

print("✅ Created 3 submission versions:")
print("  - submission.csv                  (recommended)")
print("  - benford_transformer_conservative.csv")
print("  - benford_transformer_aggressive.csv")

print(f"\n📊 Submission Statistics (submission.csv):")
print(submission_main['is_cheating'].describe())

predicted_cheating = (submission_main['is_cheating'] > 0.5).sum()
print(f"\nPredicted cheating: {predicted_cheating} / {len(test_df)} "
      f"({100*predicted_cheating/len(test_df):.1f}%)")


# ============================================================================
# 13. FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("📋 BENFORD + 2×TRANSFORMER META-ENSEMBLE SUMMARY")
print("="*70)
print(f"Benford-only OOF AUC:      {compute_auc(y_train, oof_benford_ensemble):.6f}")
if oof_transformer_blend is not None:
    print(f"Transformer-only OOF AUC:  {compute_auc(y_train, oof_transformer_blend):.6f}")
print(f"Level-2 blend OOF AUC:     {compute_auc(y_train, oof_blend_lvl2):.6f}")
print("="*70)
print("\n🎉 META-ENSEMBLED SOLUTION COMPLETE!")
print("🏆 Submit 'submission.csv' for best results.")
