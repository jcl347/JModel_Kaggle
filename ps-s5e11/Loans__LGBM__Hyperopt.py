# ============================================================
# 1. Imports & basic setup
# ============================================================
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd

from itertools import combinations

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin

import lightgbm as lgb
!pip install -q hyperopt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Toggle GPU for LightGBM (set to False if GPU complains)
USE_GPU = True

# Separate folds for tuning vs final training
N_SPLITS_TUNE = 3
N_SPLITS_FINAL = 5

# LGBM boosting budget
NUM_BOOST_ROUND = 3000
ES_ROUNDS = 200

# ============================================================
# 2. Load data
# ============================================================
train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
orig  = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')

print('Train Shape:', train.shape)
print('Test Shape: ', test.shape)
print('Orig Shape: ', orig.shape)

# ============================================================
# 3. Basic definitions
# ============================================================
TARGET = 'loan_paid_back'
CATS = [
    'gender',
    'marital_status',
    'education_level',
    'employment_status',
    'loan_purpose',
    'grade_subgrade'
]

BASE = [col for col in train.columns if col not in ['id', TARGET]]
print("BASE features:", BASE)

# ============================================================
# 4. Digit features
# ============================================================
DIGIT = []

cols_to_digitize = {
    'debt_to_income_ratio': 1000,   # -> up to 3 digits
    'credit_score': 'direct',       # 3 digits
    'interest_rate': 100,           # up to 4 digits
}

for col, multiplier in cols_to_digitize.items():
    temp_col_name = f'{col}_TEMP_INT'
    
    for df in [train, test, orig]:
        if multiplier == 'direct':
            df[temp_col_name] = df[col]
        else:
            df[temp_col_name] = (df[col] * multiplier).round(0).astype(int)

        temp_str = df[temp_col_name].astype(str)
        
        if col == 'credit_score':
            max_len = 3
        elif col == 'debt_to_income_ratio':
            max_len = 3
        elif col == 'interest_rate':
            max_len = 4
        else:
            max_len = temp_str.str.len().max()
        
        temp_str_padded = temp_str.str.zfill(max_len)
        for i in range(max_len):
            new_col_name = f'{col}_DIGIT_{i+1}'
            if df is train:
                if new_col_name not in DIGIT:
                    DIGIT.append(new_col_name)
            df[new_col_name] = temp_str_padded.str[i].astype(int)
            
    for df in [train, test, orig]:
        df.drop(columns=[temp_col_name], inplace=True)

print(f'{len(DIGIT)} DIGIT features created:', DIGIT)

# ============================================================
# 5. Rounding features
# ============================================================
ROUND = []

rounding_levels = {
    '1s': 0,
    '10s': -1,
    '100s': -2,
    '1000s': -3,
}

for col in ['annual_income', 'loan_amount']:
    for suffix, level in rounding_levels.items():
        new_col_name = f'{col}_ROUND_{suffix}'
        ROUND.append(new_col_name)
        
        for df in [train, test, orig]:
            df[new_col_name] = df[col].round(level).astype(int)

print(f'{len(ROUND)} ROUND features created:', ROUND)

# ============================================================
# 6. Interaction features (string combos of BASE)
# ============================================================
INTER = []

for col1, col2 in combinations(BASE, 2):
    new_col_name = f'{col1}_{col2}'
    INTER.append(new_col_name)
    for df in [train, test, orig]:
        df[new_col_name] = df[col1].astype(str) + '_' + df[col2].astype(str)

print(f'{len(INTER)} INTER features.')

# ============================================================
# 7. ORIG dataset encodings (mean + count per BASE column)
# ============================================================
ORIG_FEATS = []

for col in BASE:
    # mean of TARGET in orig
    mean_map = orig.groupby(col)[TARGET].mean()
    new_mean_col_name = f"orig_mean_{col}"
    mean_map.name = new_mean_col_name
    
    train = train.merge(mean_map, on=col, how='left')
    test  = test.merge(mean_map,  on=col, how='left')
    ORIG_FEATS.append(new_mean_col_name)

    # count in orig
    new_count_col_name = f"orig_count_{col}"
    count_map = orig.groupby(col).size().reset_index(name=new_count_col_name)
    
    train = train.merge(count_map, on=col, how='left')
    test  = test.merge(count_map,  on=col, how='left')
    ORIG_FEATS.append(new_count_col_name)

print(len(ORIG_FEATS), 'ORIG features created.')

# fill NaNs with global orig target mean
global_orig_mean = orig[TARGET].mean()
train[ORIG_FEATS] = train[ORIG_FEATS].fillna(global_orig_mean)
test[ORIG_FEATS]  = test[ORIG_FEATS].fillna(global_orig_mean)

# ============================================================
# 8. Final feature list
# ============================================================
FEATURES = BASE + ORIG_FEATS + INTER + ROUND + DIGIT
print(len(FEATURES), 'features in total.')

X = train[FEATURES]
y = train[TARGET]

print("X shape:", X.shape)
print("y mean:", y.mean())

# ============================================================
# 9. TargetEncoder
# ============================================================
class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target Encoder with multiple aggs, CV leakage control, and smoothing.
    """
    def __init__(self, cols_to_encode, aggs=['mean'], cv=5, smooth='auto', drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_stats_ = {}

    def fit(self, X, y):
        temp_df = X.copy()
        temp_df['target'] = y

        # global stats
        for agg_func in self.aggs:
            self.global_stats_[agg_func] = y.agg(agg_func)

        # per-column mappings
        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg_func in self.aggs:
                mapping = temp_df.groupby(col)['target'].agg(agg_func)
                self.mappings_[col][agg_func] = mapping
        
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.cols_to_encode:
            for agg_func in self.aggs:
                new_col_name = f'TE_{col}_{agg_func}'
                map_series = self.mappings_[col][agg_func]
                X_transformed[new_col_name] = X[col].map(map_series)
                X_transformed[new_col_name].fillna(self.global_stats_[agg_func], inplace=True)
        
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
            
        return X_transformed

    def fit_transform(self, X, y):
        # fit global mappings
        self.fit(X, y)

        encoded_features = pd.DataFrame(index=X.index)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=RANDOM_STATE)

        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            
            temp_df_train = X_train.copy()
            temp_df_train['target'] = y_train

            for col in self.cols_to_encode:
                for agg_func in self.aggs:
                    new_col_name = f'TE_{col}_{agg_func}'
                    
                    fold_global_stat = y_train.agg(agg_func)
                    mapping = temp_df_train.groupby(col)['target'].agg(agg_func)

                    if agg_func == 'mean':
                        counts = temp_df_train.groupby(col)['target'].count()
                        
                        m = self.smooth
                        if self.smooth == 'auto':
                            variance_between = mapping.var()
                            avg_variance_within = temp_df_train.groupby(col)['target'].var().mean()
                            if (variance_between is not None) and (variance_between > 0):
                                m = avg_variance_within / max(variance_between, 1e-9)
                            else:
                                m = 0
                        
                        smoothed_mapping = (counts * mapping + m * fold_global_stat) / (counts + m)
                        encoded_values = X_val[col].map(smoothed_mapping)
                    else:
                        encoded_values = X_val[col].map(mapping)
                    
                    encoded_features.loc[X_val.index, new_col_name] = encoded_values.fillna(fold_global_stat)

        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]
            
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
            
        return X_transformed

# ============================================================
# 10. CV setup (tuning vs final)
# ============================================================
skf_tune  = StratifiedKFold(n_splits=N_SPLITS_TUNE,  shuffle=True, random_state=RANDOM_STATE)
skf_final = StratifiedKFold(n_splits=N_SPLITS_FINAL, shuffle=True, random_state=RANDOM_STATE)

# Base columns for second-stage TE
BASE_TE_COL = ['debt_to_income_ratio', 'credit_score'] + ROUND + DIGIT
print("BASE_TE_COL:", BASE_TE_COL)

# ============================================================
# 11. Hyperopt search space for LGBM (GPU-safe max_bin)
# ============================================================
lgb_space = {
    'num_leaves':        hp.quniform('num_leaves', 31, 255, 1),
    'max_depth':         hp.quniform('max_depth', 4, 12, 1),
    'learning_rate':     hp.loguniform('learning_rate', np.log(0.005), np.log(0.05)),
    'min_child_samples': hp.quniform('min_child_samples', 20, 200, 5),
    'subsample':         hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree':  hp.uniform('colsample_bytree', 0.6, 1.0),
    'reg_lambda':        hp.loguniform('reg_lambda', np.log(1e-3), np.log(10)),
    'reg_alpha':         hp.loguniform('reg_alpha', np.log(1e-3), np.log(10)),
    'max_bin':           hp.quniform('max_bin', 64, 255, 8),  # <= 255 for GPU
}

def build_lgb_params(params):
    p = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': float(params['learning_rate']),
        'num_leaves': int(params['num_leaves']),
        'max_depth': int(params['max_depth']),
        'min_child_samples': int(params['min_child_samples']),
        'subsample': float(params['subsample']),
        'colsample_bytree': float(params['colsample_bytree']),
        'reg_lambda': float(params['reg_lambda']),
        'reg_alpha': float(params['reg_alpha']),
        'max_bin': int(params['max_bin']),
        'n_jobs': -1,
        'verbose': -1,
    }
    if USE_GPU:
        p['device'] = 'gpu'
        p['gpu_platform_id'] = 0
        p['gpu_device_id'] = 0
    return p

# ============================================================
# 12. Hyperopt objective with TUNING folds (3-fold) + TDE
# ============================================================
def lgb_objective(params):
    params_lgb = build_lgb_params(params)

    oof = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(skf_tune.split(X, y), 1):
        X_tr = X.iloc[train_idx].copy()
        X_va = X.iloc[val_idx].copy()
        y_tr = y.iloc[train_idx]
        y_va = y.iloc[val_idx]

        # 1) TDE on INTER (drop originals)
        TE_inter = TargetEncoder(
            cols_to_encode=INTER,
            cv=5,
            smooth='auto',
            aggs=['mean'],
            drop_original=True
        )
        X_tr_enc = TE_inter.fit_transform(X_tr, y_tr)
        X_va_enc = TE_inter.transform(X_va)

        # 2) TDE on BASE_TE_COL (keep originals)
        TE_base = TargetEncoder(
            cols_to_encode=BASE_TE_COL,
            cv=5,
            smooth='auto',
            aggs=['mean'],
            drop_original=False
        )
        X_tr_enc = TE_base.fit_transform(X_tr_enc, y_tr)
        X_va_enc = TE_base.transform(X_va_enc)

        # 3) Factorize categoricals within fold
        for c in CATS:
            combined = pd.concat([X_tr_enc[c], X_va_enc[c]], axis=0)
            combined, _ = combined.factorize()
            X_tr_enc[c] = combined[:len(X_tr_enc)]
            X_va_enc[c] = combined[len(X_tr_enc):]

        lgb_train = lgb.Dataset(X_tr_enc, label=y_tr)
        lgb_valid = lgb.Dataset(X_va_enc, label=y_va, reference=lgb_train)

        model = lgb.train(
            params_lgb,
            lgb_train,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(ES_ROUNDS)]
        )

        oof[val_idx] = model.predict(
            X_va_enc,
            num_iteration=model.best_iteration
        )

    score = roc_auc_score(y, oof)
    print(f"Hyperopt trial AUC: {score:.6f}")
    return {'loss': -score, 'status': STATUS_OK}

# ============================================================
# 13. Run Hyperopt TPE (reduced evals)
# ============================================================
MAX_EVALS = 5  # was 30; big runtime saver

trials = Trials()
best = fmin(
    fn=lgb_objective,
    space=lgb_space,
    algo=tpe.suggest,
    max_evals=MAX_EVALS,
    trials=trials,
    rstate=np.random.default_rng(RANDOM_STATE)
)

print("Best raw Hyperopt params:", best)
best_params_lgb = build_lgb_params(best)
print("Best LGBM params:", best_params_lgb)

# ============================================================
# 14. Final CV training with tuned params + 5-FOLD CV
# ============================================================
oof_lgb = np.zeros(len(X))
test_lgb = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf_final.split(X, y), 1):
    print(f"=== Final LGBM Fold {fold}/{N_SPLITS_FINAL} ===")

    X_tr = X.iloc[train_idx].copy()
    X_va = X.iloc[val_idx].copy()
    y_tr = y.iloc[train_idx]
    y_va = y.iloc[val_idx]
    X_te = test[FEATURES].copy()

    # 1) TDE on INTER (drop originals)
    TE_inter = TargetEncoder(
        cols_to_encode=INTER,
        cv=5,
        smooth='auto',
        aggs=['mean'],
        drop_original=True
    )
    X_tr_enc = TE_inter.fit_transform(X_tr, y_tr)
    X_va_enc = TE_inter.transform(X_va)
    X_te_enc = TE_inter.transform(X_te)

    # 2) TDE on BASE_TE_COL (keep originals)
    TE_base = TargetEncoder(
        cols_to_encode=BASE_TE_COL,
        cv=5,
        smooth='auto',
        aggs=['mean'],
        drop_original=False
    )
    X_tr_enc = TE_base.fit_transform(X_tr_enc, y_tr)
    X_va_enc = TE_base.transform(X_va_enc)
    X_te_enc = TE_base.transform(X_te_enc)

    # 3) Factorize CATS across train/val/test for this fold
    for c in CATS:
        combined = pd.concat([X_tr_enc[c], X_va_enc[c], X_te_enc[c]], axis=0)
        combined, _ = combined.factorize()
        X_tr_enc[c] = combined[:len(X_tr_enc)]
        X_va_enc[c] = combined[len(X_tr_enc):len(X_tr_enc) + len(X_va_enc)]
        X_te_enc[c] = combined[len(X_tr_enc) + len(X_va_enc):]

    lgb_train = lgb.Dataset(X_tr_enc, label=y_tr)
    lgb_valid = lgb.Dataset(X_va_enc, label=y_va, reference=lgb_train)

    model = lgb.train(
        best_params_lgb,
        lgb_train,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(ES_ROUNDS)]
    )

    val_pred = model.predict(
        X_va_enc,
        num_iteration=model.best_iteration
    )
    oof_lgb[val_idx] = val_pred

    test_lgb += model.predict(
        X_te_enc,
        num_iteration=model.best_iteration
    ) / N_SPLITS_FINAL

oof_auc = roc_auc_score(y, oof_lgb)
print("=" * 40)
print(f"Final LGBM OOF AUC: {oof_auc:.6f}")
print("=" * 40)

# ============================================================
# 15. Save OOF and submission files
# ============================================================
oof_df = pd.DataFrame({
    'id': train.id,
    TARGET: oof_lgb
})
oof_path = f"oof_lgbm_cv_{oof_auc:.6f}.csv"
oof_df.to_csv(oof_path, index=False)
print("Saved OOF to:", oof_path)

sub_df = pd.DataFrame({
    'id': test.id,
    TARGET: test_lgb
})
sub_path = f"sub_lgbm_cv_{oof_auc:.6f}.csv"
sub_df.to_csv(sub_path, index=False)
print("Saved submission to:", sub_path)