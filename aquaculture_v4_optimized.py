#!/usr/bin/env python3
"""
GeoAI Aquaculture Pond Identification Challenge
Target: 0.926894879 F1 score on test set

Strategy & Rationale:
======================
1. RICHER FEATURE ENGINEERING
   - All previous spectral indices (NDVI, NDWI, MNDWI, SAR)
   - Added: EVI, SAVI, RECI (Red-Edge Chlorophyll), AWEIsh/AWEInsh (Automated Water Extraction)
   - Added: NDPI (Normalized Difference Pond Index), BSI (Bare Soil Index)
   - Cross-band ratios across all months
   - Temporal trend features: first-half vs second-half year means (seasonality)
   - Temporal autocorrelation (consecutive month change variance)
   - Valid observation count per band (missing data as signal)

2. THREE-MODEL STACKING ENSEMBLE
   - LightGBM (handles NaN natively, excellent for tabular)
   - XGBoost (complementary learner)
   - CatBoost (strong on small datasets with ordered boosting)
   - Level-1: All three trained with 10-fold StratifiedKFold OOF
   - Level-2: Logistic Regression meta-learner on OOF probabilities

3. THRESHOLD OPTIMIZATION
   - Sweep thresholds 0.01-0.99 on full OOF blend
   - Optimize for F1 (the actual competition metric)

4. ON-DEVICE RATIONALE
   - Dataset: 1821 rows × ~500 engineered features (fits in RAM)
   - 8 CPUs available for parallel tree training
   - No data transfer overhead vs Colab
   - Deterministic outputs for reproducibility
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = '/home/mwars/Projects/geoai_aquaculture/'

train = pd.read_csv(DATA_DIR + 'Train.csv')
test  = pd.read_csv(DATA_DIR + 'Test.csv')
sub   = pd.read_csv(DATA_DIR + 'SampleSubmission.csv')

print(f'Train: {train.shape}  |  Test: {test.shape}')
print(f'Label distribution:\n{train["label"].value_counts()}')

# -9999 = missing (cloud / no observation). Replace with NaN.
train.replace(-9999, np.nan, inplace=True)
train.replace(-9999.0, np.nan, inplace=True)
test.replace(-9999, np.nan, inplace=True)
test.replace(-9999.0, np.nan, inplace=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. EXTENDED FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
BANDS  = ['VH', 'VV', 'blue', 'green', 'nir', 'nira', 're1', 're2', 're3', 'red', 'swir1', 'swir2']
MONTHS = [f'{i:02d}' for i in range(1, 13)]
EPS = 1e-6


def compute_indices(df, suffix):
    """
    Compute per-month spectral & SAR indices.
    Each index targets a specific physical property of aquaculture ponds.
    """
    nir   = df[f'nir_{suffix}']
    red   = df[f'red_{suffix}']
    green = df[f'green_{suffix}']
    blue  = df[f'blue_{suffix}']
    swir1 = df[f'swir1_{suffix}']
    swir2 = df[f'swir2_{suffix}']
    nira  = df[f'nira_{suffix}']   # Narrow NIR
    re1   = df[f're1_{suffix}']    # Red-Edge band 1 (705nm)
    re2   = df[f're2_{suffix}']    # Red-Edge band 2 (740nm)
    re3   = df[f're3_{suffix}']    # Red-Edge band 3 (783nm)
    vh    = df[f'VH_{suffix}']
    vv    = df[f'VV_{suffix}']

    feats = {}

    # Water indices – key discriminators for ponds vs land
    feats[f'NDVI_{suffix}']    = (nir - red)    / (nir + red + EPS)
    feats[f'NDWI_{suffix}']    = (green - nir)   / (green + nir + EPS)
    feats[f'MNDWI_{suffix}']   = (green - swir1) / (green + swir1 + EPS)
    feats[f'NDWI2_{suffix}']   = (nir - swir1)   / (nir + swir1 + EPS)

    # AWEI: Automated Water Extraction Index (shadow-insensitive)
    # AWEInsh = 4*(green - swir1) - (0.25*nir + 2.75*swir2)
    # AWEIsh  = blue + 2.5*green - 1.5*(nir + swir1) - 0.25*swir2
    feats[f'AWEInsh_{suffix}'] = 4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)
    feats[f'AWEIsh_{suffix}']  = blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2

    # EVI: Enhanced Vegetation Index (reduces soil/atmosphere effects)
    feats[f'EVI_{suffix}']     = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + EPS)

    # SAVI: Soil-Adjusted Vegetation Index (L=0.5 correction)
    feats[f'SAVI_{suffix}']    = 1.5 * (nir - red) / (nir + red + 0.5 + EPS)

    # Red-Edge Chlorophyll Index: sensitive to plant health (low for water)
    feats[f'RECI_{suffix}']    = (nir / (re1 + EPS)) - 1.0

    # BSI: Bare Soil Index (high = exposed soil, low = water)
    feats[f'BSI_{suffix}']     = ((swir1 + red) - (nir + blue)) / \
                                  ((swir1 + red) + (nir + blue) + EPS)

    # NDPI: Normalized Difference Pond Index (NIR/re1 – targets pond reflectance)
    feats[f'NDPI_{suffix}']    = (nir - re1) / (nir + re1 + EPS)

    # SAR features – calm water = specular reflection away from radar → very low VH
    feats[f'SAR_ratio_{suffix}'] = vh / (vv + EPS)
    feats[f'SAR_diff_{suffix}']  = vh - vv
    feats[f'SAR_sum_{suffix}']   = vh + vv
    feats[f'VH_{suffix}']        = vh
    feats[f'VV_{suffix}']        = vv

    # Blue/Green ratio (turbid vs clear water)
    feats[f'BG_ratio_{suffix}']  = blue / (green + EPS)

    # SWIR ratio (higher for vegetation than open water)
    feats[f'SWIR_ratio_{suffix}'] = swir1 / (swir2 + EPS)

    return feats


def build_features(df):
    """
    Build full feature matrix combining:
    - Per-month spectral indices
    - Temporal statistics (mean/std/min/max/range/median/p25/p75)
    - Seasonality features (first-half vs second-half year contrast)
    - Temporal change variance (how stable the pond is over time)
    - Raw band temporal stats
    - Missing data counts as features
    """
    feature_dict = {}

    # Collect all index time-series
    index_names = [
        'NDVI', 'NDWI', 'MNDWI', 'NDWI2', 'AWEInsh', 'AWEIsh',
        'EVI', 'SAVI', 'RECI', 'BSI', 'NDPI',
        'SAR_ratio', 'SAR_diff', 'SAR_sum', 'BG_ratio', 'SWIR_ratio',
        'VH', 'VV'
    ]
    # Store monthly values for each index
    monthly = {n: [] for n in index_names}

    for m in MONTHS:
        idx = compute_indices(df, m)
        feature_dict.update(idx)
        # Route each named index to its list
        for n in index_names:
            key = f'{n}_{m}'
            if key in idx:
                monthly[n].append(idx[key])

    # Temporal statistics for each index
    for name, series_list in monthly.items():
        ts = pd.concat(series_list, axis=1)
        feature_dict[f'{name}_mean']   = ts.mean(axis=1)
        feature_dict[f'{name}_std']    = ts.std(axis=1)
        feature_dict[f'{name}_min']    = ts.min(axis=1)
        feature_dict[f'{name}_max']    = ts.max(axis=1)
        feature_dict[f'{name}_range']  = ts.max(axis=1) - ts.min(axis=1)
        feature_dict[f'{name}_median'] = ts.median(axis=1)
        feature_dict[f'{name}_p25']    = ts.quantile(0.25, axis=1)
        feature_dict[f'{name}_p75']    = ts.quantile(0.75, axis=1)
        feature_dict[f'{name}_n_obs']  = ts.notna().sum(axis=1)
        # IQR: interquartile range
        feature_dict[f'{name}_iqr']    = (ts.quantile(0.75, axis=1) -
                                           ts.quantile(0.25, axis=1))

        # Seasonality: compare wet season (May-Oct) to dry season (Nov-Apr)
        # Months 05-10 = indices 4-9 in the 12-month list
        wet_months  = ts.iloc[:, 4:10]   # May – October
        dry_months  = ts.iloc[:, [0,1,2,3,10,11]]  # Nov-Apr (wrap)
        feature_dict[f'{name}_wet_mean'] = wet_months.mean(axis=1)
        feature_dict[f'{name}_dry_mean'] = dry_months.mean(axis=1)
        feature_dict[f'{name}_season_diff'] = (
            feature_dict[f'{name}_wet_mean'] - feature_dict[f'{name}_dry_mean']
        )

        # Temporal autocorrelation: mean absolute difference between consecutive months
        # Low = stable pond | High = fluctuating vegetation or cloud noise
        consecutive_diffs = []
        for i in range(len(MONTHS) - 1):
            diff = (series_list[i] - series_list[i + 1]).abs()
            consecutive_diffs.append(diff)
        if consecutive_diffs:
            diff_ts = pd.concat(consecutive_diffs, axis=1)
            feature_dict[f'{name}_consec_diff_mean'] = diff_ts.mean(axis=1)
            feature_dict[f'{name}_consec_diff_std']  = diff_ts.std(axis=1)

    # Raw band temporal statistics
    for band in BANDS:
        band_cols = [f'{band}_{m}' for m in MONTHS]
        ts = df[band_cols]
        feature_dict[f'{band}_raw_mean']   = ts.mean(axis=1)
        feature_dict[f'{band}_raw_std']    = ts.std(axis=1)
        feature_dict[f'{band}_raw_min']    = ts.min(axis=1)
        feature_dict[f'{band}_raw_max']    = ts.max(axis=1)
        feature_dict[f'{band}_raw_range']  = ts.max(axis=1) - ts.min(axis=1)
        feature_dict[f'{band}_raw_median'] = ts.median(axis=1)

    # Missing data counts as a feature: more clouds = less reliable observation
    all_cols = [f'{b}_{m}' for b in BANDS for m in MONTHS]
    feature_dict['n_valid_obs']     = df[all_cols].notna().sum(axis=1)
    feature_dict['n_missing_obs']   = df[all_cols].isna().sum(axis=1)
    feature_dict['pct_valid_obs']   = feature_dict['n_valid_obs'] / len(all_cols)

    sar_cols = [f'VH_{m}' for m in MONTHS]
    feature_dict['n_valid_sar']     = df[sar_cols].notna().sum(axis=1)

    opt_cols = [f'blue_{m}' for m in MONTHS]
    feature_dict['n_valid_optical'] = df[opt_cols].notna().sum(axis=1)

    return pd.DataFrame(feature_dict, index=df.index)


print('Building train features...')
X_train_feats = build_features(train)
print('Building test features...')
X_test_feats  = build_features(test)

# Stack engineered features alongside raw band values
raw_cols = [c for c in train.columns if c not in ['ID', 'label']]
X_train = pd.concat([train[raw_cols].reset_index(drop=True),
                     X_train_feats.reset_index(drop=True)], axis=1)
X_test  = pd.concat([test[raw_cols].reset_index(drop=True),
                     X_test_feats.reset_index(drop=True)], axis=1)
y_train = train['label'].values

print(f'Feature matrix: {X_train.shape}')
print(f'Class balance: {y_train.mean():.2%} positive (aquaculture ponds)')

# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
N_FOLDS = 10   # More folds → more stable OOF estimates on small data
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Imputed arrays needed for XGBoost and CatBoost
imputer = SimpleImputer(strategy='median')
X_train_arr = X_train.values
X_test_arr  = X_test.values

X_train_imp = imputer.fit_transform(X_train_arr)
X_test_imp  = imputer.transform(X_test_arr)

# --- LightGBM ---
lgb_params = {
    'objective':         'binary',
    'metric':            ['binary_logloss', 'auc'],
    'n_estimators':      3000,
    'learning_rate':     0.01,
    'num_leaves':        127,
    'max_depth':         -1,
    'min_child_samples': 15,
    'feature_fraction':  0.6,
    'bagging_fraction':  0.8,
    'bagging_freq':      5,
    'reg_alpha':         0.05,
    'reg_lambda':        0.05,
    'random_state':      42,
    'n_jobs':            -1,
    'verbose':           -1,
    'is_unbalance':      True,
}

# --- XGBoost ---
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_params = {
    'n_estimators':      2000,
    'learning_rate':     0.01,
    'max_depth':         7,
    'subsample':         0.8,
    'colsample_bytree':  0.6,
    'gamma':             0.1,
    'min_child_weight':  3,
    'scale_pos_weight':  pos_weight,
    'eval_metric':       'auc',
    'random_state':      42,
    'n_jobs':            -1,
    'early_stopping_rounds': 150,
    'verbosity':         0,
}

# --- CatBoost ---
cat_params = {
    'iterations':          2000,
    'learning_rate':       0.01,
    'depth':               7,
    'l2_leaf_reg':         3.0,
    'random_strength':     1.0,
    'bagging_temperature': 1.0,
    'auto_class_weights':  'Balanced',
    'eval_metric':         'AUC',
    'early_stopping_rounds': 150,
    'random_seed':         42,
    'verbose':             0,
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. CROSS-VALIDATED OOF PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
lgb_oof  = np.zeros(len(X_train))
xgb_oof  = np.zeros(len(X_train))
cat_oof  = np.zeros(len(X_train))

lgb_test  = np.zeros(len(X_test))
xgb_test  = np.zeros(len(X_test))
cat_test  = np.zeros(len(X_test))

print('\n' + '='*60)
print('Training 3-model ensemble (10-fold CV)')
print('='*60)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_imp, y_train)):
    X_tr_r, X_val_r = X_train_arr[tr_idx], X_train_arr[val_idx]
    X_tr_i, X_val_i = X_train_imp[tr_idx], X_train_imp[val_idx]
    y_tr, y_val     = y_train[tr_idx], y_train[val_idx]
    print(f'\n--- Fold {fold+1}/{N_FOLDS} ---')

    # LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_tr_r, y_tr,
        eval_set=[(X_val_r, y_val)],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(-1)]
    )
    lgb_val_p = lgb_model.predict_proba(X_val_r)[:, 1]
    lgb_oof[val_idx] = lgb_val_p
    lgb_test += lgb_model.predict_proba(X_test_arr)[:, 1] / N_FOLDS
    print(f'  LGB  AUC: {roc_auc_score(y_val, lgb_val_p):.4f}')

    # XGBoost
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(X_tr_i, y_tr, eval_set=[(X_val_i, y_val)], verbose=False)
    xgb_val_p = xgb_model.predict_proba(X_val_i)[:, 1]
    xgb_oof[val_idx] = xgb_val_p
    xgb_test += xgb_model.predict_proba(X_test_imp)[:, 1] / N_FOLDS
    print(f'  XGB  AUC: {roc_auc_score(y_val, xgb_val_p):.4f}')

    # CatBoost
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_tr_i, y_tr, eval_set=(X_val_i, y_val), use_best_model=True)
    cat_val_p = cat_model.predict_proba(X_val_i)[:, 1]
    cat_oof[val_idx] = cat_val_p
    cat_test += cat_model.predict_proba(X_test_imp)[:, 1] / N_FOLDS
    print(f'  CAT  AUC: {roc_auc_score(y_val, cat_val_p):.4f}')

# ─────────────────────────────────────────────────────────────────────────────
# 5. STACKING META-LEARNER
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('Fitting Level-2 Meta-Learner (Logistic Regression)')
print('='*60)

# Stack OOF predictions as features for meta-learner
meta_train = np.column_stack([lgb_oof, xgb_oof, cat_oof])
meta_test  = np.column_stack([lgb_test, xgb_test, cat_test])

scaler = StandardScaler()
meta_train_s = scaler.fit_transform(meta_train)
meta_test_s  = scaler.transform(meta_test)

# Logistic Regression meta-learner
meta_model = LogisticRegression(C=1.0, class_weight='balanced', random_state=42)
meta_model.fit(meta_train_s, y_train)
final_oof_proba  = meta_model.predict_proba(meta_train_s)[:, 1]
final_test_proba = meta_model.predict_proba(meta_test_s)[:, 1]

# Also keep a simple weighted average for comparison
# Weights tuned heuristically: LGB slightly better than XGB/CAT on tabular
lgb_w, xgb_w, cat_w = 0.40, 0.30, 0.30
blend_oof  = lgb_w * lgb_oof  + xgb_w * xgb_oof  + cat_w * cat_oof
blend_test = lgb_w * lgb_test + xgb_w * xgb_test + cat_w * cat_test

print(f'\nLGB   OOF AUC: {roc_auc_score(y_train, lgb_oof):.4f}')
print(f'XGB   OOF AUC: {roc_auc_score(y_train, xgb_oof):.4f}')
print(f'CAT   OOF AUC: {roc_auc_score(y_train, cat_oof):.4f}')
print(f'Blend OOF AUC: {roc_auc_score(y_train, blend_oof):.4f}')
print(f'Stack OOF AUC: {roc_auc_score(y_train, final_oof_proba):.4f}')

# ─────────────────────────────────────────────────────────────────────────────
# 6. THRESHOLD OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────
thresholds = np.arange(0.01, 0.99, 0.001)

def best_threshold(y_true, proba, name=""):
    f1s = [f1_score(y_true, (proba >= t).astype(int), zero_division=0)
           for t in thresholds]
    best_idx = np.argmax(f1s)
    best_f1  = f1s[best_idx]
    best_thr = thresholds[best_idx]
    print(f'{name:<20} Best OOF F1: {best_f1:.6f} @ thr={best_thr:.3f}')
    return best_thr, best_f1

print('\n' + '='*60)
print('Threshold Optimization on OOF Predictions')
print('='*60)

_, blend_f1 = best_threshold(y_train, blend_oof,  name="Blend (40/30/30)")
blend_thr, _ = _, blend_f1
blend_thr = thresholds[np.argmax([
    f1_score(y_train, (blend_oof >= t).astype(int)) for t in thresholds
])]

_, stack_f1 = best_threshold(y_train, final_oof_proba, name="Stack (LR meta)")
stack_thr = thresholds[np.argmax([
    f1_score(y_train, (final_oof_proba >= t).astype(int)) for t in thresholds
])]

# Pick whichever OOF strategy gives higher F1
if stack_f1 >= blend_f1:
    print(f'\n→ Using STACKED probabilities (meta-learner wins)')
    best_test_proba = final_test_proba
    best_thr = stack_thr
    best_oof_f1 = stack_f1
else:
    print(f'\n→ Using BLENDED probabilities (weighted average wins)')
    best_test_proba = blend_test
    best_thr = blend_thr
    best_oof_f1 = blend_f1

print(f'\nFinal OOF F1: {best_oof_f1:.6f} @ threshold={best_thr:.3f}')

# ─────────────────────────────────────────────────────────────────────────────
# 7. GENERATE SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────
sub['TargetF1']   = (best_test_proba >= best_thr).astype(int)
sub['TargetRAUC'] = best_test_proba

pred_pos = sub['TargetF1'].sum()
print(f'\nPredicted positives: {pred_pos} / {len(sub)} ({pred_pos/len(sub):.2%})')

out_path = DATA_DIR + 'submission_v4_final.csv'
sub.to_csv(out_path, index=False)
print(f'\nSaved: {out_path}')
print('\nSubmission preview:')
print(sub.head(5))
