"""
Train and export the preferred models for the MEPS Panel 27 project.

This script loads the engineered feature table (data/df_feat.parquet), applies a
fixed feature set (num_cols + cat_cols), trains the selected "best" models for each
target, and saves reusable artifacts for deployment (e.g., Streamlit inference).

Models trained
--------------
- HIGHCOST_Y2: RandomForestClassifier (class_weight="balanced_subsample")
- ANY_ED_Y2:   XGBClassifier
- ANY_IP_Y2:   RandomForestClassifier (class_weight="balanced_subsample")
- LOG_TOTEXPY2: xgboost.train Booster with early stopping (regression)

Training protocol
-----------------
- Train/validation/test split = 60/20/20 (stratified for classification).
- Preprocessing: numeric median imputation; categorical most-frequent imputation
  + one-hot encoding (handle_unknown="ignore").
- Classification threshold selection: choose the probability threshold on the
  validation set that maximizes F1, then refit on train+val and evaluate on test.
- Regression: early stopping on validation RMSE; refit on train+val using the best
  number of boosting iterations; evaluate on test with RMSE_log, MAE_log, and R².

Outputs
-------
Artifacts are saved under results/model_artifacts_*:
- sklearn Pipelines saved as .joblib (+ .meta.json)
- XGBoost regression saved as preprocess.joblib + booster.json (+ .meta.json)

Usage
-----
Run as a script (from anywhere inside the repository):
    python scripts/train_and_save_preferred_models.py
"""







from __future__ import annotations

from pathlib import Path
import json
import joblib
import datetime as dt

import numpy as np
import pandas as pd

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    mean_absolute_error, r2_score, mean_squared_error
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import xgboost as xgb

from sklearn.model_selection import train_test_split


# -------------------------
# 0) Paths (auto-detect root)
# -------------------------
def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(8):
        if (cur / "src").exists() and (cur / "notebooks").exists() and (cur / "results").exists():
            return cur
        cur = cur.parent
    raise FileNotFoundError("Could not find project root. Run this from inside the repository.")

PROJECT_ROOT = find_project_root(Path.cwd())
DATA_PATH = PROJECT_ROOT / "data" / "df_feat.parquet"
ART_DIR = PROJECT_ROOT / "results" / "model_artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Missing data/df_feat.parquet. This row-level derived dataset is not tracked in git. "
        "Download MEPS HC-252 from AHRQ/MEPS, save it as data/h252.xlsx, and run the "
        "preprocessing/feature-engineering notebooks first. Official download page: "
        "https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-252"
    )


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# -------------------------
# 1) FINAL thesis feature set (used in all models) - see Notebook 02 for details
# -------------------------
cat_cols = [
    "RACE_ETH",
    "REGIONY1_CAT",
    "EDU_GROUP",
    "POVCATY1_CAT",
    "FAMSIZE_Y1_GRP",
    "INS_TYPE_Y1",
]

num_cols = [
    # demographics / SES
    "AGE",
    "SEX_BIN",
    "LOG_FAMINCY1",
    "FAMSIZE_Y1",

    # employment
    "WORKED_Y1",
    "ANY_UNEMP_COMP_Y1",
    "LOG_UNEMP_COMP_Y1",
    "EMP_INFO_R12",
    "EMP_ATTACHED_ANY_R12_FILL0",

    # health status baseline
    "RTHLTH1_FAIRPOOR",
    "MNHLTH1_FAIRPOOR",

    # chronic conditions baseline
    "HIBPDXY1_BIN",
    "CHDDXY1_BIN",
    "STRKDXY1_BIN",
    "CHOLDXY1_BIN",
    "ASTHDXY1_BIN",
    "DIABDXY1_M18_BIN",

    # baseline utilisation/cost
    "LOG_TOTEXPY1",
    "ANY_ED_Y1",
    "ANY_IP_Y1",
]

FEATURES = num_cols + cat_cols


# -------------------------
# 2) Preprocess (median impute numeric; mode+onehot categorical)
# -------------------------
def make_preprocess(num_cols, cat_cols, *, scale_numeric=False) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    # trees: scale_numeric usually False
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=num_steps), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
    )
    return preprocess


# -------------------------
# 3) Train/val/test split (60/20/20)
# -------------------------
def split_train_val_test(X, y, *, random_state=42, stratify=False):
    strat = y if stratify else None
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.40, random_state=random_state, stratify=strat
    )
    strat2 = y_tmp if stratify else None
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=random_state, stratify=strat2
    )
    return X_tr, X_va, X_te, y_tr, y_va, y_te


# -------------------------
# 4) Saving helpers
# -------------------------
def save_sklearn_artifact(name: str, pipeline, meta: dict):
    path = ART_DIR / f"{name}.joblib"
    joblib.dump(pipeline, path)
    meta_path = ART_DIR / f"{name}.meta.json"
    meta2 = {
        **meta,
        "saved_at": dt.datetime.now().isoformat(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": getattr(xgb, "__version__", None),
    }
    meta_path.write_text(json.dumps(meta2, indent=2))
    return str(path), str(meta_path)

def save_xgb_booster_reg(name: str, preprocess, booster: xgb.Booster, meta: dict):
    pre_path = ART_DIR / f"{name}.preprocess.joblib"
    model_path = ART_DIR / f"{name}.booster.json"
    meta_path = ART_DIR / f"{name}.meta.json"

    joblib.dump(preprocess, pre_path)
    booster.save_model(model_path)

    meta2 = {
        **meta,
        "saved_at": dt.datetime.now().isoformat(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": getattr(xgb, "__version__", None),
    }
    meta_path.write_text(json.dumps(meta2, indent=2))
    return str(pre_path), str(model_path), str(meta_path)


# -------------------------
# 5) Threshold selection for classification (maximize F1 on val)
# -------------------------
def best_threshold_by_f1(y_true, proba, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.50, -1.0
    for t in grid:
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


# -------------------------
# 6) Train + save a classification pipeline
# -------------------------
def train_and_save_classifier(
    df: pd.DataFrame,
    *,
    target: str,
    name: str,
    estimator,
    stratify=True
):
    tmp = df[FEATURES + [target]].dropna()
    X = tmp[FEATURES].copy()
    y = tmp[target].astype(int)

    X_tr, X_va, X_te, y_tr, y_va, y_te = split_train_val_test(
        X, y, random_state=RANDOM_SEED, stratify=stratify
    )

    pre = make_preprocess(num_cols, cat_cols, scale_numeric=False)
    pipe = Pipeline([("preprocess", pre), ("model", estimator)])
    pipe.fit(X_tr, y_tr)

    va_proba = pipe.predict_proba(X_va)[:, 1]
    best_t, best_f1 = best_threshold_by_f1(y_va.values, va_proba)

    # Refit on train+val
    X_tv = pd.concat([X_tr, X_va], axis=0)
    y_tv = pd.concat([y_tr, y_va], axis=0)
    pipe.fit(X_tv, y_tv)

    te_proba = pipe.predict_proba(X_te)[:, 1]
    te_pred = (te_proba >= best_t).astype(int)

    meta = {
        "target": target,
        "model": type(estimator).__name__,
        "feature_cols": FEATURES,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "best_threshold": best_t,
        "test_metrics": {
            "AUC": float(roc_auc_score(y_te, te_proba)),
            "PR_AUC": float(average_precision_score(y_te, te_proba)),
            "F1_at_best_t": float(f1_score(y_te, te_pred, zero_division=0)),
        },
        "train_size": int(len(X_tv)),
        "test_size": int(len(X_te)),
        "prevalence_test": float(y_te.mean()),
    }

    paths = save_sklearn_artifact(name, pipe, meta)
    return meta, paths


# -------------------------
# 7) Train + save XGB regression booster (early stopping)
# -------------------------
def train_and_save_regression_xgb_booster(
    df: pd.DataFrame,
    *,
    target: str = "LOG_TOTEXPY2",
    name: str = "reg_log_totexpy2_xgb_es"
):
    tmp = df[FEATURES + [target]].dropna()
    X = tmp[FEATURES].copy()
    y = tmp[target].astype(float)

    X_tr, X_va, X_te, y_tr, y_va, y_te = split_train_val_test(
        X, y, random_state=RANDOM_SEED, stratify=False
    )

    pre = make_preprocess(num_cols, cat_cols, scale_numeric=False)
    X_tr_p = pre.fit_transform(X_tr)
    X_va_p = pre.transform(X_va)
    X_te_p = pre.transform(X_te)

    dtrain = xgb.DMatrix(X_tr_p, label=y_tr)
    dval   = xgb.DMatrix(X_va_p, label=y_va)
    dtest  = xgb.DMatrix(X_te_p, label=y_te)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "seed": RANDOM_SEED,

        # the best ES config
        "max_depth": 3,
        "eta": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "gamma": 0.5,
        "lambda": 5.0,
        "alpha": 0.0,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=[(dval, "val")],
        early_stopping_rounds=200,
        verbose_eval=False,
    )
    best_iter = int(booster.best_iteration)

    # Refit on train+val with best_iter+1; refit preprocess on train+val
    X_tv = pd.concat([X_tr, X_va], axis=0)
    y_tv = pd.concat([y_tr, y_va], axis=0)
    pre_tv = make_preprocess(num_cols, cat_cols, scale_numeric=False)
    X_tv_p = pre_tv.fit_transform(X_tv)
    dtrain_tv = xgb.DMatrix(X_tv_p, label=y_tv)

    booster_final = xgb.train(
        params=params,
        dtrain=dtrain_tv,
        num_boost_round=best_iter + 1,
        verbose_eval=False,
    )

    # test eval
    X_te_p2 = pre_tv.transform(X_te)
    dtest2 = xgb.DMatrix(X_te_p2, label=y_te)
    yhat = booster_final.predict(dtest2)

    rmse = float(np.sqrt(mean_squared_error(y_te, yhat)))
    mae = float(mean_absolute_error(y_te, yhat))
    r2  = float(r2_score(y_te, yhat))

    meta = {
        "target": target,
        "model": "xgboost.train",
        "feature_cols": FEATURES,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "best_iteration": best_iter,
        "params": params,
        "test_metrics": {"RMSE_log": rmse, "MAE_log": mae, "R2": r2},
        "train_size": int(len(X_tv)),
        "test_size": int(len(X_te)),
    }

    paths = save_xgb_booster_reg(name, preprocess=pre_tv, booster=booster_final, meta=meta)
    return meta, paths


def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("Reading:", DATA_PATH)
    df = pd.read_parquet(DATA_PATH)

    # quick column check
    miss = [c for c in FEATURES if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required feature columns: {miss}")

    # -------- HIGHCOST (RF) --------
    hc_est = RandomForestClassifier(
        n_estimators=800, max_depth=16, min_samples_leaf=3,
        random_state=RANDOM_SEED, n_jobs=-1, class_weight="balanced_subsample"
    )
    hc_meta, hc_paths = train_and_save_classifier(df, target="HIGHCOST_Y2", name="clf_highcost_rf", estimator=hc_est, stratify=True)
    print("Saved HIGHCOST:", hc_paths, "best_t:", hc_meta["best_threshold"])

    # -------- ANY_ED (XGB) --------
    ed_est = XGBClassifier(
        n_estimators=800, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        random_state=RANDOM_SEED, n_jobs=-1, tree_method="hist",
        eval_metric="logloss"
    )
    ed_meta, ed_paths = train_and_save_classifier(df, target="ANY_ED_Y2", name="clf_ed_xgb", estimator=ed_est, stratify=True)
    print("Saved ANY_ED:", ed_paths, "best_t:", ed_meta["best_threshold"])

    # -------- ANY_IP (RF tuned) --------

    rf_ip_params = dict(
    n_estimators=1500,
    max_depth=18,
    min_samples_leaf=5,
    min_samples_split=10,
    max_features=0.2,
    max_samples=0.7,
    bootstrap=True,
    )

    ip_est = RandomForestClassifier(
    **rf_ip_params,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    class_weight="balanced_subsample",
   )

    ip_meta, ip_paths = train_and_save_classifier(df, target="ANY_IP_Y2", name="clf_ip_rf", estimator=ip_est, stratify=True)
    print("Saved ANY_IP:", ip_paths, "best_t:", ip_meta["best_threshold"])

    # -------- LOG_TOTEXPY2 regression (XGB booster ES) --------
    reg_meta, reg_paths = train_and_save_regression_xgb_booster(df, target="LOG_TOTEXPY2", name="reg_log_totexpy2_xgb_es")
    print("Saved REG:", reg_paths, "best_iter:", reg_meta["best_iteration"])

    print("\nDONE. Artifacts are in:", ART_DIR)


if __name__ == "__main__":
    main()
