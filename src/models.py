
"""
Modeling utilities for the MEPS Panel 27 thesis project.

This module provides reusable baseline modeling helpers for both regression and
classification tasks. It standardizes:
- feature/target definitions (via explicit column lists),
- preprocessing for mixed numeric/categorical predictors (imputation + one-hot),
- train/validation/test splitting,
- baseline model training and evaluation.


"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
)

# ---------------------------------------------------------------------
# target column names (imported from config)
# ---------------------------------------------------------------------


from .config import (
    REG_TARGET_TOTEXPY2_LOG,
    CLASS_TARGET_HIGHCOST_Y2,
    CLASS_TARGET_ANY_ED_Y2,
    CLASS_TARGET_ANY_IP_Y2,
)

# All modeling targets used in this project (1 regression + 3 classification)
TARGET_COLS = [
    REG_TARGET_TOTEXPY2_LOG,
    CLASS_TARGET_HIGHCOST_Y2,
    CLASS_TARGET_ANY_ED_Y2,
    CLASS_TARGET_ANY_IP_Y2,
]

REG_TARGET_COL = REG_TARGET_TOTEXPY2_LOG
CLF_TARGET_COLS = [CLASS_TARGET_HIGHCOST_Y2, CLASS_TARGET_ANY_ED_Y2, CLASS_TARGET_ANY_IP_Y2]


import joblib



# ---------------------------
# Utilities
# ---------------------------

def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split into train/val/test.
    test_size and val_size are fractions of the full dataset.
    If stratify=True, preserves class balance across splits.
    """
    X = X.copy()
    y = y.copy()

    strat1 = y if stratify else None
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat1
    )

    val_frac_of_trainval = val_size / (1 - test_size)  # e.g., 0.2/0.8 = 0.25
    strat2 = y_trainval if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac_of_trainval, random_state=random_state, stratify=strat2
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def make_preprocess(
    num_cols: list[str],
    cat_cols: list[str],
    *,
    scale_numeric: bool = True
) -> ColumnTransformer:
    """
    Build preprocessing transformer:
    - numeric: median impute (+ optional scaling)
    - categorical: most_frequent impute + one-hot (ignore unseen categories)
    """
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

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


def best_threshold_by_f1(y_true: np.ndarray, proba: np.ndarray, grid: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Find threshold that maximizes F1 on validation set.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)

    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def save_pipeline(model: Pipeline, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


# ---------------------------
# Results containers
# ---------------------------

@dataclass
class RegressionResult:
    model: Pipeline
    valid_metrics: Dict[str, float]
    test_metrics: Dict[str, float]


@dataclass
class ClassificationResult:
    model: Pipeline
    best_threshold: float
    valid_metrics: Dict[str, float]
    test_metrics: Dict[str, float]


# ---------------------------
# Baseline models
# ---------------------------

def run_regression_baseline(
    df: pd.DataFrame,
    *,
    target_col: str,
    num_cols: list[str],
    cat_cols: list[str],
    random_state: int = 42,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    max_iter: int = 20000,
    scale_numeric: bool = True,
    
) -> RegressionResult:
    """
    Baseline regression on target_col using ElasticNet.
    Uses train/val/test split. Reports metrics on val and test.
    """
    features = cat_cols + num_cols  # explicit predictor list
    X = df[features].copy()
    y = df[target_col].copy()
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    preprocess = make_preprocess(num_cols, cat_cols, scale_numeric=scale_numeric)
    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=random_state)),
    ])

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y, random_state=random_state, stratify=False
    )


    model.fit(X_train, y_train)

    # validate
    val_pred = model.predict(X_val)
    valid_metrics = {
        "MAE_log": mean_absolute_error(y_val, val_pred),
        "RMSE_log": root_mean_squared_error(y_val, val_pred),
        "R2": r2_score(y_val, val_pred),
    }

    # test
    test_pred = model.predict(X_test)
    test_metrics = {
        "MAE_log": mean_absolute_error(y_test, test_pred),
        "RMSE_log": root_mean_squared_error(y_test, test_pred),
        "R2": r2_score(y_test, test_pred),
    }

    return RegressionResult(model=model, valid_metrics=valid_metrics, test_metrics=test_metrics)


def run_classification_baseline(
    df: pd.DataFrame,
    *,
    target_col: str,
    num_cols: list[str],
    cat_cols: list[str],
    random_state: int = 42,
    max_iter: int = 2000,
    class_weight: str | Dict[int, float] | None = "balanced",
    scale_numeric: bool = True,
    
) -> ClassificationResult:
    """
    Baseline classification using LogisticRegression.
    Uses train/val/test split with stratification.
    Chooses best threshold on validation set by F1.
    Reports AUC, PR-AUC and F1@best_t on test.
    """
    features = cat_cols + num_cols  # explicit predictor list
    X = df[features].copy()
    y = df[target_col].copy().astype(int)  # assumes 0/1
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    preprocess = make_preprocess(num_cols, cat_cols, scale_numeric=scale_numeric)
    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=max_iter, class_weight=class_weight)),
    ])

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y, random_state=random_state, stratify=True
    )

   

    model.fit(X_train, y_train)

    # validation -> choose threshold
    val_proba = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = best_threshold_by_f1(y_val.values, val_proba)

    valid_metrics = {
        "AUC": roc_auc_score(y_val, val_proba),
        "PR_AUC": average_precision_score(y_val, val_proba),
        "best_t": best_t,
        "best_F1": best_f1,
    }

    # test (use threshold chosen on validation)
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_t).astype(int)

    test_metrics = {
        "AUC": roc_auc_score(y_test, test_proba),
        "PR_AUC": average_precision_score(y_test, test_proba),
        "F1_at_best_t": f1_score(y_test, test_pred),
    }

    return ClassificationResult(model=model, best_threshold=best_t, valid_metrics=valid_metrics, test_metrics=test_metrics

) 
