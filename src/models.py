# src/models.py
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    f1_score,
)

from .config import RANDOM_SEED


# ========= 回归 =========

def get_default_regression_models() -> Dict[str, RegressorMixin]:
    """
    Return a dict of simple baseline regression models.
    """
    models: Dict[str, RegressorMixin] = {
        "linear_regression": LinearRegression(),
        "random_forest_reg": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }
    return models


def evaluate_regression_model(
    model: RegressorMixin,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_weight_train: pd.Series | None = None,
    sample_weight_test: pd.Series | None = None,
) -> Dict[str, float]:
    """
    Fit a regression model and compute basic metrics.
    """
    fit_kwargs = {}
    if sample_weight_train is not None:
        fit_kwargs["sample_weight"] = sample_weight_train

    model.fit(X_train, y_train, **fit_kwargs)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred, sample_weight=sample_weight_test)
    rmse = mean_squared_error(y_test, y_pred, sample_weight=sample_weight_test, squared=False)
    r2 = r2_score(y_test, y_pred, sample_weight=sample_weight_test)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# ========= 二分类 =========

def get_default_classification_models() -> Dict[str, ClassifierMixin]:
    """
    Return a dict of simple baseline classification models.
    """
    models: Dict[str, ClassifierMixin] = {
        "logistic_reg": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "random_forest_clf": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }
    return models


def evaluate_classification_model(
    model: ClassifierMixin,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_weight_train: pd.Series | None = None,
    sample_weight_test: pd.Series | None = None,
) -> Dict[str, float]:
    """
    Fit a classification model and compute basic metrics (AUC, F1).
    """
    fit_kwargs = {}
    if sample_weight_train is not None:
        fit_kwargs["sample_weight"] = sample_weight_train

    model.fit(X_train, y_train, **fit_kwargs)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba, sample_weight=sample_weight_test)
    f1 = f1_score(y_test, y_pred, sample_weight=sample_weight_test)

    return {
        "AUC": auc,
        "F1_0.5": f1,
    }
