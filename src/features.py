
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from .config import (
    REG_TARGET_TOTEXPY2,
    REG_BASELINE_TOTEXPY1,
    CLASS_TARGET_ER_Y2,
    CLASS_TARGET_IP_Y2,
    WEIGHT_COL,
)


def add_simple_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a few simple engineered features.

    Examples (modify according to available columns):
    - log(1 + TOTEXPY1)
    - age groups

    Returns
    -------
    df_feat : pandas.DataFrame
        Dataframe with additional feature columns.
    """
    df_feat = df.copy()

    # log(1 + TOTEXPY1)
    if REG_BASELINE_TOTEXPY1 in df_feat.columns:
        df_feat["LOG1P_TOTEXPY1"] = np.log1p(df_feat[REG_BASELINE_TOTEXPY1])

    # 例：根据年龄创建 age group
    if "AGE23X" in df_feat.columns:
        df_feat["AGE_GROUP"] = pd.cut(
            df_feat["AGE23X"],
            bins=[0, 17, 44, 64, 120],
            labels=["0-17", "18-44", "45-64", "65+"],
            right=True,
        )

    # TODO: 可以在这里继续加慢性病个数、是否多病共存等特征

    return df_feat


def build_regression_dataset(
    df: pd.DataFrame,
    target_col: str = REG_TARGET_TOTEXPY2,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build feature matrix X and target y for regression.

    Returns
    -------
    X : DataFrame
        Features (no target, no weight).
    y : Series
        Regression target.
    sample_weight : Series
        Sample weights if available, otherwise all ones.
    """
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    # 简单示例：将明显不能用于建模的变量排除掉
    drop_cols = [target_col, WEIGHT_COL]
    id_cols = [c for c in ["DUPERSID", "DUID", "PID"] if c in df.columns]
    drop_cols.extend(id_cols)

    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df[target_col]

    if WEIGHT_COL in df.columns:
        sample_weight = df[WEIGHT_COL]
    else:
        sample_weight = pd.Series(np.ones(len(df)), index=df.index, name="weight")

    return X, y, sample_weight


def build_binary_classification_dataset(
    df: pd.DataFrame,
    label_col: str = CLASS_TARGET_ER_Y2,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build feature matrix X and binary label y for classification.

    label_col should already be 0/1.

    Returns
    -------
    X : DataFrame
    y : Series of 0/1
    sample_weight : Series
    """
    df = df.copy()
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe.")

    drop_cols = [label_col, WEIGHT_COL]
    id_cols = [c for c in ["DUPERSID", "DUID", "PID"] if c in df.columns]
    drop_cols.extend(id_cols)

    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df[label_col]

    if WEIGHT_COL in df.columns:
        sample_weight = df[WEIGHT_COL]
    else:
        sample_weight = pd.Series(np.ones(len(df)), index=df.index, name="weight")

    return X, y, sample_weight
