
from typing import Iterable, List, Optional
import numpy as np
import pandas as pd

from .config import WEIGHT_COL


# MEPS 中常见的“特殊取值”编码（需要和文档核对）
DEFAULT_SPECIAL_MISSING = [-1, -7, -8, -9]


def replace_special_missing(
    df: pd.DataFrame,
    codes: Iterable[int] = DEFAULT_SPECIAL_MISSING
) -> pd.DataFrame:
    """
    Replace common MEPS special missing codes with NaN.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataframe.
    codes : iterable of int
        Codes to replace by NaN, e.g. [-1, -7, -8, -9].

    Returns
    -------
    df_clean : pandas.DataFrame
        Dataframe with codes replaced by NaN.
    """
    df_clean = df.copy()
    for code in codes:
        df_clean = df_clean.replace(code, np.nan)
    return df_clean


def filter_complete_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep individuals with complete panel (e.g. ALL5RDS==1), if column exists.

    If the ALL5RDS column is not present, the dataframe is returned unchanged.

    Returns
    -------
    df_filtered : pandas.DataFrame
    """
    df_filtered = df.copy()
    if "ALL5RDS" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["ALL5RDS"] == 1]
    return df_filtered


def select_core_columns(
    df: pd.DataFrame,
    extra_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Select a core set of columns for modeling.

    This is a placeholder where you will later specify:
    - ID column (e.g. DUID, PID, DUPERSID...)
    - demographic vars
    - socio-economic vars
    - baseline health / utilization vars
    - targets (TOTEXPY1, TOTEXPY2, ER, IP, etc.)
    - weights and survey design variables

    For now, this function just returns df unchanged.

    Parameters
    ----------
    df : pandas.DataFrame
    extra_cols : list of str, optional
        Extra columns you want to keep.

    Returns
    -------
    df_sel : pandas.DataFrame
    """
    # TODO: 根据文档，真正列出你要保留的核心变量列表
    # 如：
    # core_cols = [
    #     "DUPERSID", "AGE23X", "SEX", "REGION23", "RACETHX",
    #     "TOTEXPY1", "TOTEXPY2",
    #     "ERVISITY1", "ER_VISITY2", "IPDISY1", "IPDISY2",
    #     WEIGHT_COL, "VARSTR", "VARPSU",
    # ]
    # if extra_cols is not None:
    #     core_cols.extend(extra_cols)
    # core_cols = [c for c in core_cols if c in df.columns]
    #
    # return df[core_cols].copy()

    if extra_cols is not None:
        # 临时行为：如果传 extra_cols，就只保留这些存在的
        cols = [c for c in extra_cols if c in df.columns]
        return df[cols].copy()

    return df.copy()


def preprocess_meps(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    High-level preprocessing pipeline:
    1. Filter complete panel individuals (if ALL5RDS available).
    2. Replace special missing codes by NaN.
    3. Optionally, select core columns (currently returns all columns).

    Returns
    -------
    df_pre : pandas.DataFrame
        Preprocessed dataframe.
    """
    df = filter_complete_panel(df_raw)
    df = replace_special_missing(df)
    df = select_core_columns(df)

    # 可以在这里顺便处理一些明显的问题，例如负的费用值：
    # for col in ["TOTEXPY1", "TOTEXPY2"]:
    #     if col in df.columns:
    #         df.loc[df[col] < 0, col] = np.nan

    return df
