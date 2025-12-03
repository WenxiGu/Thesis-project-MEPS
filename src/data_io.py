
from typing import Optional
import pandas as pd

from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def load_raw_meps(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw MEPS longitudinal data (HC-252, Excel).

    Parameters
    ----------
    path : str, optional
        Custom path to the Excel file. If None, uses RAW_DATA_PATH from config.

    Returns
    -------
    df : pandas.DataFrame
        Raw MEPS data as loaded from Excel.
    """
    if path is None:
        path = RAW_DATA_PATH

    df = pd.read_excel(path)
    return df


def save_processed_meps(df: pd.DataFrame, path: Optional[str] = None) -> None:
    """
    Save processed MEPS dataset to disk (parquet by default).

    Parameters
    ----------
    df : pandas.DataFrame
        Processed dataset.
    path : str, optional
        Output path. If None, uses PROCESSED_DATA_PATH from config.
    """
    if path is None:
        path = PROCESSED_DATA_PATH

    path = str(path)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif path.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        # 默认 parquet
        df.to_parquet(path + ".parquet", index=False)
