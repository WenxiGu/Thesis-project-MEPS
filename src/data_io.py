
"""
I/O helpers for the MEPS Panel 27 thesis project.

This module provides small, reusable functions to:
- Load the raw MEPS longitudinal file (HC-252) from Excel.
- Save the processed, analysis-ready dataset to disk (Parquet by default).


"""


from pathlib import Path

import pandas as pd

from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def load_raw_meps(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load the raw MEPS longitudinal dataset (HC-252) from an Excel file.

    Parameters
    ----------
    path : Path | str | None
        Custom path to the Excel file. If None, uses RAW_DATA_PATH from config.py.

    Returns
    -------
    pandas.DataFrame
        Raw MEPS data as loaded from Excel.
    """
    xls_path = RAW_DATA_PATH if path is None else Path(path)
    return pd.read_excel(xls_path)


def save_processed_meps(df: pd.DataFrame, path: Path | str | None = None) -> None:
    """
    Save the processed MEPS dataset to disk (Parquet by default).

    Parameters
    ----------
    df : pandas.DataFrame
        Processed dataset.
    path : Path | str | None
        Output path. If None, uses PROCESSED_DATA_PATH from config.py.

    Notes
    -----
    - If the provided path ends with ".parquet", the dataset is saved as Parquet.
    - If it ends with ".csv", the dataset is saved as CSV.
    - Otherwise, the function defaults to Parquet and appends ".parquet".
    """
    out_path = PROCESSED_DATA_PATH if path is None else Path(path)
    suffix = out_path.suffix.lower()

    if suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    elif suffix == ".csv":
        df.to_csv(out_path, index=False)
    else:
        # Default to Parquet
        df.to_parquet(Path(str(out_path) + ".parquet"), index=False)
