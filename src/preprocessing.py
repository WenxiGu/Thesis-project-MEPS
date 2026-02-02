
"""
Preprocessing utilities for the MEPS Panel 27 project.

This module implements a whitelist-based column selection strategy and a set of
lightweight cleaning steps to produce an analysis-ready "core" dataset from the
2,600+ raw MEPS columns.

Key steps
---------
1) Keep only a pre-defined set of core variables (whitelist).
2) Optionally restrict to complete two-year panel participants.
3) Recode MEPS special missing codes to NaN.
4) Sanitize expenditure variables (e.g., negative values -> NaN).
5) Drop near-empty coverage flags (PREVCOVR, MORECOVR) by default.

These helpers are used across EDA and modeling notebooks.
"""



import numpy as np
import pandas as pd

from .config import WEIGHT_COL


# ---------------------------------------------------------------------
# Core variable whitelist
# ---------------------------------------------------------------------

# 1) Sample identifiers and panel status variables
CORE_ID_VARS = [
    "DUID",      # Dwelling unit ID
    "PID",       # Person ID within household
    "DUPERSID",  # Unique person ID (key for merges)
    "PANEL",     # Panel number (27)
    "YEARIND",   # Year coverage indicator; YEARIND=1 means present in both years
    "ALL5RDS",   # =1 if in-scope and responded in all 5 rounds
    "DIED",      # =1 if died during the 2-year period
    "INST",      # =1 if ever institutionalized during the 2-year period
    "MILITARY",  # =1 if ever active duty military during the 2-year period
    "ENTRSRVY",  # Entered survey late
    "LEFTUS",    # Left the U.S. during the panel
    "OTHER",     # Other special sample status flags
]

# Survey design & weights
CORE_DESIGN_VARS = [
    WEIGHT_COL,  # LONGWT: longitudinal person weight
    "LSAQWT",    # SAQ longitudinal weight (for SAQ variables, if used)
    "VARSTR",    # Stratum identifier
    "VARPSU",    # PSU identifier (Primary Sampling Unit)
]

# 2) Demographics
CORE_DEMO_VARS = [
    "AGEY1X",
    "AGEY2X",       # Age end of Year 1 / Year 2
    "AGELSTY1",
    "AGELSTY2",     # Last observed age in each year
    "SEX",
    "RACETHX",      # Race/ethnicity combined
    "HISPANX",      # Hispanic indicator
    "EDUCYR",       # Years of education
    "REGIONY1",
    "REGIONY2",     # Census region Year 1 / Year 2
]

# 3) Family SES & household size
CORE_SES_VARS = [
    "FAMINCY1",
    "FAMINCY2",     # Total family income Y1/Y2
    "POVCATY1",
    "POVCATY2",     # Poverty category (categorical)
    "POVLEVY1",
    "POVLEVY2",     # Income-to-poverty ratio (continuous)
    "FAMSZEY1",
    "FAMSZEY2",     # Family size (end of year)
    "RUSIZEY1",
    "RUSIZEY2",     # Reporting unit size (end of year)
]

SES_NOTES = (
    "SES is measured at the family level using total family income (FAMINCY1/2) and MEPS poverty categories (POVCATY1/2). "
    "Family-level measures are preferred,because expenditures are typically financed at the household level, and many individuals (e.g., children) have no personal earnings."
    "RU (Reporting Unit) is the MEPS interviewing unit; members share a household questionnaire and it closely approximates a practical family/household unit."
)

# 4) Health insurance coverage
CORE_INS_VARS = [
    "INSCOVY1",
    "INSCOVY2",     # Full-year covered by any insurance (indicator)
    "INSURCY1",
    "INSURCY2",     # Full-year coverage type
    "UNINSY1",
    "UNINSY2",      # Total months uninsured
    "PREVCOVR",
    "MORECOVR",     # Prior coverage / multiple coverage flags (often near-empty)
]

# 5) Employment (summary variables only)
CORE_EMP_VARS = [
    "EVRWRKY1",
    "EVRWRKY2",     # Ever worked during Year 1 / Year 2
    "EMPST1",
    "EMPST2",
    "EMPST3",
    "EMPST4",
    "EMPST5",       # Employment status in each round
    "UNEMPY1X",
    "UNEMPY2X",     # Unemployment compensation amount Y1/Y2
]

# 6) Self-reported health / mental health (selected rounds)
CORE_HEALTH_STATUS_VARS = [
    "RTHLTH1",
    "RTHLTH3",
    "RTHLTH5",      # Self-rated overall health (R1/R3/R5)
    "MNHLTH1",
    "MNHLTH3",
    "MNHLTH5",      # Self-rated mental health (R1/R3/R5)
]

# 7) Key chronic condition indicators (Y1/Y2)
CORE_CHRONIC_VARS = [
    "HIBPDXY1",
    "HIBPDXY2",         # High blood pressure
    "CHDDXY1",
    "CHDDXY2",           # Coronary heart disease
    "STRKDXY1",
    "STRKDXY2",          # Stroke
    "CHOLDXY1",
    "CHOLDXY2",          # High cholesterol
    "ASTHDXY1",
    "ASTHDXY2",          # Asthma
    "DIABDXY1_M18",
    "DIABDXY2_M18",      # Diabetes (adult measure)
]

CHRONIC_NOTES = (
    "These conditions are strong predictors of high cost and inpatient risk. "
    "Keeping a small set (5–6) supports building an interpretable baseline model and enables a simple multi-morbidity index without exploding the feature space."
)

# 8) Healthcare utilization & expenditures (Y1=2022, Y2=2023)
CORE_USE_COST_VARS = [
    # Utilization: ED & inpatient counts (outcomes + predictors)
    "ERTOTY1",
    "ERTOTY2",     # Total # ER visits per year
    "IPDISY1",
    "IPDISY2",     # # hospital discharges per year

    # Total expenditures & total charges
    "TOTEXPY1",
    "TOTEXPY2",    # Total health care expenditures (all payers)
    "TOTTCHY1",
    "TOTTCHY2",    # Total health care charges (excluding Rx)

    # Amount paid by each payer (Y1/Y2)
    "TOTSLFY1",
    "TOTSLFY2",    # Self/family (out-of-pocket)
    "TOTMCRY1",
    "TOTMCRY2",    # Medicare
    "TOTMCDY1",
    "TOTMCDY2",    # Medicaid
    "TOTPRVY1",
    "TOTPRVY2",    # Private insurance
    "TOTVAY1",
    "TOTVAY2",     # VA/CHAMPVA
    "TOTTRIY1",
    "TOTTRIY2",    # TRICARE
    "TOTOFDY1",
    "TOTOFDY2",    # Other federal sources
    "TOTSTLY1",
    "TOTSTLY2",    # Other state/local sources
    "TOTWCPY1",
    "TOTWCPY2",    # Workers' compensation
    "TOTOSRY1",
    "TOTOSRY2",    # Other sources
    "TOTPTRY1",
    "TOTPTRY2",    # Private + TRICARE combined
    "TOTOTHY1",
    "TOTOTHY2",    # Other payers combined
]

# Combine all core variables
CORE_VARS = (
    CORE_ID_VARS
    + CORE_DESIGN_VARS
    + CORE_DEMO_VARS
    + CORE_SES_VARS
    + CORE_INS_VARS
    + CORE_EMP_VARS
    + CORE_HEALTH_STATUS_VARS
    + CORE_CHRONIC_VARS
    + CORE_USE_COST_VARS
)


def select_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select core variables used in the thesis/modeling from the 2,600+ raw columns.

    Notes
    -----
    - The raw dataframe (df_raw) remains available; this function returns a subset
      used for downstream EDA/modeling.

    Returns
    -------
    DataFrame
        Dataframe containing only the columns listed in CORE_VARS that are present
        in the input df.
    """
    existing = [c for c in CORE_VARS if c in df.columns]
    missing = [c for c in CORE_VARS if c not in df.columns]

    if missing:
        print("Warning: these core variables were not found and will be skipped:")
        print(missing)

    return df[existing].copy()


# ---------------------------------------------------------------------
# Data cleaning functions
# ---------------------------------------------------------------------

# MEPS special missing codes
MEPS_MISSING_CODES = [-1, -2, -3, -7, -8, -9, -13, -15]


def replace_special_missing(
    df: pd.DataFrame,
    codes: list[int] = MEPS_MISSING_CODES,
) -> pd.DataFrame:
    """
    Replace MEPS special missing codes with NaN.

    Parameters
    ----------
    df : DataFrame
        Input dataframe (numeric + non-numeric).
    codes : list[int]
        MEPS missing codes to replace, e.g. [-1, -2, -3, -7, -8, -9, -13, -15].

    Returns
    -------
    DataFrame
        Dataframe with these codes replaced by NaN (numeric columns only).
    """
    df_clean = df.copy()

    # Only apply replacement to numeric columns to avoid altering string/categorical fields.
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[num_cols] = df_clean[num_cols].replace(codes, np.nan)

    return df_clean


# Expenditure columns (negative dollar amounts -> NaN)
EXPENDITURE_COLS = [
    "TOTEXPY1",
    "TOTEXPY2",
    "TOTTCHY1",
    "TOTTCHY2",
    "TOTSLFY1",
    "TOTSLFY2",
    "TOTMCRY1",
    "TOTMCRY2",
    "TOTMCDY1",
    "TOTMCDY2",
    "TOTPRVY1",
    "TOTPRVY2",
    "TOTVAY1",
    "TOTVAY2",
    "TOTTRIY1",
    "TOTTRIY2",
    "TOTOFDY1",
    "TOTOFDY2",
    "TOTSTLY1",
    "TOTSTLY2",
    "TOTWCPY1",
    "TOTWCPY2",
    "TOTOSRY1",
    "TOTOSRY2",
    "TOTPTRY1",
    "TOTPTRY2",
    "TOTOTHY1",
    "TOTOTHY2",
]


def clean_negative_expenditures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all expenditure variables are non-negative.
    Any negative values (after missing code replacement) are set to NaN.
    """
    df_clean = df.copy()

    for col in EXPENDITURE_COLS:
        if col in df_clean.columns:
            df_clean.loc[df_clean[col] < 0, col] = np.nan

    return df_clean


def preprocess_meps(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    High-level preprocessing pipeline (light/default version).

    Steps
    -----
    1) Select core variables (whitelist).
    2) Restrict to complete panel if available (ALL5RDS==1 and YEARIND==1).
    3) Recode MEPS special missing codes to NaN.
    4) Set negative expenditures to NaN.
    5) Drop near-empty coverage flags (PREVCOVR, MORECOVR) if missingness > 80%.

    Notes
    -----
    More aggressive steps (e.g., winsorization, outlier handling) are handled later
    in feature engineering/modeling.
    """
    df = select_core_columns(df_raw).copy()

    # Keep complete two-year panel participants (if these variables exist)
    if "ALL5RDS" in df.columns:
        df = df[df["ALL5RDS"] == 1]
    if "YEARIND" in df.columns:
        df = df[df["YEARIND"] == 1]

    # Recode special missing codes
    df = replace_special_missing(df)

    # Negative expenditures -> NaN
    df = clean_negative_expenditures(df)

    # Drop coverage flags that are almost entirely missing (not used as features by default)
    for col in ["PREVCOVR", "MORECOVR"]:
        if col in df.columns and df[col].isna().mean() > 0.8:
            df = df.drop(columns=col)

    return df









