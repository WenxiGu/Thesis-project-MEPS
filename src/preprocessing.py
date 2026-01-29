from typing import Iterable, List, Optional
import numpy as np
import pandas as pd

from .config import WEIGHT_COL



# 1. Sample identifiers and panel status variables
CORE_ID_VARS = [
    "DUID",      # dwelling unit ID 
    "PID",       # person ID within household
    "DUPERSID",  # unique person ID (used to merge panel, intermediate, and event files).
    "PANEL",     # panel number (27)
    "YEARIND",   # in both years / only 2022 / only 2023; YEARIND=1 means present in both 2022 and 2023 (recommended longitudinal subsample).
    "ALL5RDS",   # =1 if in-scope and responded all 5 rounds
    "DIED",      # =1 if died during 2-year period
    "INST",      # ever institutionalized during panel (e.g., long-term care) during the two-year period.
    "MILITARY",  # ever active duty military during the two-year period.
    "ENTRSRVY",  # entered survey late
    "LEFTUS",    # left US during panel
    "OTHER",     # other special sample status (e.g., entered mid-panel, left the U.S., etc.).
]

#“We restricted the sample to individuals present from the beginning of the panel (ENTRSRVY=0), with data collected in all five rounds (ALL5RDS=1) …”


# Survey design & weights
CORE_DESIGN_VARS = [
    WEIGHT_COL,  # LONGWT: longitudinal person weight
    "LSAQWT",    # SAQ longitudinal weight (for SAQ variables, if used)
    "VARSTR",    # stratum identifier (sampling stratum)
    "VARPSU",    # PSU identifier (Primary Sampling Unit)
]

# VARSTR and VARPSU define the complex survey design and are used for survey-weighted estimation and variance calculation.

# SAQ = Self-Administered Questionnaire (completed by adults in Rounds 2 & 4)


# 2. Demographics
CORE_DEMO_VARS = [
    "AGEY1X", "AGEY2X",   # age end of year1 / year2
    "AGELSTY1", "AGELSTY2",  # last age in each year
    "SEX",        # sex
    "RACETHX",    # race/ethnicity combined
    "HISPANX",    # Hispanic indicator
    "EDUCYR",     # years of education
    "REGIONY1", "REGIONY2",  # census region year1 / year2
]

# 3. Family socioeconomic status (SES) & household size
CORE_SES_VARS = [
    "FAMINCY1", "FAMINCY2",   # family total income Y1/Y2
    "POVCATY1", "POVCATY2",   # income as % of poverty line (categorical): poverty category / income bracket
    "POVLEVY1", "POVLEVY2",   # income-to-poverty ratio (continuous): how many times the poverty line
    "FAMSZEY1", "FAMSZEY2",   # family size (end of year)
    "RUSIZEY1", "RUSIZEY2",   # reporting unit size (end of year)
]

# Socio-economic status (SES) was primarily measured at the family level using total family income (FAMINCY1, FAMINCY2) and official MEPS poverty categories (POVCATY1, POVCATY2). 
# We chose family-level rather than individual income because medical expenditures are typically financed at the household level and 
# many individuals in the sample (e.g. children, non-working spouses) have no personal earnings despite living in high-income households.

# Reporting Unit (RU) = the interviewing unit used by MEPS
# People in the same RU share one set of questionnaires, typically answered by a single household respondent.

# An RU is not always identical to a legal 'family', but it is very close to a practical household/family unit.


# 4. Health insurance coverage
CORE_INS_VARS = [
    "INSCOVY1", "INSCOVY2",   # full-year covered by any insurance (indicator).
    "INSURCY1", "INSURCY2",   # full-year coverage type (e.g., <65 any private, <65 public only, <65 uninsured, 65+ Medicare only).
    "UNINSY1", "UNINSY2",     # total months uninsured in year1/year2.
    "PREVCOVR", "MORECOVR",   # indicators for prior coverage / multiple coverage.
]



# 5. Employment (keep summary variables only)
CORE_EMP_VARS = [
    "EVRWRKY1", "EVRWRKY2",   # ever worked during year1 / year2
    "EMPST1", "EMPST2", "EMPST3", "EMPST4", "EMPST5",  # employment status in each round
    "UNEMPY1X", "UNEMPY2X",   # unemployed compensation amount Y1/Y2
]


# 6. Self-reported health / mental health
CORE_HEALTH_STATUS_VARS = [
    "RTHLTH1", "RTHLTH3", "RTHLTH5",   # perceived health status R1/R3/R5 (self-rated overall health: excellent/very good/good/fair/poor)
    "MNHLTH1", "MNHLTH3", "MNHLTH5",   # perceived mental health R1/R3/R5 (self-rated mental health)
]





# 7. Key chronic condition indicators (Y1/Y2)
CORE_CHRONIC_VARS = [
    "HIBPDXY1", "HIBPDXY2",        # high blood pressure diagnosis
    "CHDDXY1", "CHDDXY2",          # coronary heart disease diagnosis
    "STRKDXY1", "STRKDXY2",        # stroke diagnosis
    "CHOLDXY1", "CHOLDXY2",        # high cholesterol diagnosis
    "ASTHDXY1", "ASTHDXY2",        # asthma diagnosis
    "DIABDXY1_M18", "DIABDXY2_M18" # diabetes diagnosis 2022/23 
]


# These conditions are strong predictors of high cost and inpatient risk.

# Keeping a small set (5–6) supports building an interpretable baseline model and enables a simple multi-morbidity index (count of chronic conditions) without exploding the feature space.


# 8. Healthcare utilization & expenditures (Y1=2022, Y2=2023)
CORE_USE_COST_VARS = [
    # Utilization: ED & inpatient counts (outcomes + predictors)
    "ERTOTY1", "ERTOTY2",   # total # ER visits per year
    "IPDISY1", "IPDISY2",   # # hospital discharges per year

    # Total expenditures & total charges
    "TOTEXPY1", "TOTEXPY2",     # total health care expenditures (all payers)
    "TOTTCHY1", "TOTTCHY2",     # total health care charges (excl Rx)

    # Amount paid by each payer (Y1 / Y2) — structural information + potential features
    "TOTSLFY1", "TOTSLFY2",   # total paid by self/family 
    "TOTMCRY1", "TOTMCRY2",   # total paid by Medicare 
    "TOTMCDY1", "TOTMCDY2",   # total paid by Medicaid 
    "TOTPRVY1", "TOTPRVY2",   # total paid by private insurance 
    "TOTVAY1", "TOTVAY2",     # total paid by VA/CHAMPVA 
    "TOTTRIY1", "TOTTRIY2",   # total paid by TRICARE  
    "TOTOFDY1", "TOTOFDY2",   # other federal sources
    "TOTSTLY1", "TOTSTLY2",   # other state/local sources
    "TOTWCPY1", "TOTWCPY2",   # workers’ compensation
    "TOTOSRY1", "TOTOSRY2",   # other sources
    "TOTPTRY1", "TOTPTRY2",   # private + TRICARE combined
    "TOTOTHY1", "TOTOTHY2",   # other payers combined 
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
Select core variables used in the thesis / modeling from the 2,600+ raw columns.
All other columns remain available in df_raw but are excluded from downstream EDA/modeling.

Returns
-------
df_sel : DataFrame
    DataFrame containing only columns listed in CORE_VARS that are present in the input df.
"""
    existing = [c for c in CORE_VARS if c in df.columns]
    missing = [c for c in CORE_VARS if c not in df.columns]

    if missing:
        print("Warning: these core vars not found in df and will be skipped:")
        print(missing)

    return df[existing].copy()


## data cleaning functions 

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
    codes : list of int
        MEPS missing codes to replace, e.g. [-1, -2, -3, -7, -8, -9, -13, -15].

    Returns
    -------
    df_clean : DataFrame
        Dataframe with these codes replaced by NaN.
    """
    df_clean = df.copy()
    # Only apply replacement to numeric columns to avoid altering string/categorical fields.
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[num_cols] = df_clean[num_cols].replace(codes, np.nan)
    return df_clean



# Simple cleaning: negative dollar amounts -> NaN

EXPENDITURE_COLS = [
    "TOTEXPY1", "TOTEXPY2",
    "TOTTCHY1", "TOTTCHY2",
    "TOTSLFY1", "TOTSLFY2",
    "TOTMCRY1", "TOTMCRY2",
    "TOTMCDY1", "TOTMCDY2",
    "TOTPRVY1", "TOTPRVY2",
    "TOTVAY1",  "TOTVAY2",
    "TOTTRIY1", "TOTTRIY2",
    "TOTOFDY1", "TOTOFDY2",
    "TOTSTLY1", "TOTSTLY2",
    "TOTWCPY1", "TOTWCPY2",
    "TOTOSRY1", "TOTOSRY2",
    "TOTPTRY1", "TOTPTRY2",
    "TOTOTHY1", "TOTOTHY2",
]


def clean_negative_expenditures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all expenditure variables are non-negative.
    Any negative values (after missing codes replacement)
    are set to NaN.
    """
    df_clean = df.copy()
    for col in EXPENDITURE_COLS:
        if col in df_clean.columns:
            df_clean.loc[df_clean[col] < 0, col] = np.nan
    return df_clean




def preprocess_meps(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    High-level preprocessing pipeline (light / default version):

    1. Select core variables.
    2. Optionally restrict to complete panel (ALL5RDS == 1 & YEARIND == 1).
    3. Replace MEPS special missing codes with NaN.
    4. Clean negative expenditure values.
    5. Drop coverage flags that are almost entirely missing (PREVCOVR, MORECOVR).

    More aggressive steps (winsorization, dropping extreme outliers)
    are handled later in feature engineering / modeling.
    """
    df = select_core_columns(df_raw).copy()

    
    if "ALL5RDS" in df.columns:
        df = df[df["ALL5RDS"] == 1]
    if "YEARIND" in df.columns:
        df = df[df["YEARIND"] == 1]

    # Replace special MEPS missing codes with NaN
    df = replace_special_missing(df)

    # Negative expenditures -> NaN
    df = clean_negative_expenditures(df)
       
     # Drop coverage flags that are almost entirely missing (not used as features for now)
    for col in ["PREVCOVR", "MORECOVR"]:
        if col in df.columns:
            # Could drop directly without checking; the extra check here is safer.
            if df[col].isna().mean() > 0.8:
                df = df.drop(columns=col)

    return df












