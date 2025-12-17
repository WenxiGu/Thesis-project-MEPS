


import numpy as np
import pandas as pd

from src.config import (
    REG_TARGET_TOTEXPY2_RAW,
    REG_TARGET_TOTEXPY2_LOG,
    REG_BASELINE_TOTEXPY1,
    CLASS_TARGET_HIGHCOST_Y2,
    CLASS_TARGET_ANY_ED_Y2,
    CLASS_TARGET_ANY_IP_Y2,
    ED_COUNT_Y1,
    ED_COUNT_Y2,
    IP_COUNT_Y1,
    IP_COUNT_Y2,
)

# ============================================================
# CORE_DEMO_VARS  -> demographic feature engineering
# ============================================================

def add_core_demo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demographics: AGE/AGE_GROUP, SEX, race/ethnicity, region, education.
    """
    df = df.copy()

    # --- Age ---
    age_col = "AGEY1X" if "AGEY1X" in df.columns else ("AGEY2X" if "AGEY2X" in df.columns else None)
    if age_col is not None:
        df["AGE"] = df[age_col]
        df["AGE_GROUP"] = pd.cut(
            df[age_col],
            bins=[0, 17, 44, 64, 120],
            labels=["0-17", "18-44", "45-64", "65+"],
            right=True,
        )

    # --- Sex (1=male, 2=female) -> female indicator ---
    if "SEX" in df.columns:
        df["SEX_BIN"] = (df["SEX"] == 2).astype(int)

    # --- Race/ethnicity (categorical labels for one-hot later) ---
    if "RACETHX" in df.columns:
        race_map = {
            1: "Hispanic",
            2: "NH White",
            3: "NH Black",
            4: "NH Asian",
            5: "NH Other/multiple",
        }
        df["RACE_ETH"] = df["RACETHX"].map(race_map)

    # --- Region (baseline year 1) ---
    if "REGIONY1" in df.columns:
        region_map = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
        df["REGIONY1_CAT"] = df["REGIONY1"].map(region_map)

    # --- Education (EDUCYR) ---
    if "EDUCYR" in df.columns:
        df["EDUCYR_CONT"] = df["EDUCYR"]

        def _edu_group(v):
            if pd.isna(v):
                return np.nan
            v = int(v)
            if v == 0:
                return "No school / K only"
            if 1 <= v <= 8:
                return "Elementary (1–8)"
            if 9 <= v <= 11:
                return "High school (9–11)"
            if v == 12:
                return "Grade 12 (HS grad)"
            if 13 <= v <= 15:
                return "Some college (1–3 yrs)"
            if v == 16:
                return "4 years college"
            if v >= 17:
                return "5+ years college"
            return np.nan

        df["EDU_GROUP"] = df["EDUCYR"].map(_edu_group)

    return df


# ============================================================
# CORE_SES_VARS  -> socio-economic feature engineering
# ============================================================

def add_core_ses_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    SES: family income (log), poverty category, family size.
    """
    df = df.copy()

    # log family income (clamp negative at 0)
    if "FAMINCY1" in df.columns:
        df["LOG_FAMINCY1"] = np.log1p(df["FAMINCY1"].clip(lower=0))

    # poverty category labels (keep numeric POVCATY1 too)
    if "POVCATY1" in df.columns:
        pov_map = {
            1: "Poor / negative",
            2: "Near poor",
            3: "Low income",
            4: "Middle income",
            5: "High income",
        }
        df["POVCATY1_CAT"] = df["POVCATY1"].map(pov_map)

    # family size
    if "FAMSZEY1" in df.columns:
        df["FAMSIZE_Y1"] = df["FAMSZEY1"]

        def _fam_grp(v):
            if pd.isna(v):
                return np.nan
            v = int(v)
            return "7+" if v >= 7 else str(v)

        df["FAMSIZE_Y1_GRP"] = df["FAMSZEY1"].map(_fam_grp)

    return df


# ============================================================
# CORE_INS_VARS  -> insurance feature engineering
# ============================================================

def add_core_ins_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Insurance: year-1 coverage flags + detailed type.
    """
    df = df.copy()

    # INSCOVY1: 1 any private, 2 public only, 3 uninsured
    if "INSCOVY1" in df.columns:
        inscov = df["INSCOVY1"]
        df["ANY_PRIVATE_Y1"] = (inscov == 1).astype(int)
        df["PUBLIC_ONLY_Y1"] = (inscov == 2).astype(int)
        df["UNINSURED_Y1"] = (inscov == 3).astype(int)

    # INSURCY1: detailed category -> label for one-hot later
    if "INSURCY1" in df.columns:

        def _map_insurc(v):
            if pd.isna(v):
                return np.nan
            v = int(v)
            if v == 1:
                return "<65 any private"
            if v == 2:
                return "<65 public only"
            if v == 3:
                return "<65 uninsured"
            if v == 4:
                return "65+ Medicare only"
            if v == 5:
                return "65+ Medicare + private"
            if v == 6:
                return "65+ Medicare + other public"
            if v == 7:
                return "65+ uninsured"
            if v == 8:
                return "65+ other coverage"
            return np.nan

        df["INS_TYPE_Y1"] = df["INSURCY1"].map(_map_insurc)

    return df


# ============================================================
# CORE_EMP_VARS  -> employment feature engineering
# ============================================================

def add_core_emp_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Employment:
    - EVRWRKY1 -> WORKED_Y1
    - UNEMPY1X -> ANY_UNEMP_COMP_Y1, LOG_UNEMP_COMP_Y1
    - EMPST1/3/5 -> *_EMPBIN + summary
    """
    df = df.copy()

    # ever worked in year 1 (1 yes, 2 no)
    if "EVRWRKY1" in df.columns:
        df["WORKED_Y1"] = (df["EVRWRKY1"] == 1).astype(int)

    # unemployment compensation income
    if "UNEMPY1X" in df.columns:
        s = df["UNEMPY1X"].fillna(0)
        df["ANY_UNEMP_COMP_Y1"] = (s > 0).astype(int)
        df["LOG_UNEMP_COMP_Y1"] = np.log1p(s)



  # EMPST1/2 -> employment status binary indicators + summary
    """
    Year-1 EMPST summary using rounds 1 and 2 (if available).

    Creates:
    - EMPST1_EMPBIN, EMPST2_EMPBIN: 1 employed/attached, 0 not employed, NaN unknown/NIU
    - EMP_INFO_R12: 1 if any of the two rounds has info, else 0
    - EMP_ATTACHED_ANY_R12: 1 if employed/attached in any available round,
                            0 if not employed in all available rounds,
                            NaN if no info at all
    """
    
    round_cols = [c for c in ["EMPST1", "EMPST2"] if c in df.columns]
    if not round_cols:
        return df

    def emp_to_bin(v):
        if pd.isna(v):
            return np.nan
        v = int(v)
        if v in [1, 2, 3]:
            return 1  # employed/attached
        if v == 4:
            return 0  # not employed
        return np.nan

    # 1) Convert each round to a binary variable
    empbin_cols = []
    for col in round_cols:
        out = f"{col}_EMPBIN"
        df[out] = df[col].map(emp_to_bin)
        empbin_cols.append(out)

    # 2) Flag whether we have *any* employment info in rounds 1–2
    df["EMP_INFO_R12"] = df[empbin_cols].notna().any(axis=1).astype(int)

    # 3) Only define EMP_ATTACHED_ANY_R12 when info exists; otherwise keep NaN
    df["EMP_ATTACHED_ANY_R12"] = np.where(
        df["EMP_INFO_R12"] == 0,
        np.nan,                               # no info at all (NIU/unknown)
        (df[empbin_cols] == 1).any(axis=1).astype(int)  # 1 if any round says employed/attached else 0
    )
    
    # 4) Model-friendly version: fill NaN with 0, but keep EMP_INFO_R12 so model can distinguish NIU vs real 0
    df["EMP_ATTACHED_ANY_R12_FILL0"] = df["EMP_ATTACHED_ANY_R12"].fillna(0).astype(int)

    return df




# ============================================================
# CORE_HEALTH_STATUS_VARS  -> self-reported health features
# ============================================================

def add_core_health_status_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Health status:
    - RTHLTH1, MNHLTH1 -> FAIR/POOR indicators
    """
    df = df.copy()

    if "RTHLTH1" in df.columns:
        df["RTHLTH1_FAIRPOOR"] = df["RTHLTH1"].isin([4, 5]).astype(int)

    if "MNHLTH1" in df.columns:
        df["MNHLTH1_FAIRPOOR"] = df["MNHLTH1"].isin([4, 5]).astype(int)

    return df


# ============================================================
# CORE_CHRONIC_VARS  -> chronic condition features
# ============================================================

CHRONIC_Y1_COLS = [
    "HIBPDXY1",
    "CHDDXY1",
    "STRKDXY1",
    "CHOLDXY1",
    "ASTHDXY1",
    "DIABDXY1_M18",
]

def add_core_chronic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chronic conditions:
    - create *_BIN (1 if condition present)
    - MULTIMORBIDITY_Y1 and MULTIMORBIDITY_GE2
    """
    df = df.copy()

    chronic_bin_cols = []
    for col in CHRONIC_Y1_COLS:
        if col in df.columns:
            bin_col = col + "_BIN"
            df[bin_col] = (df[col] == 1).astype(int)
            chronic_bin_cols.append(bin_col)

    if chronic_bin_cols:
        df["MULTIMORBIDITY_Y1"] = df[chronic_bin_cols].sum(axis=1)
        df["MULTIMORBIDITY_GE2"] = (df["MULTIMORBIDITY_Y1"] >= 2).astype(int)

    return df


# ============================================================
# CORE_USE_COST_VARS  -> baseline utilisation + targets
# ============================================================

def add_core_use_cost_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use & cost:
    - baseline: LOG_TOTEXPY1, ANY_ED_Y1, ANY_IP_Y1
    - targets: LOG_TOTEXPY2, HIGHCOST_Y2, ANY_ED_Y2, ANY_IP_Y2
    """
    df = df.copy()

    # baseline cost
    if REG_BASELINE_TOTEXPY1 in df.columns:
        df["LOG_TOTEXPY1"] = np.log1p(df[REG_BASELINE_TOTEXPY1])

    # baseline utilisation
    if ED_COUNT_Y1 in df.columns:
        df["ANY_ED_Y1"] = (df[ED_COUNT_Y1] > 0).astype(int)
    if IP_COUNT_Y1 in df.columns:
        df["ANY_IP_Y1"] = (df[IP_COUNT_Y1] > 0).astype(int)

    # regression target
    if REG_TARGET_TOTEXPY2_RAW in df.columns:
        df[REG_TARGET_TOTEXPY2_LOG] = np.log1p(df[REG_TARGET_TOTEXPY2_RAW])

        q90 = df[REG_TARGET_TOTEXPY2_RAW].quantile(0.90)
        df[CLASS_TARGET_HIGHCOST_Y2] = (df[REG_TARGET_TOTEXPY2_RAW] >= q90).astype(int)

    # classification targets
    if ED_COUNT_Y2 in df.columns:
        df[CLASS_TARGET_ANY_ED_Y2] = (df[ED_COUNT_Y2] > 0).astype(int)
    if IP_COUNT_Y2 in df.columns:
        df[CLASS_TARGET_ANY_IP_Y2] = (df[IP_COUNT_Y2] > 0).astype(int)

    return df


# ============================================================
# MASTER PIPELINE (mirrors the CORE_* grouping order)
# ============================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering pipeline organised by CORE_* variable groups.
    Assumes df already passed through preprocess_meps().
    """
    df_feat = df.copy()

    df_feat = add_core_demo_features(df_feat)
    df_feat = add_core_ses_features(df_feat)
    df_feat = add_core_ins_features(df_feat)
    df_feat = add_core_emp_features(df_feat)
    df_feat = add_core_health_status_features(df_feat)
    df_feat = add_core_chronic_features(df_feat)
    df_feat = add_core_use_cost_features(df_feat)

    return df_feat

