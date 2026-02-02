
"""
Project-wide configuration for the MEPS Panel 27 thesis project.

This module centralizes:
- Project paths (raw data, processed data, outputs)
- Random seed(s) for reproducibility
- Target/label column names used across preprocessing, feature engineering,
  and modeling notebooks.

Keeping these constants in one place makes the pipeline easier to maintain and
reduces the risk of inconsistent column naming across files.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

# Project root directory (assumes this file lives under <root>/src/ and is imported from there)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw data path
RAW_DATA_PATH = PROJECT_ROOT / "data" / "h252.xlsx"

# Processed (analysis-ready) dataset path
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "meps_panel27_processed.parquet"

# Output directories
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

RANDOM_SEED = 42

# ---------------------------------------------------------------------
# target/label column names
# ---------------------------------------------------------------------

# Raw total expenditure (primary regression outcome)
REG_TARGET_TOTEXPY2_RAW = "TOTEXPY2"

# Year 1 baseline expenditure (important predictor)
REG_BASELINE_TOTEXPY1 = "TOTEXPY1"

# Regression target used for modeling: log(1 + TOTEXPY2)
REG_TARGET_TOTEXPY2_LOG = "LOG_TOTEXPY2"

# High-cost label (e.g., top 10% of TOTEXPY2)
CLASS_TARGET_HIGHCOST_Y2 = "HIGHCOST_Y2"

# Acute event labels (constructed from count variables in feature engineering)
CLASS_TARGET_ANY_ED_Y2 = "ANY_ED_Y2"  # 1 if any ED visit in Y2
CLASS_TARGET_ANY_IP_Y2 = "ANY_IP_Y2"  # 1 if any inpatient stay in Y2

# Reference count variable names
ED_COUNT_Y1 = "ERTOTY1"
ED_COUNT_Y2 = "ERTOTY2"
IP_COUNT_Y1 = "IPDISY1"
IP_COUNT_Y2 = "IPDISY2"

# MEPS longitudinal person weight (Panel 27)
WEIGHT_COL = "LONGWT"
