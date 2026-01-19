# app/model_loader.py
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

# -----------------------
# Hard-coded feature list
# -----------------------
CAT_COLS = [
    "RACE_ETH",
    "REGIONY1_CAT",
    "EDU_GROUP",
    "POVCATY1_CAT",
    "FAMSIZE_Y1_GRP",
    "INS_TYPE_Y1",
]

NUM_COLS = [
    # demographics / SES
    "AGE",
    "SEX_BIN",
    "LOG_FAMINCY1",
    "FAMSIZE_Y1",

    # employment
    "WORKED_Y1",
    "ANY_UNEMP_COMP_Y1",
    "LOG_UNEMP_COMP_Y1",
    "EMP_INFO_R12",
    "EMP_ATTACHED_ANY_R12_FILL0",  # model-friendly version

    # health status baseline
    "RTHLTH1_FAIRPOOR",
    "MNHLTH1_FAIRPOOR",

    # chronic conditions baseline
    "HIBPDXY1_BIN",
    "CHDDXY1_BIN",
    "STRKDXY1_BIN",
    "CHOLDXY1_BIN",
    "ASTHDXY1_BIN",
    "DIABDXY1_M18_BIN",

    # baseline utilisation/cost
    "LOG_TOTEXPY1",
    "ANY_ED_Y1",
    "ANY_IP_Y1",
]

FEATURES = CAT_COLS + NUM_COLS


def get_project_root() -> Path:
    """Find project root by looking for data/df_feat.parquet (up to 8 levels)."""
    cur = Path(__file__).resolve()
    for i, cand in enumerate(cur.parents):
        if i >= 8:
            break
        if (cand / "data" / "df_feat.parquet").exists():
            return cand
    # fallback: parent of app/
    return cur.parents[1]


def get_artifact_dir(root: Path | None = None) -> Path:
    root = root or get_project_root()
    return root / "results" / "model_artifacts"


def ensure_features(df: pd.DataFrame, cols: list[str]) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"Missing required feature columns: {miss}")


def load_classification_artifact(art_dir: Path, name: str):
    """
    name examples: 'clf_highcost_rf', 'clf_ed_xgb', 'clf_ip_rf'
    returns: (pipeline, meta_dict)
    """
    pipe = joblib.load(art_dir / f"{name}.joblib")
    meta = json.loads((art_dir / f"{name}.meta.json").read_text())
    return pipe, meta


def load_regression_booster_artifact(art_dir: Path, name: str):
    """
    name example: 'reg_log_totexpy2_xgb_es'
    returns: (preprocess, booster, meta_dict)
    """
    pre = joblib.load(art_dir / f"{name}.preprocess.joblib")

    booster = xgb.Booster()
    booster.load_model(art_dir / f"{name}.booster.json")

    meta = json.loads((art_dir / f"{name}.meta.json").read_text())
    return pre, booster, meta


def predict_classification(pipe, meta: dict, df: pd.DataFrame) -> np.ndarray:
    """
    Return proba for positive class.
    If meta has no feature_cols, fall back to hard-coded FEATURES.
    """
    cols = (meta or {}).get("feature_cols") or FEATURES
    ensure_features(df, cols)
    X = df[cols].copy()
    return pipe.predict_proba(X)[:, 1]


def predict_regression(pre, booster: xgb.Booster, meta: dict, df: pd.DataFrame) -> np.ndarray:
    """
    Return predicted LOG_TOTEXPY2 (log-cost).
    If meta has no feature_cols, fall back to hard-coded FEATURES.
    """
    cols = (meta or {}).get("feature_cols") or FEATURES
    ensure_features(df, cols)
    X = df[cols].copy()

    Xp = pre.transform(X)
    d = xgb.DMatrix(Xp)

    best_iter = int((meta or {}).get("best_iteration", -1))
    if best_iter >= 0:
        return booster.predict(d, iteration_range=(0, best_iter + 1))
    return booster.predict(d)
