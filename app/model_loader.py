# app/model_loader.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


def get_project_root() -> Path:
    """Find the project root by looking for data/df_feat.parquet."""
    cur = Path(__file__).resolve()
    for _ in range(8):
        cand = cur.parents[_]
        if (cand / "data" / "df_feat.parquet").exists():
            return cand
    # fallback: parent of app/
    return Path(__file__).resolve().parents[1]


def get_artifact_dir(root: Path | None = None) -> Path:
    root = root or get_project_root()
    return root / "results" / "model_artifacts"


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
    """Return proba for positive class using meta['feature_cols']."""
    cols = meta["feature_cols"]
    X = df[cols].copy()
    proba = pipe.predict_proba(X)[:, 1]
    return proba


def predict_regression(pre, booster: xgb.Booster, meta: dict, df: pd.DataFrame) -> np.ndarray:
    """Return predicted LOG_TOTEXPY2 using meta['feature_cols'] and meta['best_iteration']."""
    cols = meta["feature_cols"]
    X = df[cols].copy()

    Xp = pre.transform(X)
    d = xgb.DMatrix(Xp)

    best_iter = int(meta.get("best_iteration", -1))
    if best_iter >= 0:
        pred = booster.predict(d, iteration_range=(0, best_iter + 1))
    else:
        pred = booster.predict(d)

    return pred
