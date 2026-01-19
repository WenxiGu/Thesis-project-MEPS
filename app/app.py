# app/app.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import sys, time
print("BOOT:", time.ctime(), flush=True)
sys.stdout.flush()

from model_loader import (
    FEATURES,
    get_project_root,
    get_artifact_dir,
    load_classification_artifact,
    load_regression_booster_artifact,
    predict_classification,
    predict_regression,
)

# -----------------------
# Config: artifact names
# -----------------------
CLF_MAP = {
    "HIGHCOST_Y2 (RF)": "clf_highcost_rf",
    "ANY_ED_Y2 (XGB)": "clf_ed_xgb",
    "ANY_IP_Y2 (RF)": "clf_ip_rf",
}
REG_NAME = "reg_log_totexpy2_xgb_es"


# -----------------------
# Helpers
# -----------------------
def read_uploaded(uploaded) -> pd.DataFrame:
    if uploaded.name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if uploaded.name.endswith(".parquet"):
        return pd.read_parquet(uploaded)
    raise ValueError("Unsupported file type. Please upload CSV or Parquet.")


def topk_select(df: pd.DataFrame, score_col: str, top_frac: float) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    n = len(df)
    k = int(round(top_frac * n))
    df = df.sort_values(score_col, ascending=False)
    df["selected_topk"] = 0
    if k > 0:
        df.iloc[:k, df.columns.get_loc("selected_topk")] = 1
    return df, k


def add_reason_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight rule-based tags for individual-level interpretability.
    No SHAP needed; suitable for MVP/demo.
    """
    df = df.copy()

    q75 = None
    if "LOG_TOTEXPY1" in df.columns and df["LOG_TOTEXPY1"].notna().any():
        q75 = df["LOG_TOTEXPY1"].quantile(0.75)

    tags = []
    for _, r in df.iterrows():
        t = []
        if q75 is not None and r.get("LOG_TOTEXPY1", -np.inf) >= q75:
            t.append("High prior-year spending")
        if r.get("ANY_ED_Y1", 0) == 1:
            t.append("Prior ED use")
        if r.get("ANY_IP_Y1", 0) == 1:
            t.append("Prior inpatient")
        if r.get("AGE", 0) >= 65:
            t.append("Older age (65+)")
        if r.get("RTHLTH1_FAIRPOOR", 0) == 1:
            t.append("Poor self-rated health")
        if r.get("HIBPDXY1_BIN", 0) == 1:
            t.append("Hypertension")

        tags.append(", ".join(t[:3]))
    df["reasons"] = tags
    return df


# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Health Outcome Predictor (MVP)", layout="wide")
st.title("Health Outcome Predictor (MVP)")

ROOT = get_project_root()
ART_DIR = get_artifact_dir(ROOT)


with st.sidebar.expander("Debug info"):
    st.write(f"Project root: {ROOT}")
    st.write(f"Artifacts: {ART_DIR}")

# Sidebar
st.sidebar.header("Data input")
use_demo = st.sidebar.checkbox("Use demo df_feat.parquet", value=False)
uploaded = st.sidebar.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])

if use_demo:
    demo_path = ROOT / "data" / "df_feat.parquet"
    if not demo_path.exists():
        st.error(f"Demo file not found: {demo_path}")
        st.stop()
    df = pd.read_parquet(demo_path)
    st.sidebar.success(f"Loaded demo: {demo_path.name} ({len(df):,} rows)")
else:
    if uploaded is None:
        st.info("Upload a dataset or enable demo mode.")
        st.stop()
    df = read_uploaded(uploaded)
    st.sidebar.success(f"Loaded: {uploaded.name} ({len(df):,} rows)")

st.sidebar.header("Mode")
mode = st.sidebar.radio("Task", ["Classification (risk ranking)", "Regression (cost ranking)"])
top_frac = st.sidebar.selectbox("Top-k fraction", [0.05, 0.10, 0.20], index=1)
max_rows = st.sidebar.slider("Rows to display", 50, 2000, 200, 50)

# Display columns
default_show = [c for c in ["AGE", "LOG_TOTEXPY1", "ANY_ED_Y1", "ANY_IP_Y1"] if c in df.columns]
show_cols = st.sidebar.multiselect("Extra columns to display", df.columns.tolist(), default=default_show)

with st.sidebar.expander("Features used (hard-coded)"):
    st.write(FEATURES)

# -----------------------
# Main
# -----------------------
if mode.startswith("Classification"):
    target_label = st.sidebar.selectbox("Target model", list(CLF_MAP.keys()))
    artifact_name = CLF_MAP[target_label]

    pipe, meta = load_classification_artifact(ART_DIR, artifact_name)

    try:
        proba = predict_classification(pipe, meta, df)
    except KeyError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()

    out = df.copy()
    out["score"] = proba
    out, k = topk_select(out, "score", top_frac)
    out = add_reason_tags(out)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows scored", f"{len(out):,}")
    c2.metric(f"Top {int(top_frac*100)}% selected", f"{k:,}")
    c3.metric("Mean risk score", f"{out['score'].mean():.3f}")

    st.subheader("Ranked list")
    cols_to_show = ["score", "selected_topk", "reasons"] + show_cols
    cols_to_show = [c for c in cols_to_show if c in out.columns]
    st.dataframe(out[cols_to_show].head(max_rows), use_container_width=True)

    st.download_button(
        "Download scored CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name=f"scored_{artifact_name}.csv",
        mime="text/csv",
    )

else:
    pre, booster, meta = load_regression_booster_artifact(ART_DIR, REG_NAME)

    try:
        pred = predict_regression(pre, booster, meta, df)
    except KeyError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()

    

    out = df.copy()
    out["pred_log_cost"] = pred
    out["pred_cost_usd"] = np.expm1(out["pred_log_cost"]).clip(lower=0)

    # rank by USD cost (equivalent monotonic transform, but clearer)
    out, k = topk_select(out, "pred_cost_usd", top_frac)

    mean_usd = out["pred_cost_usd"].mean()
    med_usd = out["pred_cost_usd"].median()
    p90_usd = float(out["pred_cost_usd"].quantile(0.90))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows scored", f"{len(out):,}")
    c2.metric(f"Top {int(top_frac*100)}% selected", f"{k:,}")
    c3.metric("Mean predicted cost (USD)", f"{mean_usd:,.0f}")
    c4.metric("Median predicted cost (USD)", f"{med_usd:,.0f}")

    st.caption(f"Top {int(top_frac*100)}% threshold ≈ ${p90_usd:,.0f} predicted Year-2 total expenditure.")

    st.subheader("Ranked list (highest predicted cost first)")
    base_cols = ["pred_cost_usd", "selected_topk"]
    # keep log cost optional (nice for debugging)
    optional_cols = ["pred_log_cost"]
    cols_to_show = base_cols + optional_cols + show_cols
    cols_to_show = [c for c in cols_to_show if c in out.columns]

    st.dataframe(
        out.sort_values("pred_cost_usd", ascending=False)[cols_to_show].head(max_rows),
        use_container_width=True,
    )

    st.download_button(
        "Download scored CSV",
        data=out.sort_values("pred_cost_usd", ascending=False).to_csv(index=False).encode("utf-8"),
        file_name="scored_reg_totexpy2_usd.csv",
        mime="text/csv",
    )
