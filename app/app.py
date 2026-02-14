"""
Streamlit application for MEPS Panel 27 risk and cost prediction.

This app loads the saved *preferred* trained model artifacts and exposes an interactive UI to generate:

- Regression: predicted Year-2 total expenditure on the log scale (`LOG_TOTEXPY2`)
  with an optional back-transform for interpretability.
- Classification: predicted probabilities for Year-2 outcomes:
  `HIGHCOST_Y2`, `ANY_ED_Y2`, and `ANY_IP_Y2`.

Key components
--------------
- Feature input form: collects the engineered predictors defined in
  `model_loader.FEATURES` (numeric + categorical).
- Artifact management: delegates loading and prediction logic to `model_loader.py`,
  including preprocessing, sklearn pipelines, and the XGBoost Booster for regression.
- Reproducibility: prints a boot timestamp to logs for deployment debugging.

How to run (local)
------------------
From the project root (or the `app/` directory), run:
    streamlit run app/app.py

Notes
-----
- The app assumes the artifact directory exists under `results/` and contains the
  saved pipelines/models and their metadata files.
- This file intentionally keeps UI logic separate from model I/O and prediction
  logic (see `model_loader.py`).
"""




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
    df["reason_tags"] = tags

    return df


def get_required_cols(meta: dict | None, fallback: list[str]) -> list[str]:
    return (meta or {}).get("feature_cols") or fallback


def get_missing_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c not in df.columns]


# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Health Outcome Predictor (MVP)", layout="wide")
st.title("Health Outcome Predictor (MVP)")
st.caption(
    "Demo uses a feature-engineered dataset (df_feat.parquet). "
    "Uploaded data should follow the same feature schema used for model training."
)


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
    st.markdown(
        "**How to read this**\n\n"
        "Score = predicted probability of the Year‑2 target (higher = higher risk).\n\n"
        "Selected_topk = 1 if the row is in the top‑k fraction.\n\n"
        "Top‑k threshold = lowest score among selected rows.\n\n"
        "Lift = selected mean risk / overall mean risk.\n\n"
        "Reason tags are simple rule‑based hints (not causal explanations).\n"
    )

if mode.startswith("Classification"):
    target_label = st.sidebar.selectbox("Target model", list(CLF_MAP.keys()))
    artifact_name = CLF_MAP[target_label]

    pipe, meta = load_classification_artifact(ART_DIR, artifact_name)

    req_cols = get_required_cols(meta, FEATURES)
    missing = get_missing_cols(df, req_cols)
    if missing:
        st.error(f"Missing required feature columns: {missing}")
        st.stop()

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
    if "LOG_TOTEXPY1" in out.columns:
        out["TOTEXPY1_USD"] = np.expm1(out["LOG_TOTEXPY1"]).clip(lower=0)

    overall_mean = float(out["score"].mean())
    selected_mean = float(out.loc[out["selected_topk"] == 1, "score"].mean())
    lift = (selected_mean / overall_mean) if overall_mean > 0 else float("nan")

    l1, l2, l3 = st.columns(3)
    l1.metric("Overall mean risk", f"{overall_mean:.3f}")
    l2.metric("Selected mean risk", f"{selected_mean:.3f}")
    l3.metric("Lift (selected / overall)", f"{lift:.2f}×")


    c1, c2, c3 = st.columns(3)
    c1.metric("Rows scored", f"{len(out):,}")
    c2.metric(f"Top {int(top_frac*100)}% selected", f"{k:,}")
    thr = float(out.loc[out["selected_topk"] == 1, "score"].min()) if k > 0 else None
    c3.metric("Top-k score threshold", f"{thr:.3f}" if thr is not None else "N/A")


    st.subheader("Ranked list")
    display_cols = []
    for c in show_cols:
        if c == "LOG_TOTEXPY1" and "TOTEXPY1_USD" in out.columns:
            display_cols.append("TOTEXPY1_USD")
        else:
            display_cols.append(c)
    cols_to_show = ["score", "selected_topk", "reason_tags"] + display_cols
    cols_to_show = [c for c in cols_to_show if c in out.columns]
    st.dataframe(
        out[cols_to_show].head(max_rows),
        use_container_width=True,
        column_config={
            "TOTEXPY1_USD": st.column_config.NumberColumn(
                "TOTEXPY1_USD (USD)",
                format="$%,.0f",
            ),
        },
    )

    st.download_button(
        "Download scored CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name=f"scored_{artifact_name}.csv",
        mime="text/csv",
    )

else:
    pre, booster, meta = load_regression_booster_artifact(ART_DIR, REG_NAME)

    req_cols = get_required_cols(meta, FEATURES)
    missing = get_missing_cols(df, req_cols)
    if missing:
        st.error(f"Missing required feature columns: {missing}")
        st.stop()

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



    # rank by USD cost (equivalent transform, but clearer)
    out, k = topk_select(out, "pred_cost_usd", top_frac)

    overall_mean_cost = float(out["pred_cost_usd"].mean())
    selected_mean_cost = float(out.loc[out["selected_topk"] == 1, "pred_cost_usd"].mean())
    cost_lift = (selected_mean_cost / overall_mean_cost) if overall_mean_cost > 0 else float("nan")

    r1, r2, r3 = st.columns(3)
    r1.metric("Overall mean cost (USD)", f"{overall_mean_cost:,.0f}")
    r2.metric("Selected mean cost (USD)", f"{selected_mean_cost:,.0f}")
    r3.metric("Lift (selected / overall)", f"{cost_lift:.2f}×")


    mean_usd = out["pred_cost_usd"].mean()
    med_usd = out["pred_cost_usd"].median()
    p90_usd = float(out["pred_cost_usd"].quantile(0.90))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows scored", f"{len(out):,}")
    c2.metric(f"Top {int(top_frac*100)}% selected", f"{k:,}")
    thr = float(out.loc[out["selected_topk"] == 1, "pred_cost_usd"].min()) if k > 0 else None
    c3.metric("Top-k cost threshold (USD)", f"{thr:,.0f}" if thr is not None else "N/A")
    c4.metric("Median predicted cost (USD)", f"{med_usd:,.0f}")


    # formatted display column (currency with 3 decimals)
    out["pred_cost_usd_fmt"] = out["pred_cost_usd"].map(lambda x: f"${x:,.3f}")

    st.subheader("Ranked list (highest predicted cost first)")
    base_cols = ["pred_cost_usd_fmt", "selected_topk"]
    # keep log cost optional 
    optional_cols = ["pred_log_cost"]
    cols_to_show = base_cols + optional_cols + show_cols
    cols_to_show = [c for c in cols_to_show if c in out.columns]

    
    st.dataframe(
        out.sort_values("pred_cost_usd", ascending=False)[cols_to_show].head(max_rows),
        use_container_width=True,
        column_config={
            "pred_cost_usd_fmt": "pred_cost_usd (USD)",
            "pred_log_cost": st.column_config.NumberColumn(
                "pred_log_cost",
                format="%.4f",
            ),
        },
    )


    
     
    st.download_button(
    "Download scored CSV",
    data=out.sort_values("pred_cost_usd", ascending=False)
          .to_csv(index=False).encode("utf-8"),
    file_name="scored_reg_totexpy2_usd.csv",
    mime="text/csv",
)
