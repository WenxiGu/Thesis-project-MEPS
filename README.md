# MEPS Panel 27 Risk & Cost Prediction (Thesis Project)

This repository contains the code, data artifacts, and MVP app for a thesis that predicts Year-2 health care costs and acute events from Year-1 features using the MEPS Household Component (Panel 27, HC-252; 2022–2023). The primary use case is **risk stratification via top‑k ranking** rather than deterministic individual prediction.

**What this repo includes**
- End‑to‑end preprocessing, feature engineering, and modeling utilities in `src/`.
- Notebooks that document EDA, feature engineering, modeling, and evaluation in `notebooks/`.
- Pretrained model artifacts and summary tables in `results/`.
- A Streamlit MVP (“Health Outcome Predictor”) for top‑k selection in `app/`.
- Reference documents in `docs/`.

**Data**
- Source: MEPS‑HC Panel 27 longitudinal file **HC‑252** (links 2022 + 2023 full‑year files). The public‑use file includes **8,292 individuals** and **2,648 variables** . Raw input is stored as `data/h252.xlsx` (also zipped as `data/h252xlsx.zip`).
- Core modeling dataset: `data/df_pre.parquet` (cleaned core variables), `data/df_feat.parquet` (feature‑engineered, model‑ready).
- Survey weights: LONGWT and design variables (VARSTR, VARPSU) are retained in the core dataset (see `src/config.py`).
- Reference docs: `docs/thesis .docx`, `docs/MEPS HC - 252 intro.pdf`.

**Prediction targets**
- Regression: `LOG_TOTEXPY2 = log1p(TOTEXPY2)`. (LOG_TOTEXPY2 is the natural log of (1 + Year‑2 total expenditures), which stabilizes the heavy‑tailed cost distribution; predictions can be converted back to dollars with TOTEXPY2 = exp(LOG_TOTEXPY2) - 1.)
- Classification: `HIGHCOST_Y2` (top‑cost indicator), `ANY_ED_Y2` (any ED visit), `ANY_IP_Y2` (any inpatient stay).

**Modeling approach (summary)**
- Preprocessing: median imputation for numeric, most‑frequent for categorical, one‑hot encoding.
- Baselines and helpers in `src/models.py`.
- Evaluation emphasizes ranking metrics (ROC AUC, PR‑AUC) plus F1 at a validation‑selected threshold.

**Key results (test set)**
- HIGHCOST_Y2: AUC 0.868, PR‑AUC 0.459, F1@t* 0.536 (Random Forest).
- ANY_ED_Y2: AUC 0.728, PR‑AUC 0.350, F1@t* 0.366 (XGBoost).
- ANY_IP_Y2: AUC 0.755, PR‑AUC 0.225, F1@t* 0.285 (Tuned Random Forest).
- LOG_TOTEXPY2: RMSE_log 2.15, R² 0.53 (XGBoost w/ early stopping).

**Repository structure**
- `app/` Streamlit MVP for top‑k selection and CSV export (see `app/README.md`).
- `data/` raw and processed datasets.
- `notebooks/` EDA and modeling workflow.
- `results/` figures, tables, and model artifacts used by the app.
- `src/` reusable preprocessing, feature engineering, and modeling code.

**Setup**
Python 3.12 is recommended.


**Reproduce the analysis**
Notebooks (run in order):
- `notebooks/01_EDA.ipynb`
- `notebooks/02_Data Preparation & Feature Engineering.ipynb`
- `notebooks/03_Modeling_baseline.ipynb`
- `notebooks/04_Model Predictions.ipynb`
- `notebooks/05_weighted evaluation.ipynb`

Train and export preferred models:
```bash
python src/train_and_save_preferred_models.py
```


**Run the MVP app**
```bash
streamlit run app/app.py
```

The app expects a feature‑engineered dataset with the schema defined in `app/model_loader.py` and can use `data/df_feat.parquet` in demo mode.

**Notes on use**
- The models are intended for ranking and top‑k selection, not deterministic clinical prediction.
- The MVP includes simple rule‑based “reason tags” for interpretability and plausibility checks; these are not causal attributions.

**Citations (brief)**
- MEPS Household Component (HC‑252, Panel 27), AHRQ/NCHS.
- MEPS‑HC design & methods report (Cohen, 1997), as cited in the thesis.
- Thesis document in `docs/thesis .docx` and HC‑252 intro in `docs/MEPS HC - 252 intro.pdf`.
