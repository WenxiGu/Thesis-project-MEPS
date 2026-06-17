# MEPS Panel 27 Risk & Cost Prediction (Thesis Project)

This repository contains the code, notebooks, model artifacts, aggregate outputs, and a
Streamlit MVP for a master's thesis project that predicts Year-2 health care costs and
acute utilization from Year-1 features using the Medical Expenditure Panel Survey
Household Component (MEPS-HC), Panel 27 Longitudinal Public Use File HC-252
(2022-2023).

The primary use case is risk stratification via top-k ranking rather than deterministic
individual clinical prediction.

## What This Repository Includes

- End-to-end preprocessing, feature engineering, and modeling utilities in `src/`
- Notebooks for EDA, feature engineering, modeling, and evaluation in `notebooks/`
- Aggregate figures, summary tables, and trained model artifacts in `results/`
- A Streamlit MVP for top-k selection in `app/`
- Reference documentation in `docs/`

## Data Source and Download

This project uses the official MEPS-HC Panel 27 Longitudinal Public Use File:

- Dataset: **HC-252: Panel 27 Longitudinal Data File**
- Years linked: **2022 and 2023**
- Publisher: **Agency for Healthcare Research and Quality (AHRQ), Medical Expenditure Panel Survey (MEPS)**
- Official dataset page: <https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-252>
- Official documentation: <https://meps.ahrq.gov/data_stats/download_data/pufs/h252/h252doc.shtml>
- Codebook / variable definitions: <https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_codebook.jsp?PUFId=H252&sortBy=Start>

The HC-252 public-use file includes 8,292 individuals and 2,648 variables. For this
project, the Excel version of the file was used as the raw input.

## Data Use and Redistribution

This repository does **not** redistribute the raw MEPS public-use microdata or
row-level derived datasets. To reproduce the analysis, download the official HC-252
file directly from AHRQ/MEPS and place it locally under `data/`.

Recommended local file layout:

```text
data/
  h252.xlsx          # official MEPS HC-252 Excel file downloaded from AHRQ
  df_pre.parquet    # locally generated cleaned/intermediate dataset
  df_feat.parquet   # locally generated feature-engineered dataset
```

These files are ignored by git. See `data/README.md` for details.

By using MEPS data, users are responsible for complying with the MEPS/AHRQ Data Use
Agreement. In particular, the data should be used only for statistical reporting and
analysis; users must not attempt to identify individuals or establishments; and MEPS
should be cited as the data source in any research outputs.

## Prediction Targets

- Regression: `LOG_TOTEXPY2 = log1p(TOTEXPY2)`
- Classification:
  - `HIGHCOST_Y2`: high-cost status in Year 2
  - `ANY_ED_Y2`: any emergency department visit in Year 2
  - `ANY_IP_Y2`: any inpatient stay in Year 2

## Modeling Approach

- Preprocessing: median imputation for numeric features, most-frequent imputation for
  categorical features, and one-hot encoding
- Models: ElasticNet, logistic regression, Random Forest, XGBoost, and MLP
- Evaluation: ROC-AUC, PR-AUC, F1 at validation-selected thresholds, and top-k risk
  ranking views for capacity-constrained use cases

## Key Results (Test Set)

- `HIGHCOST_Y2`: AUC 0.868, PR-AUC 0.459, F1@t* 0.536 (Random Forest)
- `ANY_ED_Y2`: AUC 0.728, PR-AUC 0.350, F1@t* 0.366 (XGBoost)
- `ANY_IP_Y2`: AUC 0.755, PR-AUC 0.225, F1@t* 0.285 (Tuned Random Forest)
- `LOG_TOTEXPY2`: RMSE_log 2.15, R2 0.53 (XGBoost with early stopping)

## Repository Structure

```text
app/        Streamlit MVP for top-k selection and CSV export
data/       Local data directory; data files are not tracked
docs/       Thesis and MEPS reference documents
notebooks/  EDA, data preparation, modeling, and evaluation workflow
results/    Aggregate figures, summary tables, and trained model artifacts
src/        Reusable preprocessing, feature engineering, and modeling code
```

## Setup

Python 3.12 is recommended.

Notebook environment:

```bash
conda env create -f notebooks/environment_thesis.yml
conda activate meps
```

## Reproduce the Analysis

1. Download the official HC-252 Excel file from AHRQ/MEPS:
   <https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-252>
2. Save the Excel file locally as:

```text
data/h252.xlsx
```

3. Run the notebooks in order:

```text
notebooks/01_EDA.ipynb
notebooks/02_Data Preparation & Feature Engineering.ipynb
notebooks/03_modeling_baseline.ipynb
notebooks/04_Model Predictions.ipynb
notebooks/05_weighted evaluation.ipynb
```

4. Train and export preferred models:

```bash
python src/train_and_save_preferred_models.py
```

## Run the MVP App

Create the app environment and launch Streamlit:

```bash
conda env create -f app/environment_app.yml
conda activate meps_mvp
python -m streamlit run app/app.py
```

The app expects a feature-engineered dataset with the schema defined in
`app/model_loader.py`. For local demo mode, generate `data/df_feat.parquet` by running
the preprocessing and feature engineering workflow first.

## Notes on Use

- The models are intended for risk ranking and top-k prioritization, not diagnosis or
  deterministic clinical prediction.
- The MVP includes simple rule-based reason tags for readability and plausibility
  checks. They are not causal explanations or model-derived local attributions.
- Outputs should be interpreted as statistical decision-support artifacts and reviewed
  with appropriate human oversight.

## Citation

Data source: Agency for Healthcare Research and Quality (AHRQ), Medical Expenditure
Panel Survey Household Component (MEPS-HC), Panel 27 Longitudinal Public Use File
HC-252.
