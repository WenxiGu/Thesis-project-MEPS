# Health Outcome Predictor (MVP)

A Streamlit MVP that scores individuals using pre-trained models and supports capacity-constrained **top-k** ranking for:
- **Classification (risk ranking)**: predicted probability for Year-2 outcomes
- **Regression (cost ranking)**: predicted Year-2 total expenditure in **USD** (inverse transform from log1p)

## Repository structure
- `app/app.py` — Streamlit UI (top-k selection + table + CSV export)
- `app/model_loader.py` — artifact loading + inference helpers (+ fixed feature schema used by the MVP)
- `app/sanity.py` — minimal Streamlit sanity check
- `data/df_feat.parquet` — demo feature-engineered dataset (feature-level input)
- `results/model_artifacts/` — trained artifacts (pipelines/boosters + metadata)

## Requirements
- Python 3.12 recommended
- Dependencies in `requirements.txt` (pip)

## Quick start (local)
From the repository root:

```bash
pip install -r requirements.txt
streamlit run app/app.py
