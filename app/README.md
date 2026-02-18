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
- Environment file: `app/environment_app.yml` (conda)

## Local run 
Create the MVP app environment (conda) and run:
```bash
conda env create -f app/environment_app.yml
conda activate meps_mvp
python -m streamlit run app/app.py
```

The app expects a feature‑engineered dataset with the schema defined in `app/model_loader.py` and can use `data/df_feat.parquet` in demo mode.
