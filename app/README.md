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

## Local run (macOS)**
Create a virtual environment (choose one):
```bash
# venv
python3 -m venv .venv
source .venv/bin/activate
```

```bash
# conda (recommended on mac)
conda create -n meps_app python=3.12 -y
conda activate meps_app
```

Install dependencies:
```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

If you hit XGBoost/OpenMP errors on macOS:
```bash
brew install libomp
python -m pip install --force-reinstall --no-cache-dir xgboost
```

Or use conda to avoid OpenMP issues:
```bash
conda install -c conda-forge xgboost llvm-openmp -y
```

Run the app:
```bash
python -m streamlit run app/app.py
```

The app expects a feature‑engineered dataset with the schema defined in `app/model_loader.py` and can use `data/df_feat.parquet` in demo mode.
