
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

SPECIAL_NA = {-1: np.nan, -7: np.nan, -8: np.nan, -9: np.nan}

def weighted_mae(y_true, y_pred, w):
    return (np.abs(y_true - y_pred) * w).sum() / w.sum()

def load_data(path="/mnt/data/h252.xlsx", sheet="H252"):
    df = pd.read_excel(path, sheet_name=sheet)
    df = df.replace(SPECIAL_NA)
    df = df[(df.get("YEARIND",1)==1) & (df.get("ALL5RDS",1)==1)].copy()
    return df

def demo():
    df = load_data()
    y = df["TOTEXPY2"].astype(float)
    W = df.get("LONGWT", pd.Series(np.ones(len(df)))).astype(float)
    num_cols = [c for c in ["AGEY1X","TOTEXPY1","IPDISY1","ERTOTY1"] if c in df.columns]
    cat_cols = [c for c in ["SEX","INSCY1"] if c in df.columns]
    X = df[num_cols + cat_cols].copy()

    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    mdl = Pipeline([("pre", pre), ("mdl", XGBRegressor(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0
    ))])

    Xtr, Xte, ytr, yte, wtr, wte = train_test_split(X, y, W, test_size=0.2, random_state=42)
    mdl.fit(Xtr, ytr, mdl__sample_weight=wtr)
    pred = mdl.predict(Xte)
    print("Weighted MAE:", weighted_mae(yte, pred, wte))

if __name__ == "__main__":
    demo()
