"""Robustness checks quoted in the Smart Gatekeeper paper (Discussion).

Data: the 10% public subsample of the essential-oil WRRF dataset released with
Hassnain, Lee & Azhar (2025), https://doi.org/10.5281/zenodo.16784954
(download it yourself; it is not redistributed in this repository).

Usage:
    python robustness_check.py path/to/ActualData_Subsample.xlsx

What it does (Extra Trees, all 11 variables unless stated, target = final-effluent BOD):
  1. Validation comparison: shuffled 5-fold CV vs contiguous-block 5-fold CV vs the
     final 15% chronological holdout.
  2. One-at-a-time ablation on the chronological holdout.
  3. Online-only model: drop the laboratory variables F-COD, F-SS, F-TKN, F-NO3 together.
Writes robustness_results.csv next to the script and prints the numbers.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

warnings.filterwarnings("ignore")

TARGET = "F-BOD (mg/L)"
LAB_VARS = ["F-COD (mg/L)", "F-SS (mg/L)", "F-TKN (mg/L)", "F-NO3 (mg/L)"]
TEST_FRAC = 0.15
SEED = 0


def load(path):
    df = pd.read_excel(path)
    df.columns = [" ".join(str(c).split()) for c in df.columns]      # "F-SS  (mg/L)" -> "F-SS (mg/L)"
    return df.sort_values("Timestamp", kind="stable").reset_index(drop=True)


def model():
    return ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)


def holdout_r2(df, features):
    n = len(df)
    s = int(n * (1 - TEST_FRAC))
    X, y = df[features].values, df[TARGET].values
    pred = model().fit(X[:s], y[:s]).predict(X[s:])
    return r2_score(y[s:], pred)


def main(path):
    df = load(path)
    all_vars = [c for c in df.columns if c not in ("Timestamp", TARGET)]
    online = [c for c in all_vars if c not in LAB_VARS]
    X, y = df[all_vars].values, df[TARGET].values
    rows = []

    shuffled = r2_score(y, cross_val_predict(model(), X, y, cv=KFold(5, shuffle=True, random_state=SEED)))
    blocks = r2_score(y, cross_val_predict(model(), X, y, cv=KFold(5)))
    chrono = holdout_r2(df, all_vars)
    rows += [("validation: shuffled 5-fold CV", shuffled),
             ("validation: contiguous-block 5-fold CV", blocks),
             ("validation: final 15% chronological holdout", chrono)]

    base = chrono
    for f in all_vars:
        rows.append((f"ablation: R2 loss when '{f}' removed",
                     base - holdout_r2(df, [c for c in all_vars if c != f])))

    rows.append((f"online-only inputs ({len(online)}): holdout R2", holdout_r2(df, online)))
    rows.append(("online-only inputs: names", "; ".join(online)))

    out = pd.DataFrame(rows, columns=["check", "value"])
    out.to_csv("robustness_results.csv", index=False)
    with pd.option_context("display.max_colwidth", 90, "display.width", 140):
        print(out.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
