"""Permutation importance for COD and BOD using ALL measured variables as candidate drivers
(pH, temperature, DO, flows, SRT, suspended solids, TKN, nitrate; the other target is excluded).
Random Forest COD and BOD (direct) models.

Reproduces the models behind Table 1 of the paper (COD R2 = 0.684, BOD direct R2 = 0.903)
and draws Figure: permutation importance on the final 15% chronological test set.

Data: 10% public subsample released with Hassnain, Lee & Azhar (2025),
https://doi.org/10.5281/zenodo.16784954 (not redistributed here).

Usage:
    python feature_importance_figure.py path/to/ActualData_Subsample.xlsx
Outputs: feature_importance.png / .pdf / feature_importance.csv
Note: importance of correlated inputs (the flows, in particular) can be understated,
because shuffling one input leaves a correlated one intact.
"""
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

warnings.filterwarnings("ignore")

SEED, TEST_FRAC, N_REPEATS = 42, 0.15, 30
FEATURES = {"F-PH": "pH", "Temperature (C)": "Temperature", "DO (mg/L)": "Dissolved oxygen",
            "Flow (L/s)": "Inlet flow", "Flow-Recycle (L/s)": "Recycle-sludge flow",
            "Flow-Aeration (L/s)": "Aeration flow", "SRT (days)": "Sludge retention time",
            "F-SS (mg/L)": "Suspended solids (SS)", "F-TKN (mg/L)": "Total Kjeldahl N", "F-NO3 (mg/L)": "Nitrate"}
TARGETS = {"COD": "F-COD (mg/L)", "BOD": "F-BOD (mg/L)"}


def fit_rf(X, y):
    grid = {"max_depth": [8, None], "min_samples_leaf": [1, 5]}
    est = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1)
    gs = GridSearchCV(est, grid, cv=TimeSeriesSplit(5), scoring="neg_root_mean_squared_error", n_jobs=-1)
    return gs.fit(X, y).best_estimator_


def main(path):
    df = pd.read_excel(path)
    df.columns = [" ".join(str(c).split()) for c in df.columns]
    df = df.sort_values("Timestamp", kind="stable").reset_index(drop=True)
    X = df[list(FEATURES)].values
    s = int(len(df) * (1 - TEST_FRAC))

    results, panels = [], {}
    for name, col in TARGETS.items():
        y = df[col].values
        rf = fit_rf(X[:s], y[:s])
        r2 = r2_score(y[s:], rf.predict(X[s:]))
        pi = permutation_importance(rf, X[s:], y[s:], n_repeats=N_REPEATS, random_state=SEED, scoring="r2")
        from scipy.stats import spearmanr
        rho = [spearmanr(df[c], df[col])[0] for c in FEATURES]
        imp = pd.DataFrame({"input": list(FEATURES.values()), "importance": pi.importances_mean,
                            "sd": pi.importances_std, "spearman_rho": rho}).sort_values("importance")
        panels[name] = (imp, r2)
        results.append(imp.assign(target=name, test_R2=round(r2, 3)))
        print(f"{name}: test R2 = {r2:.3f}")
        print(imp.sort_values("importance", ascending=False).round(4).to_string(index=False), "\n")
    pd.concat(results).to_csv("feature_importance.csv", index=False)

    ink, grey, blue = "#222222", "#8a8a8a", "#1f5fae"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": grey, "axes.labelcolor": ink,
                         "xtick.color": ink, "ytick.color": ink, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), dpi=300)
    for ax, (name, (imp, r2)), tag in zip(axes, panels.items(), "ab"):
        ax.barh(imp["input"], imp["importance"], xerr=imp["sd"], color=blue, height=0.6,
                error_kw={"ecolor": ink, "elinewidth": 0.8, "capsize": 2})
        for i, (v, e) in enumerate(zip(imp["importance"], imp["sd"])):
            ax.text(max(v, 0) + e + 0.01 * max(imp["importance"].max(), 0.05), i, f"{v:.2f}", va="center", fontsize=8, color=ink)
        ax.set_title(f"({tag}) {name} (RF, test R² = {r2:.3f})", fontsize=8.5, loc="left", color=ink)
        ax.set_xlabel("R² drop when shuffled")
        ax.set_xlim(left=min(0, (imp["importance"] - imp["sd"]).min() * 1.1), right=imp["importance"].max() * 1.45)
        ax.grid(axis="x", color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=2.0)
    fig.savefig("feature_importance.png", dpi=300)
    fig.savefig("feature_importance.pdf")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
