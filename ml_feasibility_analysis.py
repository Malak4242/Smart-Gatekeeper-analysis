"""
ML Feasibility: Predicting Organic Pollution Load from Low-Cost Sensors
==========================================================================

Purpose
-------
Demonstrates the feasibility of predicting an organic pollution indicator
(dissolved organic carbon, DOC) using only cheap, easily-measured sensor
readings (turbidity, ammonium, phosphate, sulfate) rather than expensive lab
analysis. This supports the ML layer of the proposed effluent monitoring
system: rather than requiring costly COD/BOD lab tests, a trained model
estimates organic pollution load in real time from low-cost sensors.

IMPORTANT — honest framing for the paper
------------------------------------------
Direct COD/BOD lab measurements were not available at matching timestamps
with the sensor readings in this dataset. DOC (dissolved organic carbon) is
used as a validated proxy for organic pollution load. This demonstrates
feasibility of the general approach; extending to COD/BOD directly would
require a dataset with matched timestamps between grab-sample lab COD/BOD
values and continuous sensor readings.

Dataset
-------
Lechevallier, P., Gruber, G., Bares, V., Neuenhofer, N., Waldner, L.,
Mahajan, A., Mutzner, L., & Rieckermann, J. (2025). Dataset on wastewater
quality monitoring with adsorption and reflectance spectrometry in the UV-vis
range. Scientific Data, 12(1), 1296. https://doi.org/10.1038/s41597-025-05459-x

How to run
----------
1. pip install pandas numpy matplotlib seaborn scikit-learn scipy
2. Place this script next to your "data" folder (same as the correlation script)
3. python ml_feasibility_analysis.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

np.random.seed(42)


# ---------------------------------------------------------------------------
# 1. LOAD THE DATA (same loading logic as the correlation script)
# ---------------------------------------------------------------------------

DATA_DIR = "data"
LAB_FILE = os.path.join(
    DATA_DIR, "2_data", "2_data", "5_laboratory_reference_measurements",
    "laboratory_measurements.csv"
)

if not os.path.exists(LAB_FILE):
    hits = glob.glob(os.path.join(DATA_DIR, "**", "laboratory_measurements.csv"),
                     recursive=True)
    if hits:
        LAB_FILE = hits[0]
    else:
        raise FileNotFoundError(
            "Could not find laboratory_measurements.csv. Check your data folder."
        )

print(f"Loading data from: {LAB_FILE}")
lab = pd.read_csv(LAB_FILE)
print(f"Loaded {len(lab)} rows\n")


# ---------------------------------------------------------------------------
# 2. DEFINE FEATURES (cheap sensors) AND TARGET (organic load proxy)
# ---------------------------------------------------------------------------
# Features: variables that correspond to cheap, real-time-capable sensors
# Target:   DOC, used as the organic pollution load proxy (see docstring)
# ---------------------------------------------------------------------------

feature_cols = ["lab_turbidity_ntu", "lab_nh4_mg_l", "lab_po4_mg_l", "lab_so4_mg_l"]
target_col = "lab_doc_mg_l"

all_cols = feature_cols + [target_col]
missing = [c for c in all_cols if c not in lab.columns]
if missing:
    raise ValueError(f"Expected columns not found: {missing}. "
                      f"Available columns: {list(lab.columns)}")

data = lab[all_cols].apply(pd.to_numeric, errors="coerce").dropna()
print(f"Rows with complete feature + target data: {len(data)}\n")

X = data[feature_cols]
y = data[target_col]


# ---------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
# 80% of data used to train the model, 20% held out to honestly evaluate it
# on samples the model has never seen.
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}\n")


# ---------------------------------------------------------------------------
# 4. TRAIN THE RANDOM FOREST MODEL
# ---------------------------------------------------------------------------
# Random Forest handles correlated features naturally (relevant here, since
# turbidity/nutrients showed moderate correlation in the earlier analysis)
# and requires no feature scaling.
# ---------------------------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,          # kept shallow to avoid overfitting on a small dataset
    min_samples_leaf=3,
    random_state=42
)
model.fit(X_train, y_train)


# ---------------------------------------------------------------------------
# 4b. LOG-TRANSFORMED MODEL (handles the right-skewed DOC distribution)
# ---------------------------------------------------------------------------
# DOC concentrations are right-skewed (a small number of high-pollution
# samples, most samples in a lower range). Tree-based models tend to
# underpredict the high tail of skewed targets. Training on log(DOC) instead
# of raw DOC is a standard, well-established correction for this situation
# in environmental data analysis.
# ---------------------------------------------------------------------------

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

model_log = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=3,
    random_state=42
)
model_log.fit(X_train, y_train_log)

y_pred_log_space = model_log.predict(X_test)
y_pred_log_model = np.expm1(y_pred_log_space)


# ---------------------------------------------------------------------------
# 5. EVALUATE ON THE HELD-OUT TEST SET
# ---------------------------------------------------------------------------

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# Evaluate the log-transformed model on the SAME original mg/L scale, so the
# two approaches are directly comparable

rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log_model))
mae_log = mean_absolute_error(y_test, y_pred_log_model)

print("=" * 60)
print("MODEL COMPARISON (both evaluated on original mg/L scale)")
print("=" * 60)
print(f"{'Metric':<12} {'Raw-scale model':>18} {'Log-transformed model':>24}")
print(f"{'R2':<12} {r2:>18.3f} {r2_log:>24.3f}")
print(f"{'RMSE':<12} {rmse:>18.3f} {rmse_log:>24.3f}")
print(f"{'MAE':<12} {mae:>18.3f} {mae_log:>24.3f}")
print()

# Simple, honest interpretation guide printed for reference
if r2 > 0.7:
    verdict = "Strong predictive performance."
elif r2 > 0.4:
    verdict = "Moderate predictive performance - demonstrates feasibility but with meaningful error."
elif r2 > 0.1:
    verdict = "Weak predictive performance - limited feasibility with these features alone."
else:
    verdict = "Little to no predictive power with these features."
print(f"Interpretation: {verdict}\n")


# ---------------------------------------------------------------------------
# 6. FEATURE IMPORTANCE
# ---------------------------------------------------------------------------
# Shows which cheap sensors contribute most to the prediction - useful for
# justifying which sensors are worth keeping if cost is a constraint.
# ---------------------------------------------------------------------------

importances = pd.Series(model.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=False)

print("Feature importance (higher = more predictive of DOC):")
for feat, imp in importances.items():
    print(f"  {feat:22s}  {imp:.3f}")
print()

plt.figure(figsize=(7, 4))
importances.plot(kind="barh", color="#4C72B0")
plt.xlabel("Feature importance")
plt.title("Which sensors matter most for predicting organic load (DOC)?")
plt.tight_layout()
plt.savefig("ml_feature_importance.png", dpi=200)
plt.close()
print("Saved: ml_feature_importance.png\n")


# ---------------------------------------------------------------------------
# 7. PREDICTED VS ACTUAL SCATTER PLOT
# ---------------------------------------------------------------------------

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color="#55A868", edgecolor="k", linewidth=0.3)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, 'r--', label="Perfect prediction")
plt.xlabel("Actual DOC (mg/L)")
plt.ylabel("Predicted DOC (mg/L)")
plt.title(f"Predicted vs. actual DOC — raw-scale model\n(R2 = {r2:.3f}, n_test = {len(y_test)})")
plt.legend()
plt.tight_layout()
plt.savefig("ml_predicted_vs_actual.png", dpi=200)
plt.close()
print("Saved: ml_predicted_vs_actual.png\n")


# ---------------------------------------------------------------------------
# 7b. PREDICTED VS ACTUAL — LOG-TRANSFORMED MODEL (both on mg/L scale)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].scatter(y_test, y_pred, alpha=0.6, color="#55A868", edgecolor="k", linewidth=0.3)
lims_all = [0, max(y_test.max(), y_pred.max(), y_pred_log_model.max()) * 1.05]
axes[0].plot(lims_all, lims_all, 'r--', label="Perfect prediction")
axes[0].set_xlabel("Actual DOC (mg/L)")
axes[0].set_ylabel("Predicted DOC (mg/L)")
axes[0].set_title(f"Raw-scale model\nR2 = {r2:.3f}")
axes[0].legend()
axes[0].set_xlim(lims_all)
axes[0].set_ylim(lims_all)

axes[1].scatter(y_test, y_pred_log_model, alpha=0.6, color="#C44E52", edgecolor="k", linewidth=0.3)
axes[1].plot(lims_all, lims_all, 'r--', label="Perfect prediction")
axes[1].set_xlabel("Actual DOC (mg/L)")
axes[1].set_ylabel("Predicted DOC (mg/L)")
axes[1].set_title(f"Log-transformed model\nR2 = {r2_log:.3f}")
axes[1].legend()
axes[1].set_xlim(lims_all)
axes[1].set_ylim(lims_all)

plt.suptitle(f"Model comparison: raw vs. log-transformed target (n_test = {len(y_test)})")
plt.tight_layout()
plt.savefig("ml_model_comparison.png", dpi=200)
plt.close()
print("Saved: ml_model_comparison.png\n")


# ---------------------------------------------------------------------------
# 8. WRITE UP FINDINGS FOR THE PAPER
# ---------------------------------------------------------------------------

with open("ml_findings.txt", "w") as f:
    f.write("ML feasibility analysis: predicting organic load from cheap sensors\n")
    f.write("======================================================================\n\n")
    f.write("Target variable: DOC (dissolved organic carbon), used as an organic\n")
    f.write("pollution load proxy. Direct COD/BOD lab measurements were not\n")
    f.write("available at matching timestamps with sensor readings in this dataset.\n\n")
    f.write(f"Features used: {', '.join(feature_cols)}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n\n")

    f.write("Performance on held-out test set (both evaluated on mg/L scale)\n")
    f.write("------------------------------------------------------------------\n")
    f.write(f"{'Metric':<12} {'Raw-scale model':>18} {'Log-transformed model':>24}\n")
    f.write(f"{'R2':<12} {r2:>18.3f} {r2_log:>24.3f}\n")
    f.write(f"{'RMSE':<12} {rmse:>18.3f} {rmse_log:>24.3f}\n")
    f.write(f"{'MAE':<12} {mae:>18.3f} {mae_log:>24.3f}\n\n")

    f.write("Note: DOC is right-skewed (mean {:.1f}, std {:.1f}, max {:.1f} mg/L).\n".format(
        y_test.mean(), y_test.std(), y_test.max()))
    f.write("The log-transformed model is trained on log(1+DOC) and is a standard\n")
    f.write("correction for skewed environmental concentration data; both models\n")
    f.write("are reported here on the original mg/L scale for direct comparison.\n\n")

    f.write("Feature importance\n")
    f.write("-------------------\n")
    for feat, imp in importances.items():
        f.write(f"  {feat}: {imp:.3f}\n")
    f.write("\n")

    f.write("Model: Random Forest Regressor (200 trees, max depth 6)\n")

print("Wrote findings to: ml_findings.txt")
print("\nDone. You now have three artefacts for the paper:")
print("  1. ml_predicted_vs_actual.png   (Figure for Results section)")
print("  2. ml_feature_importance.png    (Figure showing which sensors matter)")
print("  3. ml_findings.txt              (text + numbers for Results section)")