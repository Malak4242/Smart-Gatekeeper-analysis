# Smart-Gatekeeper-analysis
AIoT-GIS Environmental Decision Support System for Real-Time Industrial Wastewater Compliance Monitoring: A Case Study of the Lake Qarun Basin 
# Smart Gatekeeper — Analysis Scripts

Sensor correlation and machine learning feasibility analysis supporting the **Smart Gatekeeper** IoT-GIS environmental decision support system for real-time industrial wastewater compliance monitoring (Lake Qarun basin case study).

This repository contains the analysis code used to:
1. Quantify redundancy among common water quality sensors using Pearson correlation.
2. Test the feasibility of predicting organic pollution load (using dissolved organic carbon, DOC, as a proxy) from low-cost sensor readings via a Random Forest model.

## Contents

| File | Description |
|---|---|
| `correlation_analysis.py` | Runs Pearson correlation analysis on the reference wastewater dataset to identify redundant sensors, and generates correlation heatmaps for a "core" and "extended" variable group. |
| `ml_feasibility_analysis.py` | Trains a Random Forest Regressor to predict DOC from turbidity, ammonium, phosphate, and sulfate readings, and compares a raw-scale model against a log-transformed model. |

## Dataset

This analysis uses the following open dataset (not included in this repository due to file size):

> Lechevallier, P., Gruber, G., Bares, V., Neuenhofer, N., Waldner, L., Mahajan, A., Mutzner, L., & Rieckermann, J. (2025). *Dataset on wastewater quality monitoring with adsorption and reflectance spectrometry in the UV-vis range.* Scientific Data, 12(1), 1296. https://doi.org/10.1038/s41597-025-05459-x

Download from: https://doi.org/10.25678/000D3C (file: `2_data.zip`)

**Setup:** Extract `2_data.zip` and place it inside a folder named `data/` in the project root, so that the following path exists:
```
data/2_data/5_laboratory_reference_measurements/laboratory_measurements.csv
```

## Requirements

```
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## How to run

```
python correlation_analysis.py
python ml_feasibility_analysis.py
```

Each script will print results to the console and save output figures (`.png`) and findings (`.txt`) to the project directory.

## Outputs

**`correlation_analysis.py`** produces:
- `correlation_heatmap_core_group.png`
- `correlation_heatmap_extended_group.png`
- `correlation_findings.txt`

**`ml_feasibility_analysis.py`** produces:
- `ml_predicted_vs_actual.png`
- `ml_feature_importance.png`
- `ml_model_comparison.png`
- `ml_findings.txt`

## Context

This code supports the Results section of the paper *"AIoT-GIS Environmental Decision Support System for Real-Time Industrial Wastewater Compliance Monitoring: A Case Study of the Lake Qarun Basin"* (Ahmed & Khafagy). The correlation analysis validates sensor redundancy assumptions used to justify the reduced sensor set in the proposed system architecture; the ML feasibility analysis evaluates whether organic pollution load can be estimated from low-cost sensors in place of costly COD/BOD lab testing.


## Soft-sensor analysis (COD/BOD)
See [`soft_sensor/`](soft_sensor/README.md) for the chronological-validation robustness check and the variable-importance analysis on the essential-oil WRRF subsample (Hassnain et al., 2025).
