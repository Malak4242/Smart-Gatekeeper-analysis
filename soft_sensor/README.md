# Soft-sensor analysis: COD and BOD from operational variables

Supports the Results and Discussion of the paper (seven-model comparison, direct vs two-stage BOD,
robustness check, variable importance).

**Data (not included):** 10% public subsample of the essential-oil wastewater dataset, Hassnain, Lee & Azhar (2025),
https://doi.org/10.5281/zenodo.16784954. Download `ActualData_Subsample.xlsx` and pass its path to the scripts.

| File | What it does |
|---|---|
| `robustness_check.py` | Extra Trees, all 11 variables: shuffled 5-fold CV vs contiguous-block CV vs final 15% chronological holdout; drop-one ablation; online-only (7 inputs) model. Writes `robustness_results.csv`. |
| `feature_importance_figure.py` | Random Forest COD and BOD (direct) with all ten non-target variables; permutation importance (30 repeats) on the final 15% chronological test set, plus Spearman rho. Writes `feature_importance.png/.pdf/.csv`. |

```
pip install pandas numpy scikit-learn scipy matplotlib openpyxl
python robustness_check.py path/to/ActualData_Subsample.xlsx
python feature_importance_figure.py path/to/ActualData_Subsample.xlsx
```

Records are sorted by timestamp; the first 85% is used for training/tuning (5-fold TimeSeriesSplit, RMSE), the last 15% for testing.

## Reproduced numbers
| Check | Result |
|---|---|
| Extra Trees, shuffled 5-fold CV | R² 0.974 |
| Extra Trees, contiguous-block 5-fold CV | R² 0.938 |
| Extra Trees, final 15% chronological holdout | R² 0.924 |
| Extra Trees, 7 online-only inputs, holdout | R² 0.927 |
| RF with all ten variables, test R² | COD 0.659, BOD 0.891 |

## Caveats
- Importance values show what the model relies on, not causal effects. Correlated inputs (the flows, suspended solids) share credit, and flow may partly stand in for the operating period.
- The ten-variable models include laboratory variables (suspended solids, TKN, nitrate) and are used only for the importance analysis; the soft-sensor models in the paper use the seven operational inputs.
- The dataset has data-quality issues (BOD > COD in 71.8% of rows, median pH 3.15, repeated dates). It is not a Lake Qarun dataset.
