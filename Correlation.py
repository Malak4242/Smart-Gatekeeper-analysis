"""
Correlation Analysis on Lechevallier et al. (2025) Wastewater Dataset
========================================================================

Purpose
-------
Quantitatively identify redundant sensors in a multi-parameter IoT monitoring
setup for industrial effluent, using Pearson correlation on real laboratory
measurements. Output: correlation matrix, heatmap, and a minimal sensor set
recommendation.

Dataset
-------
Lechevallier, P., Gruber, G., Bares, V., Neuenhofer, N., Waldner, L.,
Mahajan, A., Mutzner, L., & Rieckermann, J. (2025). Dataset on wastewater
quality monitoring with adsorption and reflectance spectrometry in the UV-vis
range. Scientific Data, 12(1), 1296. https://doi.org/10.1038/s41597-025-05459-x

Download from: https://doi.org/10.25678/000D3C
File needed  : 2_data.zip -> extract to a folder called "data"

How to run
----------
1. pip install pandas numpy matplotlib seaborn scipy
2. Put this file next to the "data" folder from the dataset
3. python correlation_analysis.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# ---------------------------------------------------------------------------
# 1. LOAD THE LABORATORY REFERENCE MEASUREMENTS
# ---------------------------------------------------------------------------

DATA_DIR = "data"  # adjust if your folder name differs
LAB_FILE = os.path.join(
    DATA_DIR, "2_data", "5_laboratory_reference_measurements",
    "laboratory_measurements.csv"
)

# Fallback search if the exact path doesn't match
if not os.path.exists(LAB_FILE):
    hits = glob.glob(os.path.join(DATA_DIR, "**", "laboratory_measurements.csv"),
                     recursive=True)
    if hits:
        LAB_FILE = hits[0]
    else:
        raise FileNotFoundError(
            "Could not find laboratory_measurements.csv. Make sure you have "
            "extracted 2_data.zip from the ERIC open repository and placed "
            "it inside a folder called 'data' next to this script."
        )

print(f"Loading lab measurements from: {LAB_FILE}")
lab = pd.read_csv(LAB_FILE)
print(f"Loaded {len(lab)} rows, {len(lab.columns)} columns")
print(f"Columns found in file:\n{list(lab.columns)}\n")


# ---------------------------------------------------------------------------
# 2. SELECT THE POLLUTION-RELEVANT VARIABLES
# ---------------------------------------------------------------------------
# Column names in the dataset follow the pattern: lab_<variable>_<unit>
# We search for the variables relevant to sensor selection.
# ---------------------------------------------------------------------------

candidates = [
    "turbidity", "nh4", "nsol", "ntot", "doc", "toc", "tss", "po4", "so4"
]
selected_cols = []
for c in candidates:
    # Match whole underscore-delimited tokens to avoid partial overlaps
    match = [col for col in lab.columns
             if any(part == c for part in col.lower().split("_"))]
    if match:
        selected_cols.append(match[0])

print("All candidate columns found in file:")
for col in selected_cols:
    n_available = lab[col].apply(pd.to_numeric, errors="coerce").notna().sum()
    print(f"  - {col:25s} ({n_available} non-missing values)")
print()

# Split into CORE (measured for ~all 529 samples per the paper) and
# EXTENDED (measured only for the smaller subset: TSS, TOC, and N-total)
core_cols = [c for c in selected_cols if not any(k in c for k in ["tss", "toc", "ntot"])]
extended_cols = selected_cols  # all variables, smaller n expected

print(f"CORE group ({len(core_cols)} variables, larger n expected):")
print(f"  {core_cols}\n")
print(f"EXTENDED group ({len(extended_cols)} variables, smaller n expected):")
print(f"  {extended_cols}\n")

print("Variables selected for correlation analysis:")
for col in selected_cols:
    print(f"  - {col}")
print()

if len(selected_cols) < 2:
    raise ValueError(
        "Fewer than 2 matching columns found. Print lab.columns above and "
        "adjust the 'candidates' list to match the actual column names."
    )


# ---------------------------------------------------------------------------
# 3. RUN THE ANALYSIS TWICE: CORE GROUP AND EXTENDED GROUP
# ---------------------------------------------------------------------------

def run_correlation_analysis(cols, label):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}\n")

    d = lab[cols].apply(pd.to_numeric, errors="coerce").dropna()
    print(f"Rows remaining after removing missing values: {len(d)}\n")

    pearson = d.corr(method="pearson")
    spearman = d.corr(method="spearman")

    print("Pearson correlation matrix:")
    print(pearson.round(2))
    print()

    def pearson_pvalue_matrix(df):
        cs = df.columns
        p = pd.DataFrame(np.zeros((len(cs), len(cs))), index=cs, columns=cs)
        for i in cs:
            for j in cs:
                if i == j:
                    p.loc[i, j] = 0.0
                else:
                    _, pval = stats.pearsonr(df[i], df[j])
                    p.loc[i, j] = pval
        return p

    pvalues = pearson_pvalue_matrix(d)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        pearson, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        vmin=-1, vmax=1, square=True,
        cbar_kws={"label": "Pearson correlation coefficient (r)"}
    )
    plt.title(
        f"{label}\n(Lechevallier et al., 2025 dataset, n = {len(d)} samples)",
        fontsize=11
    )
    plt.tight_layout()
    fname = f"correlation_heatmap_{label.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Saved heatmap to: {fname}\n")

    print(f"Strong correlations (|r| > 0.7, p < 0.05) in {label}:")
    print("-" * 70)
    strong_pairs = []
    for i, c1 in enumerate(pearson.columns):
        for j, c2 in enumerate(pearson.columns):
            if j <= i:
                continue
            r = pearson.loc[c1, c2]
            p = pvalues.loc[c1, c2]
            if abs(r) > 0.7 and p < 0.05:
                print(f"  {c1:25s} <-> {c2:25s}  r = {r:+.2f}  (p = {p:.4f})")
                strong_pairs.append((c1, c2, r))
    if not strong_pairs:
        print("  (none found above the 0.7 threshold)")
    print()

    return d, pearson, strong_pairs


core_data, core_pearson, core_strong = run_correlation_analysis(
    core_cols, "CORE group"
)
ext_data, ext_pearson, ext_strong = run_correlation_analysis(
    extended_cols, "EXTENDED group"
)


# ---------------------------------------------------------------------------
# 4. WRITE UP THE FINDINGS FOR THE PAPER
# ---------------------------------------------------------------------------

with open("correlation_findings.txt", "w") as f:
    f.write("Correlation analysis findings\n")
    f.write("=============================\n\n")
    f.write("Dataset: Lechevallier et al. (2025)\n\n")

    f.write(f"CORE group (n = {len(core_data)} samples)\n")
    f.write("-" * 40 + "\n")
    f.write(core_pearson.round(2).to_string())
    f.write("\n\nStrong pairs (|r| > 0.7, p < 0.05):\n")
    for c1, c2, r in core_strong:
        f.write(f"  {c1} <-> {c2}: r = {r:+.2f}\n")
    f.write("\n\n")

    f.write(f"EXTENDED group (n = {len(ext_data)} samples)\n")
    f.write("-" * 40 + "\n")
    f.write(ext_pearson.round(2).to_string())
    f.write("\n\nStrong pairs (|r| > 0.7, p < 0.05):\n")
    for c1, c2, r in ext_strong:
        f.write(f"  {c1} <-> {c2}: r = {r:+.2f}\n")
    f.write("\n")

print("Wrote text findings to: correlation_findings.txt")
print("\nDone. You now have these artefacts for the paper:")
print("  1. correlation_heatmap_core_group.png       (larger sample, main figure)")
print("  2. correlation_heatmap_extended_group.png   (smaller sample, supplementary)")
print("  3. correlation_findings.txt                 (text for Results section)")