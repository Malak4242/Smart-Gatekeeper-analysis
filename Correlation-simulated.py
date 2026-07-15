"""
Qarun-Representative Synthetic Sensor Data: Simulation and Correlation Analysis
=================================================================================

Purpose
-------
Since no public multi-parameter sensor dataset exists for industrial discharge
near Lake Qarun, this script generates a synthetic dataset using Monte Carlo
simulation, parameterized with published Lake Qarun water quality statistics
and established physicochemical relationships. The same correlation-analysis
method used on the real Lechevallier et al. (2025) dataset is then applied
here, to illustrate what sensor redundancy looks like for the actual sensor
set proposed for the Qarun system (salinity/EC, pH, temperature, DO,
turbidity, flow).

IMPORTANT — read before citing results
---------------------------------------
This is a SIMULATION, not real field data. Some correlations are guaranteed
by construction (e.g., EC and salinity, since one is computed from the other)
and should be reported as such, not as independent findings. Only the
emergent relationships (e.g., temperature-DO, temperature-salinity,
turbidity-vs-others) reflect genuine modeling assumptions worth discussing.

Parameter sources (cite these in your methodology section)
------------------------------------------------------------
- Salinity 35-38.5 PSU, EC 43-45.2 dS/m: Qarun basin water quality study
  (cite the specific paper you pulled these numbers from)
- pH 8.7-9.3: same source
- DO 6.94-10.58 mg/L (APHA accepted range 4.0-12.6 mg/L): same source
- Temperature ~20 degC average, rising with evaporation: same source (2019
  Qarun Lake temperature study, Figure 11)
- Turbidity and flow: no Qarun-specific published values found; modeled
  using generic industrial-discharge assumptions (state this explicitly as
  a limitation)

How to run
----------
1. pip install pandas numpy matplotlib seaborn scipy
2. python qarun_simulation_and_correlation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(42)  # reproducibility - anyone re-running this gets the same result

N_SAMPLES = 500  # matches the scale of the Lechevallier core-group analysis


# ---------------------------------------------------------------------------
# 1. TEMPERATURE (degrees C)
# ---------------------------------------------------------------------------
# Baseline ~20 degC per published Qarun 2019 temperature study, with natural
# day-to-day / seasonal variation. Clipped to a physically realistic range.
# ---------------------------------------------------------------------------

temperature = np.random.normal(loc=20.0, scale=3.5, size=N_SAMPLES)
temperature = np.clip(temperature, 10, 32)


# ---------------------------------------------------------------------------
# 2. SALINITY (PSU) -- linked to temperature via evaporation
# ---------------------------------------------------------------------------
# Physical reasoning: higher temperature -> higher evaporation -> water
# volume drops while dissolved salt remains -> salinity concentrates.
# We encode this as a modest positive linear dependency on temperature,
# around the published baseline range of 35-38.5 PSU, plus random noise.
# ---------------------------------------------------------------------------

salinity_baseline = 36.5  # midpoint of published 35-38.5 PSU range
temp_deviation = temperature - temperature.mean()
salinity = (
    salinity_baseline
    + 0.25 * temp_deviation          # evaporation-driven concentration effect
    + np.random.normal(0, 1.2, N_SAMPLES)   # natural measurement/process noise
)

# Inject occasional salinity SPIKE EVENTS: represents an industrial discharge
# event with elevated salt content (the scenario your monitoring system is
# specifically designed to catch)
salinity_event = np.random.random(N_SAMPLES) < 0.08  # 8% of samples
salinity = np.where(
    salinity_event,
    salinity + np.random.uniform(4, 10, N_SAMPLES),  # spike magnitude
    salinity
)
salinity = np.clip(salinity, 30, 55)


# ---------------------------------------------------------------------------
# 3. ELECTRICAL CONDUCTIVITY (dS/m) -- derived from salinity
# ---------------------------------------------------------------------------
# EC and salinity are essentially the same physical signal (dissolved ion
# concentration), so this is a near-direct conversion, not an independent
# variable. A conversion factor is calibrated so that salinity ~36.5 PSU
# maps to EC ~44 dS/m, matching the published Qarun EC range (43-45.2 dS/m).
# Small sensor noise is added on top of the conversion.
# ---------------------------------------------------------------------------

EC_CONVERSION_FACTOR = 44.0 / 36.5
EC = salinity * EC_CONVERSION_FACTOR + np.random.normal(0, 0.3, N_SAMPLES)


# ---------------------------------------------------------------------------
# 4. pH -- baseline near published 8.7-9.3 range, with occasional excursions
# ---------------------------------------------------------------------------
# Modeled as largely independent of salinity/temperature (pH in Qarun is
# driven mainly by carbonate chemistry and biological activity, not directly
# by evaporation), plus occasional cleaning-cycle-type pH spikes.
# ---------------------------------------------------------------------------

pH = np.random.normal(loc=9.0, scale=0.18, size=N_SAMPLES)

pH_event = np.random.random(N_SAMPLES) < 0.05  # 5% of samples
pH = np.where(
    pH_event,
    pH + np.random.uniform(0.4, 0.9, N_SAMPLES),
    pH
)
pH = np.clip(pH, 6.5, 10.5)


# ---------------------------------------------------------------------------
# 5. DISSOLVED OXYGEN (mg/L) -- inversely related to temperature
# ---------------------------------------------------------------------------
# Standard water chemistry: oxygen solubility decreases as temperature rises.
# Baseline calibrated to published Qarun DO range (6.94-10.58 mg/L, APHA
# accepted range 4.0-12.6 mg/L). Salinity also modestly reduces oxygen
# solubility (a real, well-documented effect), so a smaller negative term
# for salinity is included too.
# ---------------------------------------------------------------------------

DO = (
    14.5
    - 0.22 * temperature          # primary driver: warmer water holds less O2
    - 0.05 * (salinity - salinity_baseline)  # secondary: salinity effect
    + np.random.normal(0, 0.6, N_SAMPLES)    # measurement noise
)
DO = np.clip(DO, 3, 13)


# ---------------------------------------------------------------------------
# 6. TURBIDITY (NTU) -- modeled as largely independent
# ---------------------------------------------------------------------------
# No Qarun-specific turbidity values were found in the literature reviewed
# for this study (state this explicitly as a limitation). Modeled using a
# generic industrial-effluent turbidity distribution with occasional spikes
# representing suspended-solid discharge events.
# ---------------------------------------------------------------------------

turbidity = np.random.lognormal(mean=3.2, sigma=0.5, size=N_SAMPLES)  # skewed, realistic for turbidity
turbidity_event = np.random.random(N_SAMPLES) < 0.06
turbidity = np.where(
    turbidity_event,
    turbidity + np.random.uniform(50, 150, N_SAMPLES),
    turbidity
)
turbidity = np.clip(turbidity, 1, 400)


# ---------------------------------------------------------------------------
# 7. FLOW (L/s) -- steady baseline with occasional anomalies
# ---------------------------------------------------------------------------
# Represents discharge flow rate. Occasional drops simulate either production
# downtime or the tampering scenario (manual valve bypass) the system is
# designed to flag.
# ---------------------------------------------------------------------------

flow = np.random.normal(loc=12.0, scale=1.5, size=N_SAMPLES)
flow_event = np.random.random(N_SAMPLES) < 0.05
flow = np.where(
    flow_event,
    flow * np.random.uniform(0.2, 0.5, N_SAMPLES),  # sudden drop
    flow
)
flow = np.clip(flow, 0, 20)


# ---------------------------------------------------------------------------
# 8. ASSEMBLE THE DATASET
# ---------------------------------------------------------------------------

df = pd.DataFrame({
    "sim_salinity_psu": salinity,
    "sim_ec_dsm": EC,
    "sim_ph": pH,
    "sim_temperature_c": temperature,
    "sim_do_mgl": DO,
    "sim_turbidity_ntu": turbidity,
    "sim_flow_ls": flow,
})

df.to_csv("qarun_simulated_data.csv", index=False)
print(f"Generated {len(df)} synthetic samples -> qarun_simulated_data.csv\n")
print("Summary statistics:")
print(df.describe().round(2))
print()


# ---------------------------------------------------------------------------
# 9. CORRELATION ANALYSIS (SAME METHOD AS THE LECHEVALLIER ANALYSIS)
# ---------------------------------------------------------------------------

pearson = df.corr(method="pearson")


def pearson_pvalue_matrix(data):
    cols = data.columns
    p = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    for i in cols:
        for j in cols:
            if i == j:
                p.loc[i, j] = 0.0
            else:
                _, pval = stats.pearsonr(data[i], data[j])
                p.loc[i, j] = pval
    return p


pvalues = pearson_pvalue_matrix(df)

print("Pearson correlation matrix (simulated Qarun-context data):")
print(pearson.round(2))
print()

plt.figure(figsize=(9, 7))
sns.heatmap(
    pearson, annot=True, fmt=".2f", cmap="coolwarm", center=0,
    vmin=-1, vmax=1, square=True,
    cbar_kws={"label": "Pearson correlation coefficient (r)"}
)
plt.title(
    f"Correlation between sensors — simulated Qarun discharge context\n"
    f"(Monte Carlo simulation, n = {len(df)} samples)",
    fontsize=11
)
plt.tight_layout()
plt.savefig("correlation_heatmap_qarun_simulated.png", dpi=200)
plt.close()
print("Saved heatmap to: correlation_heatmap_qarun_simulated.png\n")

print("Correlations, sorted strongest to weakest (excluding self-pairs):")
print("-" * 70)
pairs = []
for i, c1 in enumerate(pearson.columns):
    for j, c2 in enumerate(pearson.columns):
        if j <= i:
            continue
        r = pearson.loc[c1, c2]
        p = pvalues.loc[c1, c2]
        pairs.append((c1, c2, r, p))

pairs.sort(key=lambda x: -abs(x[2]))
for c1, c2, r, p in pairs:
    flag = ""
    if c1 == "sim_ec_dsm" and c2 == "sim_salinity_psu" or c1 == "sim_salinity_psu" and c2 == "sim_ec_dsm":
        flag = "  <-- by construction (EC derived from salinity), not an independent finding"
    print(f"  {c1:22s} <-> {c2:22s}  r = {r:+.2f}  (p = {p:.4f}){flag}")
print()


# ---------------------------------------------------------------------------
# 10. WRITE UP FINDINGS FOR THE PAPER
# ---------------------------------------------------------------------------

with open("qarun_correlation_findings.txt", "w") as f:
    f.write("Qarun-context simulated data: correlation analysis findings\n")
    f.write("=============================================================\n\n")
    f.write(f"Samples: {len(df)} (Monte Carlo simulation)\n\n")
    f.write("NOTE: this is simulated data, parameterized from published Lake\n")
    f.write("Qarun water quality statistics and known physicochemical\n")
    f.write("relationships. Not real field measurements.\n\n")

    f.write("Pearson correlation matrix\n")
    f.write("---------------------------\n")
    f.write(pearson.round(2).to_string())
    f.write("\n\n")

    f.write("All pairs, sorted by |r|:\n")
    for c1, c2, r, p in pairs:
        f.write(f"  {c1} <-> {c2}: r = {r:+.2f} (p = {p:.4f})\n")
    f.write("\n")

print("Wrote findings to: qarun_correlation_findings.txt")
print("\nDone. Compare this heatmap and matrix against the Lechevallier")
print("core-group results to build your urban-vs-Qarun contrast argument.")