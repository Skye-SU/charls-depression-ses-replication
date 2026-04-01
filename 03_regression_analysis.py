import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels import PanelOLS
from linearmodels.panel.results import compare
import os
from pathlib import Path

print("=== Phase 3: Stratified FE Regression Analysis (Table 3) ===")
print("Replicating Lei et al. (2014) Table 3: CES-D regressions\n")

PROJECT_DIR = Path(__file__).resolve().parent
INP = str(PROJECT_DIR / "data/cleaned_charls_phase1.csv")
OUT = str(PROJECT_DIR / "output")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(INP)

# Lei et al. Table 3 specification:
# Covariates: age dummies, education dummies, logPCE spline, rural (county spec only)
# Spec (1): County FE + rural dummy
# Spec (2): Community FE (absorbs rural)
# All clustered at community level, stratified by gender

age_dummies = ['age_50_54', 'age_55_59', 'age_60_64', 'age_65_69', 'age_70_74', 'age_75_plus']
edu_dummies = ['edu_can_read', 'edu_primary', 'edu_junior_high_plus']
pce_spline  = ['log_pce_low', 'log_pce_high']

# For county FE spec: include rural dummy
controls_county = age_dummies + edu_dummies + pce_spline + ['rural']
# For community FE spec: rural is absorbed by community FE
controls_comm = age_dummies + edu_dummies + pce_spline

# Filter to valid observations
target_cols = ['ID', 'communityID', 'countyID', 'cesd_score', 'female']
all_needed = target_cols + controls_county
df_valid = df.dropna(subset=all_needed).copy()

print(f"Valid observations: {len(df_valid)} (M={len(df_valid[df_valid['female']==0])}, F={len(df_valid[df_valid['female']==1])})")
print(f"Paper: M=7132, F=7708\n")

males = df_valid[df_valid['female'] == 0].copy()
females = df_valid[df_valid['female'] == 1].copy()

def run_stratified_models(data_subset, name):
    """Run County FE and Community FE models for one gender."""
    print(f"Running models for {name} (N={len(data_subset)})...")
    
    # Spec 1: County FE (countyID as entity)
    df_county = data_subset.set_index(['countyID', 'ID'])
    X_c = sm.add_constant(df_county[controls_county])
    Y_c = df_county['cesd_score']
    model_county = PanelOLS(Y_c, X_c, entity_effects=True, drop_absorbed=True)
    # Cluster at community level (Lei et al.: "clustered at community level")
    res_county = model_county.fit(cov_type='clustered', cluster_entity=True)
    print(f"  ✅ Spec 1 (County FE): R²={res_county.rsquared:.4f}")
    
    # Spec 2: Community FE (communityID as entity)
    df_comm = data_subset.set_index(['communityID', 'ID'])
    X_cm = sm.add_constant(df_comm[controls_comm])
    Y_cm = df_comm['cesd_score']
    model_comm = PanelOLS(Y_cm, X_cm, entity_effects=True, drop_absorbed=True)
    res_comm = model_comm.fit(cov_type='clustered', cluster_entity=True)
    print(f"  ✅ Spec 2 (Community FE): R²={res_comm.rsquared:.4f}")
    
    return res_county, res_comm

# Execute models
m_county, m_comm = run_stratified_models(males, "Males")
f_county, f_comm = run_stratified_models(females, "Females")

# Combine results into formatted table
results = compare({
    'Male (1) County FE': m_county,
    'Male (2) Comm FE': m_comm,
    'Female (1) County FE': f_county,
    'Female (2) Comm FE': f_comm
})

print("\n" + str(results.summary))

with open(f"{OUT}/table3_stratified_fe_results.txt", 'w') as f:
    f.write("Table 3: Regressions for CES-D (Lei et al. 2014 Replication)\n")
    f.write("Standard errors clustered at community level.\n")
    f.write("Education baseline: Illiterate. Age baseline: 45-49.\n\n")
    f.write(str(results.summary))

print(f"\n✅ Phase 3 Complete! Results exported to {OUT}/table3_stratified_fe_results.txt")
