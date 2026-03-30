import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels import PanelOLS
from linearmodels.panel.results import compare
import os

print("=== Phase 4: High-Dimensional FE Regression Analysis (Table 3) ===")

INP = "/Users/xiwen/.gemini/antigravity/scratch/intelligence/charls_replication/data/cleaned_charls_phase1.csv"
OUT = "/Users/xiwen/.gemini/antigravity/scratch/intelligence/charls_replication/output"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(INP)

# Ensure required columns are present and clean
target_cols = ['ID', 'communityID', 'countyID', 'cesd_score', 'female']
controls = ['edu_primary', 'edu_middle', 'edu_high', 'rural_hukou', 
            'married', 'widowed', 'never_married', 
            'log_pce_low', 'log_pce_high',
            'age_50_54', 'age_55_59', 'age_60_64', 'age_65_69', 'age_70_74', 'age_75_plus']

df_valid = df.dropna(subset=target_cols + controls).copy()

males = df_valid[df_valid['female'] == 0].copy()
females = df_valid[df_valid['female'] == 1].copy()

def run_panel_models(data_subset, name):
    print(f"\nRunning High-Dimensional Fixed Effects Models for {name} (N={len(data_subset)})...")
    
    # Exogenous X matrix (Intercept is automatically handled by PanelOLS if absorbing effects, but good to add constant)
    X = sm.add_constant(data_subset[controls])
    Y = data_subset['cesd_score']
    
    # Specification 1: County FE
    # MultiIndex requirement: [Entity, Time]
    # We use countyID as Entity, and ID as Time to satisfy the panel dimension requirements
    df_county = data_subset.set_index(['countyID', 'ID'])
    X_c = sm.add_constant(df_county[controls])
    Y_c = df_county['cesd_score']
    
    model_county = PanelOLS(Y_c, X_c, entity_effects=True, drop_absorbed=True)
    res_county = model_county.fit(cov_type='robust')
    print(f"  ✅ Spec 1 (County FE) R^2: {res_county.rsquared:.4f}")

    # Specification 2: Community FE
    df_comm = data_subset.set_index(['communityID', 'ID'])
    X_cm = sm.add_constant(df_comm[controls])
    Y_cm = df_comm['cesd_score']
    
    model_comm = PanelOLS(Y_cm, X_cm, entity_effects=True, drop_absorbed=True)
    res_comm = model_comm.fit(cov_type='robust')
    print(f"  ✅ Spec 2 (Community FE) R^2: {res_comm.rsquared:.4f}")
    
    return res_county, res_comm

# Execute models for Males and Females
m_county, m_comm = run_panel_models(males, "Males")
f_county, f_comm = run_panel_models(females, "Females")

# Combine results into a formatted table
results = compare({
    'Male (County FE)': m_county,
    'Male (Comm FE)': m_comm,
    'Female (County FE)': f_county,
    'Female (Comm FE)': f_comm
})

print("\n" + str(results.summary))

with open(f"{OUT}/table3_stratified_fe_results.txt", 'w') as f:
    f.write(str(results.summary))

print(f"\n✅ Phase 4 Complete! Stratified regression results exported to {OUT}/table3_stratified_fe_results.txt")
