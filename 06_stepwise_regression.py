import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels import PanelOLS
from linearmodels.panel.results import compare
import os
from pathlib import Path

print("=== Phase 6: Stepwise Endogenous Regression (Tables 4-6) ===")
print("Lei et al. specification: progressively adding childhood health, family shocks, current health\n")

PROJECT_DIR = Path(__file__).resolve().parent
INP = str(PROJECT_DIR / "data/cleaned_charls_phase1.csv")
OUT = str(PROJECT_DIR / "output")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(INP)

# Base controls (Table 3 specification)
base_controls = [
    'edu_can_read', 'edu_primary', 'edu_junior_high_plus',
    'rural', 'married', 'widowed', 'never_married',
    'log_pce_low', 'log_pce_high',
    'age_50_54', 'age_55_59', 'age_60_64', 'age_65_69', 'age_70_74', 'age_75_plus'
]

# Stepwise layers (matching Lei et al. Tables 4, 5, 6)
add_childhood = ['poor_childhood_health', 'lower_leg_length']  # Table 4
add_family = ['child_died']                                     # Table 5
add_health = ['chronic_disease', 'adl_limit']                   # Table 6

target_cols = ['ID', 'communityID', 'cesd_score', 'female']

# Filter based on available columns
all_potential = base_controls + add_childhood + add_family + add_health
available_controls = [c for c in all_potential if c in df.columns and df[c].notna().any()]
available_base = [c for c in base_controls if c in available_controls]
available_childhood = [c for c in add_childhood if c in available_controls]
available_family = [c for c in add_family if c in available_controls]
available_health = [c for c in add_health if c in available_controls]

print(f"Available base controls: {len(available_base)}")
print(f"Available childhood vars: {available_childhood}")
print(f"Available family vars: {available_family}")
print(f"Available health vars: {available_health}")

df_valid = df.dropna(subset=target_cols + available_base).copy()
# For stepwise, also need to drop NA on endogenous vars
for v in available_childhood + available_family + available_health:
    df_valid = df_valid.dropna(subset=[v])

males = df_valid[df_valid['female'] == 0].copy()
females = df_valid[df_valid['female'] == 1].copy()

def build_stepwise_models(data_subset):
    df_cm = data_subset.set_index(['communityID', 'ID'])
    Y = df_cm['cesd_score']
    
    # Model 1: Base Model (Table 3 equivalent)
    X1 = sm.add_constant(df_cm[available_base])
    res1 = PanelOLS(Y, X1, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
    
    # Model 2: Base + Childhood Health (Table 4)
    step2_vars = available_base + available_childhood
    X2 = sm.add_constant(df_cm[step2_vars])
    res2 = PanelOLS(Y, X2, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
    
    # Model 3: Base + Childhood + Family (Table 5)
    step3_vars = available_base + available_childhood + available_family
    X3 = sm.add_constant(df_cm[step3_vars])
    res3 = PanelOLS(Y, X3, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
    
    # Model 4: Full Model (Table 6)
    step4_vars = available_base + available_childhood + available_family + available_health
    X4 = sm.add_constant(df_cm[step4_vars])
    res4 = PanelOLS(Y, X4, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
    
    return res1, res2, res3, res4

def extract_coefficients(res_tuple, model_names):
    records = []
    for model, name in zip(res_tuple, model_names):
        records.append({
            'Model': name,
            'edu_primary_coef': model.params.get('edu_primary', np.nan),
            'edu_primary_se': model.std_errors.get('edu_primary', np.nan),
            'edu_junior_high_plus_coef': model.params.get('edu_junior_high_plus', np.nan),
            'edu_junior_high_plus_se': model.std_errors.get('edu_junior_high_plus', np.nan),
            'log_pce_low_coef': model.params.get('log_pce_low', np.nan),
            'log_pce_low_se': model.std_errors.get('log_pce_low', np.nan)
        })
    return pd.DataFrame(records)

print("\nRunning models for Males...")
male_models = build_stepwise_models(males)
male_results = compare({f"M{i}": m for i, m in enumerate(male_models, 1)})
with open(f"{OUT}/table6_stepwise_males.txt", 'w') as f:
    f.write("Tables 4-6: Stepwise Regressions for Males\n")
    f.write("Community FE, clustered SE\n\n")
    f.write(str(male_results.summary))

print("Running models for Females...")
female_models = build_stepwise_models(females)
female_results = compare({f"M{i}": m for i, m in enumerate(female_models, 1)})
with open(f"{OUT}/table6_stepwise_females.txt", 'w') as f:
    f.write("Tables 4-6: Stepwise Regressions for Females\n")
    f.write("Community FE, clustered SE\n\n")
    f.write(str(female_results.summary))

# Export shrinkage data for visualization
models_seq = ['Model 1 (Base)', 'Model 2 (+Childhood)', 'Model 3 (+Family)', 'Model 4 (+Health)']
df_m_shrinkage = extract_coefficients(male_models, models_seq)
df_m_shrinkage['Gender'] = 'Male'
df_f_shrinkage = extract_coefficients(female_models, models_seq)
df_f_shrinkage['Gender'] = 'Female'

df_shrinkage = pd.concat([df_m_shrinkage, df_f_shrinkage])
df_shrinkage.to_csv(f"{OUT}/coefficient_shrinkage_data.csv", index=False)

print(f"\n✅ Phase 6 Complete! Stepwise results exported.")
