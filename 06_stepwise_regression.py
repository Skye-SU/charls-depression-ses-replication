import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels import PanelOLS
from linearmodels.panel.results import compare
import os
from pathlib import Path

print("=== Phase 6: Stepwise Endogenous Regression (Tables 4-6) ===")

PROJECT_DIR = Path(__file__).resolve().parent
INP = str(PROJECT_DIR / "data/cleaned_charls_phase1.csv")
OUT = str(PROJECT_DIR / "output")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(INP)

# Base controls
base_controls = ['edu_primary', 'edu_middle', 'edu_high', 'rural_hukou', 
                 'married', 'widowed', 'never_married', 
                 'log_pce_low', 'log_pce_high',
                 'age_50_54', 'age_55_59', 'age_60_64', 'age_65_69', 'age_70_74', 'age_75_plus']

# Stepwise layers
add_biomarker = ['lower_leg_length']
add_family = ['child_died']
add_health = ['chronic_disease_count', 'adl_limit']

target_cols = ['ID', 'communityID', 'cesd_score', 'female']
all_controls = base_controls + add_biomarker + add_family + add_health
df_valid = df.dropna(subset=target_cols + all_controls).copy()

males = df_valid[df_valid['female'] == 0].copy()
females = df_valid[df_valid['female'] == 1].copy()

def build_stepwise_models(data_subset):
    df_cm = data_subset.set_index(['communityID', 'ID'])
    Y = df_cm['cesd_score']
    
    # Model 1: Base Model (Table 3 equivalent)
    X1 = sm.add_constant(df_cm[base_controls])
    res1 = PanelOLS(Y, X1, entity_effects=True, drop_absorbed=True).fit(cov_type='robust')
    
    # Model 2: Base + Biomarker (Table 4)
    X2 = sm.add_constant(df_cm[base_controls + add_biomarker])
    res2 = PanelOLS(Y, X2, entity_effects=True, drop_absorbed=True).fit(cov_type='robust')
    
    # Model 3: Base + Biomarker + Family (Table 5)
    X3 = sm.add_constant(df_cm[base_controls + add_biomarker + add_family])
    res3 = PanelOLS(Y, X3, entity_effects=True, drop_absorbed=True).fit(cov_type='robust')
    
    # Model 4: Full Model (Table 6)
    X4 = sm.add_constant(df_cm[all_controls])
    res4 = PanelOLS(Y, X4, entity_effects=True, drop_absorbed=True).fit(cov_type='robust')
    
    return res1, res2, res3, res4

def extract_coefficients(res_tuple, model_names):
    records = []
    for model, name in zip(res_tuple, model_names):
        records.append({
            'Model': name,
            'edu_middle_coef': model.params.get('edu_middle', np.nan),
            'edu_middle_se': model.std_errors.get('edu_middle', np.nan),
            'edu_high_coef': model.params.get('edu_high', np.nan),
            'edu_high_se': model.std_errors.get('edu_high', np.nan),
            'log_pce_low_coef': model.params.get('log_pce_low', np.nan),
            'log_pce_low_se': model.std_errors.get('log_pce_low', np.nan)
        })
    return pd.DataFrame(records)

print("Running models for Males...")
male_models = build_stepwise_models(males)
male_results = compare({f"M{i}": m for i, m in enumerate(male_models, 1)})
with open(f"{OUT}/table6_stepwise_males.txt", 'w') as f:
    f.write(str(male_results.summary))

print("Running models for Females...")
female_models = build_stepwise_models(females)
female_results = compare({f"M{i}": m for i, m in enumerate(female_models, 1)})
with open(f"{OUT}/table6_stepwise_females.txt", 'w') as f:
    f.write(str(female_results.summary))

# Export shrinkage data for visualization
models_seq = ['Model 1 (Base)', 'Model 2 (+Bio)', 'Model 3 (+Family)', 'Model 4 (+Health)']
df_m_shrinkage = extract_coefficients(male_models, models_seq)
df_m_shrinkage['Gender'] = 'Male'
df_f_shrinkage = extract_coefficients(female_models, models_seq)
df_f_shrinkage['Gender'] = 'Female'

df_shrinkage = pd.concat([df_m_shrinkage, df_f_shrinkage])
df_shrinkage.to_csv(f"{OUT}/coefficient_shrinkage_data.csv", index=False)

print(f"\n✅ Phase 5 Complete! Stepwise mechanisms exported.")
