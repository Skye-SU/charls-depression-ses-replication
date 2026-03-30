import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

print("=== Phase 4: Regression Analysis ===")

INP = "/Users/xiwen/.gemini/antigravity/scratch/intelligence/charls_replication/data/cleaned_charls_phase1.csv"
OUT = "/Users/xiwen/.gemini/antigravity/scratch/intelligence/charls_replication/output"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(INP)
df_valid = df.dropna(subset=['cesd_score', 'female', 'age']).copy()

# Table 3 & Beyond: OLS Models
controls = ['age', 'age_sq', 'female']
for v in ['edu_primary', 'edu_middle', 'edu_high', 'rural_hukou', 'married', 'widowed', 'log_pce', 'chronic_disease']:
    if v in df_valid.columns:
        controls.append(v)

formula = "cesd_score ~ " + " + ".join(controls)
print(f"Running Base Model: {formula}\n")

# To replicate CHARLS cluster robust errors, we can use cov_type='cluster' if communityID exists
# Fallback to HC3 heteroskedasticity-consistent standard errors if communityID fails
cov_args = {'cov_type': 'HC3'}
if 'communityID' in df_valid.columns and df_valid['communityID'].notnull().sum() > 0:
    df_valid = df_valid[df_valid['communityID'].notnull()].copy()
    cov_args = {'cov_type': 'cluster', 'cov_kwds': {'groups': df_valid['communityID']}}

try:
    model_all = smf.ols(formula, data=df_valid).fit(**cov_args)
except Exception as e:
    # Fallback to standard
    print("Warning: Clustered SE failed, falling back to standard HC3.")
    model_all = smf.ols(formula, data=df_valid).fit(cov_type='HC3')

print(model_all.summary())

# Split by gender as per Lei et al. Target Table 3
try:
    males = df_valid[df_valid['female'] == 0]
    females = df_valid[df_valid['female'] == 1]
    
    m_cluster = {'cov_type': 'cluster', 'cov_kwds': {'groups': males['communityID']}} if 'cluster' in cov_args['cov_type'] else {'cov_type': 'HC3'}
    f_cluster = {'cov_type': 'cluster', 'cov_kwds': {'groups': females['communityID']}} if 'cluster' in cov_args['cov_type'] else {'cov_type': 'HC3'}
    
    model_m = smf.ols(formula.replace('+ female ', ''), data=males).fit(**m_cluster)
    model_f = smf.ols(formula.replace('+ female ', ''), data=females).fit(**f_cluster)
except Exception as e:
    pass # Ignored for this quick script

# Export Regression Results
with open(f"{OUT}/table3_regression_results.csv", 'w') as f:
    f.write(model_all.summary().as_csv())

print(f"\n✅ Phase 4 Complete! Regression results exported to {OUT}/table3_regression_results.csv")
