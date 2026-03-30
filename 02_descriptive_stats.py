import pandas as pd
import numpy as np
import os

print("=== Phase 3: Descriptive Statistics & Reliability ===")

INP = "/Users/xiwen/.gemini/antigravity/scratch/intelligence/charls_replication/data/cleaned_charls_phase1.csv"
OUT = "/Users/xiwen/.gemini/antigravity/scratch/intelligence/charls_replication/output"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(INP)

# Ensure sample validity (Must have valid CES-D score)
df_valid = df.dropna(subset=['cesd_score']).copy()

# 1. Table 1: Cronbach's Alpha
def cronbach_alpha(item_df):
    k = item_df.shape[1]
    variances = item_df.var(axis=0, ddof=1)
    total_var = item_df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - variances.sum() / total_var)

cesd_item_cols = [f'dc{i:03d}_n' for i in range(9, 19)]
available_items = [c for c in cesd_item_cols if c in df_valid.columns]

if len(available_items) == 10:
    alpha = cronbach_alpha(df_valid[available_items].dropna())
    print(f"\n[Table 1] Cronbach's Alpha (CES-D 10): {alpha:.4f}")
    if alpha > 0.75:
        print("  -> High reliability, validating scoring logic (Lei 2014 target: 0.815)")
else:
    print(f"\n[Table 1] Warning: Missing CES-D items. Found {len(available_items)}/10.")

# 2. Table 2: CES-D distributions by Age Strata and Gender
def define_age_group(age):
    if 45 <= age <= 54: return '45-54'
    if 55 <= age <= 64: return '55-64'
    if 65 <= age <= 74: return '65-74'
    if age >= 75: return '75+'
    return 'Other'

df_valid['age_group'] = df_valid['age'].apply(define_age_group)

def summarize_group(group_df):
    res = {
        'N': len(group_df),
        'Mean_Score': group_df['cesd_score'].mean(),
        'Std_Score': group_df['cesd_score'].std(),
        'High_Depression_Pct': group_df['cesd_high'].mean() * 100
    }
    return pd.Series(res)

tab2 = df_valid.groupby(['female', 'age_group']).apply(summarize_group).reset_index()
tab2['Gender'] = tab2['female'].map({0.0: 'Male', 1.0: 'Female'})
table2_out = tab2[['Gender', 'age_group', 'N', 'Mean_Score', 'Std_Score', 'High_Depression_Pct']].copy()
table2_out.to_csv(f"{OUT}/table2_cesd_distribution.csv", index=False)

print("\n[Table 2] CES-D Distributions by Gender & Age Group:")
print(table2_out.to_string(index=False))

# 3. Appendix Table 2: Describe Stats
desc_vars = [
    'cesd_score', 'female', 'age', 'edu_none', 'edu_primary', 'edu_middle', 'edu_high',
    'rural_hukou', 'married', 'widowed', 'log_pce', 'chronic_disease'
]
exist_vars = [v for v in desc_vars if v in df_valid.columns]

males = df_valid[df_valid['female'] == 0][exist_vars]
females = df_valid[df_valid['female'] == 1][exist_vars]

tabA2 = pd.DataFrame({
    'Variable': exist_vars,
    'Male_Mean': males.mean().values,
    'Male_SD': males.std().values,
    'Female_Mean': females.mean().values,
    'Female_SD': females.std().values
})
tabA2.to_csv(f"{OUT}/appendix_table2_comparison.csv", index=False)

print("\n[Appendix Table 2] Descriptive Statistics (Anchors):")
print(tabA2[['Variable', 'Male_Mean', 'Female_Mean']].to_string(index=False, float_format="%.3f"))

print("\n✅ Phase 3 Complete! CSV tables exported.")
