import pandas as pd
import numpy as np
import os
from pathlib import Path

print("=== Phase 2: Descriptive Statistics & Reliability ===")

PROJECT_DIR = Path(__file__).resolve().parent
INP = str(PROJECT_DIR / "data/cleaned_charls_phase1.csv")
OUT = str(PROJECT_DIR / "output")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(INP)

# Ensure sample validity (Must have valid CES-D score)
df_valid = df.dropna(subset=['cesd_score']).copy()

# ============================================================
# 1. Table 1: Cronbach's Alpha (Lei et al. target: 0.815)
# ============================================================
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
        print(f"  -> High reliability, consistent with Lei et al. (2014) target: 0.815")
else:
    print(f"\n[Table 1] Warning: Missing CES-D items. Found {len(available_items)}/10.")

# ============================================================
# 2. Table 2: CES-D distributions by Age Strata and Gender
#    (Lei et al. Table 2: weighted means and % >= 10)
# ============================================================
def define_age_group(age):
    if 45 <= age <= 49: return '45-49'
    if 50 <= age <= 54: return '50-54'
    if 55 <= age <= 59: return '55-59'
    if 60 <= age <= 64: return '60-64'
    if 65 <= age <= 69: return '65-69'
    if 70 <= age <= 74: return '70-74'
    if age >= 75: return '75+'
    return 'Other'

df_valid['age_group'] = df_valid['age'].apply(define_age_group)
has_weights = 'ind_weight' in df_valid.columns and df_valid['ind_weight'].notna().any()

def summarize_group(group_df):
    if has_weights and 'ind_weight' in group_df.columns:
        w = group_df['ind_weight'].fillna(1)
        w_sum = w.sum()
        mean_score = (group_df['cesd_score'] * w).sum() / w_sum
        high_pct = (group_df['cesd_high'] * w).sum() / w_sum * 100
    else:
        mean_score = group_df['cesd_score'].mean()
        high_pct = group_df['cesd_high'].mean() * 100
    res = {
        'N': len(group_df),
        'Mean_Score': mean_score,
        'Std_Score': group_df['cesd_score'].std(),
        'High_Depression_Pct': high_pct
    }
    return pd.Series(res)

tab2 = df_valid.groupby(['female', 'age_group']).apply(summarize_group).reset_index()
tab2['Gender'] = tab2['female'].map({0.0: 'Male', 1.0: 'Female'})
table2_out = tab2[['Gender', 'age_group', 'N', 'Mean_Score', 'Std_Score', 'High_Depression_Pct']].copy()
table2_out.to_csv(f"{OUT}/table2_cesd_distribution.csv", index=False)

print("\n[Table 2] CES-D Distributions by Gender & Age Group (weighted):")
print(table2_out.to_string(index=False))

# Print paper benchmarks
print("\n  Paper benchmarks: Men Total Mean=7.1, CESD>=10: 30%")
print("                    Women Total Mean=8.9, CESD>=10: 43%")

# ============================================================
# 3. Appendix Table 2: Descriptive Stats by Gender
# ============================================================
desc_vars = [
    'cesd_score', 'age', 'edu_can_read', 'edu_primary', 'edu_junior_high_plus',
    'rural', 'married', 'widowed', 'log_pce',
    'poor_childhood_health', 'lower_leg_length', 'chronic_disease', 'adl_limit'
]
exist_vars = [v for v in desc_vars if v in df_valid.columns]

males = df_valid[df_valid['female'] == 0][exist_vars]
females = df_valid[df_valid['female'] == 1][exist_vars]

tabA2 = pd.DataFrame({
    'Variable': exist_vars,
    'Male_Mean': males.mean().values,
    'Male_SD': males.std().values,
    'Male_N': [males[v].notna().sum() for v in exist_vars],
    'Female_Mean': females.mean().values,
    'Female_SD': females.std().values,
    'Female_N': [females[v].notna().sum() for v in exist_vars]
})
tabA2.to_csv(f"{OUT}/appendix_table2_comparison.csv", index=False)

print("\n[Appendix Table 2] Descriptive Statistics:")
print(tabA2[['Variable', 'Male_Mean', 'Female_Mean']].to_string(index=False, float_format="%.3f"))

print("\n✅ Phase 2 Complete! CSV tables exported.")
