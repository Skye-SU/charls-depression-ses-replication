import pandas as pd
import numpy as np
import pyreadstat
import os
from pathlib import Path

print("=== CHARLS 2011 Replication: Phase 1 — Data Cleaning ===")
print("Target: Lei et al. (2014) Social Science & Medicine\n")

# Configure CHARLS raw data path (update this to your local CHARLS directory)
BASE = os.environ.get("CHARLS_DATA", str(Path.home() / "Documents/Career_Job/Work/Materials/Quantitative/CHARLS"))
PROJECT_DIR = Path(__file__).resolve().parent
OUT = str(PROJECT_DIR / "data")
os.makedirs(OUT, exist_ok=True)

# ============================================================
# 1. Load All Required CHARLS Modules
# ============================================================
print("Loading CHARLS .dta modules...")
demo, _ = pyreadstat.read_dta(f"{BASE}/demographic_background/demographic_background.dta")
health, _ = pyreadstat.read_dta(f"{BASE}/health_status_and_functioning/health_status_and_functioning.dta")

# Sample weights (Lei et al. p.282: "weighted using individual sample weights")
try:
    weight, _ = pyreadstat.read_dta(f"{BASE}/weight/weight.dta")
    print("  ✅ Weight module loaded")
except FileNotFoundError:
    print("  ⚠ Weight module not found, unweighted analysis will proceed.")
    weight = pd.DataFrame({'ID': demo['ID'], 'ind_weight': 1.0})

try:
    psu, _ = pyreadstat.read_dta(f"{BASE}/PSU/psu.dta")
    print("  ✅ PSU module loaded")
except FileNotFoundError:
    print("  ⚠ PSU module not found, using empty fallback.")
    psu = pd.DataFrame({'communityID': demo['communityID'].unique(), 'urban_nbs': 0})

try:
    biomarker, _ = pyreadstat.read_dta(f"{BASE}/biomarker/biomarker.dta")
    print("  ✅ Biomarker module loaded")
except Exception:
    print("  ⚠ Biomarker module not found, lower_leg_length will be unavailable.")
    biomarker = pd.DataFrame({'ID': demo['ID']})

try:
    family, _ = pyreadstat.read_dta(f"{BASE}/family_information/family_information.dta")
    print("  ✅ Family module loaded")
except Exception:
    print("  ⚠ Family module not found, family shock variables will be unavailable.")
    family = pd.DataFrame({'ID': demo['ID']})

# Household income module for PCE construction
try:
    hhincome, _ = pyreadstat.read_dta(f"{BASE}/household_income/household_income.dta")
    print("  ✅ Household income module loaded")
except Exception:
    print("  ⚠ Household income module not found, PCE will be unavailable.")
    hhincome = None

# ============================================================
# 2. Merge All Modules
# ============================================================
print("\nMerging data modules...")

def safe_merge(left, right, on_key='ID'):
    """Merge avoiding duplicate columns."""
    if right is None or on_key not in right.columns:
        return left
    drop_cols = [c for c in right.columns if c in left.columns and c != on_key]
    return left.merge(right.drop(columns=drop_cols, errors='ignore'), on=on_key, how='left')

# Core merge: demo + health
drop_health = [c for c in ['rgender', 'householdID', 'communityID'] if c in health.columns]
df = demo.merge(health.drop(columns=drop_health, errors='ignore'), on='ID', how='inner')

# Add weight (merge on ID)
if 'ind_weight' in weight.columns:
    df = safe_merge(df, weight[['ID', 'ind_weight']], on_key='ID')

# Add biomarker, family, PSU
df = safe_merge(df, biomarker, on_key='ID')
df = safe_merge(df, family, on_key='ID')
df = safe_merge(df, psu, on_key='communityID')

# ============================================================
# 3. Sample Restriction (Age >= 45, per Lei et al.)
# ============================================================
df['birth_yr'] = pd.to_numeric(df['ba002_1'], errors='coerce')
df['age'] = 2011 - df['birth_yr']
df['age'] = df['age'].fillna(pd.to_numeric(df['ba004'], errors='coerce'))
initial_n = len(df)
df = df[df['age'] >= 45].copy()

# Age dummies (dropping 45-49 as baseline, per Lei et al. Table 3)
df['age_50_54'] = ((df['age'] >= 50) & (df['age'] <= 54)).astype(float)
df['age_55_59'] = ((df['age'] >= 55) & (df['age'] <= 59)).astype(float)
df['age_60_64'] = ((df['age'] >= 60) & (df['age'] <= 64)).astype(float)
df['age_65_69'] = ((df['age'] >= 65) & (df['age'] <= 69)).astype(float)
df['age_70_74'] = ((df['age'] >= 70) & (df['age'] <= 74)).astype(float)
df['age_75_plus'] = (df['age'] >= 75).astype(float)

print(f"Sample restriction: Kept {len(df)}/{initial_n} observations (Age >= 45)")

# ============================================================
# 4. Construct CES-D 10 Score (Radloff 1977 scoring)
# ============================================================
print("Constructing CES-D 10 scale...")
cesd_items = [f'dc{i:03d}' for i in range(9, 19)]
neg_items = [c for c in cesd_items if c not in ['dc013', 'dc016']]
pos_items = ['dc013', 'dc016']  # Hopeful, Happy (reverse scored)

# Negative items: 1->0, 2->1, 3->2, 4->3
for col in neg_items:
    if col in df.columns:
        df[f'{col}_n'] = pd.to_numeric(df[col], errors='coerce') - 1

# Positive items (reverse): 1->3, 2->2, 3->1, 4->0
for col in pos_items:
    if col in df.columns:
        df[f'{col}_n'] = 3 - (pd.to_numeric(df[col], errors='coerce') - 1)

scored_cols = [f'{c}_n' for c in cesd_items if f'{c}_n' in df.columns]
df['cesd_score'] = df[scored_cols].mean(axis=1) * 10
df['cesd_high'] = (df['cesd_score'] >= 10).astype(float)

# ============================================================
# 5. Gender
# ============================================================
if 'rgender' in df.columns:
    df['female'] = (pd.to_numeric(df['rgender'], errors='coerce') == 2).astype(float)
elif 'ba000_w2_3' in df.columns:
    df['female'] = (pd.to_numeric(df['ba000_w2_3'], errors='coerce') == 2).astype(float)
else:
    df['female'] = 0.0

# ============================================================
# 6. Education (4-category, per Lei et al. Table 3)
#    bd001: 1=No formal education, 2=Can read/literacy class, 3=Sishu/home school
#           4=Elementary school, 5=Middle school, 6=High school, 7+=Higher
#    Paper categories:
#      - Illiterate (bd001==1): OMITTED BASELINE
#      - Can read and write (bd001 in [2,3]): some primary but not completed
#      - Finished primary (bd001==4)
#      - Junior high and above (bd001>=5)
# ============================================================
print("Constructing education dummies (4-category, Lei et al. specification)...")
if 'bd001' in df.columns:
    df['edu_raw'] = pd.to_numeric(df['bd001'], errors='coerce')
    df['edu_can_read']        = (df['edu_raw'].isin([2, 3])).astype(float)
    df['edu_primary']         = (df['edu_raw'] == 4).astype(float)
    df['edu_junior_high_plus'] = (df['edu_raw'] >= 5).astype(float)
else:
    df['edu_can_read'] = 0.0
    df['edu_primary'] = 0.0
    df['edu_junior_high_plus'] = 0.0

# ============================================================
# 7. Rural Residence (from PSU module)
#    Lei et al. Appendix Table 2: Rural ~ 51% men, 49% women
# ============================================================
if 'urban_nbs' in df.columns:
    df['rural'] = (pd.to_numeric(df['urban_nbs'], errors='coerce') == 0).astype(float)
else:
    df['rural'] = 0.0

# ============================================================
# 8. Marital Status
# ============================================================
if 'be001' in df.columns:
    df['marital_raw'] = pd.to_numeric(df['be001'], errors='coerce')
    df['married'] = (df['marital_raw'] == 1).astype(float)
    df['widowed'] = (df['marital_raw'] == 4).astype(float)
    df['never_married'] = (df['marital_raw'] == 6).astype(float)

# ============================================================
# 9. Per Capita Expenditure (PCE) with Piecewise Linear Spline
#    Lei et al. p.312-313: "log of household per capita expenditure"
#    "we use a linear spline around the median log pce"
# ============================================================
print("Constructing log PCE with piecewise spline...")
if hhincome is not None:
    hi = hhincome.copy()
    
    # Annualize weekly food expenditure (*52)
    weekly_cols = ['ge006', 'ge007', 'ge008']
    for c in weekly_cols:
        if c in hi.columns:
            hi[c] = pd.to_numeric(hi[c], errors='coerce').clip(lower=0)
    hi['annual_food'] = hi[[c for c in weekly_cols if c in hi.columns]].sum(axis=1, min_count=1) * 52
    
    # Annualize monthly non-food expenditure (*12)
    monthly_cols = [f'ge009_{i}' for i in range(1, 8)]
    exist_monthly = [c for c in monthly_cols if c in hi.columns]
    for c in exist_monthly:
        hi[c] = pd.to_numeric(hi[c], errors='coerce').clip(lower=0)
    hi['annual_monthly'] = hi[exist_monthly].sum(axis=1, min_count=1) * 12
    
    # Annual expenditure items
    annual_cols = [f'ge010_{i}' for i in range(1, 15)]
    exist_annual = [c for c in annual_cols if c in hi.columns]
    for c in exist_annual:
        hi[c] = pd.to_numeric(hi[c], errors='coerce').clip(lower=0)
    hi['annual_other'] = hi[exist_annual].sum(axis=1, min_count=1)
    
    # Total household expenditure
    hi['total_expenditure'] = hi[['annual_food', 'annual_monthly', 'annual_other']].sum(axis=1, min_count=1)
    
    # Household size (ge004 = number of people eating together)
    hi['hh_size'] = pd.to_numeric(hi['ge004'], errors='coerce').clip(lower=1)
    
    # Per capita expenditure
    hi['pce'] = hi['total_expenditure'] / hi['hh_size']
    hi['log_pce'] = np.log(hi['pce'].clip(lower=1))
    hi.loc[hi['pce'] <= 0, 'log_pce'] = np.nan
    
    # Merge PCE to individual-level data via householdID
    df = df.merge(hi[['householdID', 'log_pce']].dropna(), on='householdID', how='left')
    
    # Piecewise linear spline at median (Lei et al. specification)
    median_log_pce = df['log_pce'].median()
    df['log_pce_low']  = np.minimum(df['log_pce'], median_log_pce)
    df['log_pce_high'] = np.maximum(df['log_pce'] - median_log_pce, 0)
    
    print(f"  PCE: N={df['log_pce'].notna().sum()}, Mean={df['log_pce'].mean():.2f}, Median={median_log_pce:.2f}")
else:
    df['log_pce'] = np.nan
    df['log_pce_low'] = np.nan
    df['log_pce_high'] = np.nan
    print("  ⚠ PCE unavailable (household_income module missing)")

# ============================================================
# 10. Table 4-6 Endogenous Controls
# ============================================================
print("Constructing endogenous controls...")

# Lower leg length (biomarker qi012) - proxy for childhood nutrition
if 'qi012' in df.columns:
    df['lower_leg_length'] = pd.to_numeric(df['qi012'], errors='coerce')
else:
    df['lower_leg_length'] = np.nan

# Poor childhood health (ba010: self-rated childhood health before 16, 5=poor)
if 'ba010' in df.columns:
    df['poor_childhood_health'] = (pd.to_numeric(df['ba010'], errors='coerce') == 5).astype(float)
else:
    df['poor_childhood_health'] = np.nan

# Family shocks: child died in past 2 years
# From family_information module - variable may vary
df['child_died'] = np.nan  # TODO: Identify exact variable in family_information

# Recent widowed (derived from marital status change - proxied by current widowed status)
# In cross-sectional baseline, we approximate using be002 (year became widowed)

# Chronic disease (da002: number of chronic conditions diagnosed)
if 'da002' in df.columns:
    df['chronic_disease_count'] = pd.to_numeric(df['da002'], errors='coerce').fillna(0)
    df['chronic_disease'] = (df['chronic_disease_count'] > 0).astype(float)
else:
    df['chronic_disease_count'] = np.nan
    df['chronic_disease'] = np.nan

# ADL/IADL disability
if 'db001' in df.columns:
    df['adl_limit'] = (pd.to_numeric(df['db001'], errors='coerce') > 1).astype(float)
else:
    df['adl_limit'] = np.nan

# ============================================================
# 11. Geographic IDs
# ============================================================
# Extract countyID from communityID (first 6 chars in CHARLS coding)
df['countyID'] = df['communityID'].astype(str).str[:6]

# ============================================================
# 12. Export Clean Dataset
# ============================================================
clean_cols = [
    'ID', 'householdID', 'communityID', 'countyID',
    'age', 'female', 'ind_weight',
    'cesd_score', 'cesd_high',
    'age_50_54', 'age_55_59', 'age_60_64', 'age_65_69', 'age_70_74', 'age_75_plus',
    'edu_can_read', 'edu_primary', 'edu_junior_high_plus',
    'rural', 'married', 'widowed', 'never_married',
    'log_pce', 'log_pce_low', 'log_pce_high',
    'lower_leg_length', 'poor_childhood_health', 'child_died',
    'chronic_disease_count', 'chronic_disease', 'adl_limit'
]

# Add CES-D item scores for Cronbach's alpha
for c in cesd_items:
    rescaled_c = f"{c}_n"
    if rescaled_c in df.columns:
        clean_cols.append(rescaled_c)

clean_cols = list(dict.fromkeys([c for c in clean_cols if c in df.columns]))

out_df = df[clean_cols].copy()
out_file = f"{OUT}/cleaned_charls_phase1.csv"
out_df.to_csv(out_file, index=False)

print(f"\n✅ Phase 1 Complete! Output saved to: {out_file}")
print(f"Total N = {len(out_df)}")
print(f"Appendix Table 2 Anchors (Paper: Men=0.51, Women=0.49 rural):")
print(f"  Male Rural:   {out_df[out_df['female']==0]['rural'].mean()*100:.1f}%")
print(f"  Female Rural: {out_df[out_df['female']==1]['rural'].mean()*100:.1f}%")
if 'log_pce' in out_df.columns:
    m_pce = out_df[out_df['female']==0]['log_pce'].mean()
    f_pce = out_df[out_df['female']==1]['log_pce'].mean()
    print(f"  Male logPCE mean:   {m_pce:.2f} (Paper: 9.07)")
    print(f"  Female logPCE mean: {f_pce:.2f} (Paper: 9.11)")
if 'ind_weight' in out_df.columns:
    print(f"  Sample weights: Available ({out_df['ind_weight'].notna().sum()} non-null)")
