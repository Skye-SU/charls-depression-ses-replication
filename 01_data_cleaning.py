import pandas as pd
import numpy as np
import pyreadstat
import os
from pathlib import Path

print("=== CHARLS 2011 Replication: Phase 1 — Data Cleaning ===")

# Configure CHARLS raw data path (update this to your local CHARLS directory)
BASE = os.environ.get("CHARLS_DATA", str(Path.home() / "Documents/CHARLS"))
PROJECT_DIR = Path(__file__).resolve().parent
OUT = str(PROJECT_DIR / "data")
os.makedirs(OUT, exist_ok=True)

# 1. Load Datasets
print("Loading CHARLS .dta modules...")
demo, _ = pyreadstat.read_dta(f"{BASE}/demographic_background/demographic_background.dta")
health, _ = pyreadstat.read_dta(f"{BASE}/health_status_and_functioning/health_status_and_functioning.dta")

try:
    psu, _ = pyreadstat.read_dta(f"{BASE}/PSU/psu.dta")
except FileNotFoundError:
    print("  ⚠ PSU module not found, using empty fallback.")
    psu = pd.DataFrame({'communityID': demo['communityID'].unique(), 'urban_nbs': 0})

try:
    biomarker, _ = pyreadstat.read_dta(f"{BASE}/biomarker/biomarker.dta")
except FileNotFoundError:
    print("  ⚠ Biomarker module not found, lower_leg_length will be unavailable.")
    biomarker = pd.DataFrame({'ID': demo['ID']})

try:
    family, _ = pyreadstat.read_dta(f"{BASE}/family_information/family_information.dta")
except FileNotFoundError:
    print("  ⚠ Family module not found, family shock variables will be unavailable.")
    family = pd.DataFrame({'ID': demo['ID']})

print("Merging data on ID...")
drop_health = [c for c in ['rgender', 'householdID', 'communityID'] if c in health.columns]
df = demo.merge(health.drop(columns=drop_health, errors='ignore'), on='ID', how='inner')

def safe_merge(left, right, on_key='ID'):
    if on_key in right.columns:
        drop_cols = [c for c in ['householdID', 'communityID'] if c in right.columns and on_key != 'communityID'] # keep keys
        try:
            return left.merge(right.drop(columns=drop_cols), on=on_key, how='left')
        except:
            return left.merge(right, on=on_key, how='left')
    return left

df = safe_merge(df, biomarker, on_key='ID')
df = safe_merge(df, family, on_key='ID')
df = safe_merge(df, psu, on_key='communityID')

# 3. Sample Restriction (Age >= 45)
df['birth_yr'] = pd.to_numeric(df['ba002_1'], errors='coerce')
df['age'] = 2011 - df['birth_yr']
df['age'] = df['age'].fillna(pd.to_numeric(df['ba004'], errors='coerce'))
initial_n = len(df)
df = df[df['age'] >= 45].copy()

# Advanced: Age Dummies (dropping 45-49 as baseline)
df['age_50_54'] = ((df['age'] >= 50) & (df['age'] <= 54)).astype(float)
df['age_55_59'] = ((df['age'] >= 55) & (df['age'] <= 59)).astype(float)
df['age_60_64'] = ((df['age'] >= 60) & (df['age'] <= 64)).astype(float)
df['age_65_69'] = ((df['age'] >= 65) & (df['age'] <= 69)).astype(float)
df['age_70_74'] = ((df['age'] >= 70) & (df['age'] <= 74)).astype(float)
df['age_75_plus'] = (df['age'] >= 75).astype(float)

print(f"Sample restriction: Kept {len(df)}/ {initial_n} observations (Age >= 45)")

# 4. Construct CES-D 10 Score
print("Constructing CES-D 10 scale...")
cesd_items = [f'dc{i:03d}' for i in range(9, 19)]
neg_items = [c for c in cesd_items if c not in ['dc013', 'dc016']]
pos_items = ['dc013', 'dc016']

for col in neg_items:
    if col in df.columns:
        df[f'{col}_n'] = pd.to_numeric(df[col], errors='coerce') - 1

for col in pos_items:
    if col in df.columns:
        df[f'{col}_n'] = 3 - (pd.to_numeric(df[col], errors='coerce') - 1)

scored_cols = [f'{c}_n' for c in cesd_items if f'{c}_n' in df.columns]
df['cesd_score'] = df[scored_cols].mean(axis=1) * 10
df['cesd_high'] = (df['cesd_score'] >= 10).astype(float)

# 5. Sociodemographics & Advanced SES
print("Constructing SES, Hukou & Splines...")

if 'rgender' in df.columns:
    df['female'] = (pd.to_numeric(df['rgender'], errors='coerce') == 2).astype(float)
elif 'ba000_w2_3' in df.columns:
    df['female'] = (pd.to_numeric(df['ba000_w2_3'], errors='coerce') == 2).astype(float)
else:
    df['female'] = 0.0

if 'bd001' in df.columns:
    df['edu_raw'] = pd.to_numeric(df['bd001'], errors='coerce')
    df['edu_none'] = (df['edu_raw'] <= 2).astype(float)
    df['edu_primary'] = ((df['edu_raw'] == 3) | (df['edu_raw'] == 4)).astype(float)
    df['edu_middle'] = (df['edu_raw'] == 5).astype(float)
    df['edu_high'] = (df['edu_raw'] >= 6).astype(float)

# NEW: CORRECT RURAL MAPPING based on PSU urban_nbs (0=Rural, 1=Urban) -> Target ~51%
if 'urban_nbs' in df.columns:
    df['rural_hukou'] = (pd.to_numeric(df['urban_nbs'], errors='coerce') == 0).astype(float)
else:
    df['rural_hukou'] = 0.0

if 'be001' in df.columns:
    df['marital_raw'] = pd.to_numeric(df['be001'], errors='coerce')
    df['married'] = (df['marital_raw'] == 1).astype(float)
    df['widowed'] = (df['marital_raw'] == 4).astype(float)
    df['never_married'] = (df['marital_raw'] == 6).astype(float)

# TODO: Extract real PCE from CHARLS household_income module.
# The household_income .dta contains consumption expenditure data.
# Steps: (1) Load household_income module, (2) Compute per-capita expenditure,
# (3) Take log, (4) Construct piecewise linear spline at median.
# For now, these columns are set to NaN to clearly mark them as unprocessed.
df['log_pce'] = np.nan
df['log_pce_low'] = np.nan
df['log_pce_high'] = np.nan

# Table 4, 5, 6 Endogenous controls
# Lower leg length from biomarker module (variable qi012)
if 'qi012' in df.columns:
    df['lower_leg_length'] = pd.to_numeric(df['qi012'], errors='coerce')
else:
    # TODO: Ensure biomarker module is loaded to extract lower_leg_length.
    df['lower_leg_length'] = np.nan

# Family shocks
# TODO: Extract child mortality from family_information module.
# Requires identifying the correct variable for "child died in past 2 years".
df['child_died'] = np.nan

if 'da002' in df.columns:
    df['chronic_disease_count'] = pd.to_numeric(df['da002'], errors='coerce').fillna(0)
    df['chronic_disease'] = (df['chronic_disease_count'] > 0).astype(float)
else:
    # TODO: Ensure health module is loaded for chronic disease data.
    df['chronic_disease_count'] = np.nan
    df['chronic_disease'] = np.nan

if 'db001' in df.columns:
    df['adl_limit'] = (pd.to_numeric(df['db001'], errors='coerce') > 1).astype(float)
else:
    # TODO: Ensure health module is loaded for ADL limitation data.
    df['adl_limit'] = np.nan

# Extract countyID from communityID (first 6 characters usually represent county in CHARLS)
df['countyID'] = df['communityID'].astype(str).str[:6]

clean_cols = ['ID', 'householdID', 'communityID', 'countyID', 'age', 'female', 'cesd_score', 'cesd_high', 
              'age_50_54', 'age_55_59', 'age_60_64', 'age_65_69', 'age_70_74', 'age_75_plus',
              'edu_none', 'edu_primary', 'edu_middle', 'edu_high', 
              'rural_hukou', 'married', 'widowed', 'never_married',
              'log_pce', 'log_pce_low', 'log_pce_high', 
              'lower_leg_length', 'child_died', 'chronic_disease_count', 'chronic_disease', 'adl_limit']

for c in cesd_items:
    rescaled_c = f"{c}_n"
    if rescaled_c in df.columns:
        clean_cols.append(rescaled_c) # keep rescaled items for cronbach alpha

clean_cols = list(dict.fromkeys([c for c in clean_cols if c in df.columns]))

out_df = df[clean_cols].copy()
out_file = f"{OUT}/cleaned_charls_phase1.csv"
out_df.to_csv(out_file, index=False)

print(f"\n✅ Advanced Data Cleaning Complete! Output saved to: {out_file}")
print(f"Total N = {len(out_df)}")
print(f"Male Rural Proporion: {out_df[out_df['female']==0]['rural_hukou'].mean()*100:.1f}%")
print(f"Female Rural Proporion: {out_df[out_df['female']==1]['rural_hukou'].mean()*100:.1f}%")
