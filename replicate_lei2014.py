"""
Replication Script: Lei et al. (2014)
"Depressive Symptoms and SES among the Mid-Aged and Elderly in China"
Social Science & Medicine

Data: CHARLS 2011 National Baseline
Output: Table 1 (Summary Statistics) + Table 2 (OLS/Logit Regression)
"""

import pandas as pd
import numpy as np
import pyreadstat
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 0. Paths
# ==============================================================================
BASE = "/Users/xiwen/Documents/Career_Job/大厂简历/RA投递材料包/定量统计/CHARLS"
OUT  = "/Users/xiwen/.gemini/antigravity/scratch/intelligence/charls_replication/output"

import os; os.makedirs(OUT, exist_ok=True)

# ==============================================================================
# 1. Load Data
# ==============================================================================
print("Loading data files...")
demo, _    = pyreadstat.read_dta(f"{BASE}/demographic_background/demographic_background.dta")
health, _  = pyreadstat.read_dta(f"{BASE}/health_status_and_functioning/health_status_and_functioning.dta")
income, _  = pyreadstat.read_dta(f"{BASE}/household_income/household_income.dta")
family, _  = pyreadstat.read_dta(f"{BASE}/family_information/family_information.dta")
weight, _  = pyreadstat.read_dta(f"{BASE}/weight/weight.dta")
psu, _     = pyreadstat.read_dta(f"{BASE}/PSU/PSU.dta")

print(f"  demo:   {len(demo)} rows, cols: {list(demo.columns[:10])}")
print(f"  health: {len(health)} rows, cols: {list(health.columns[:10])}")
print(f"  income: {len(income)} rows, cols: {list(income.columns[:10])}")
print(f"  family: {len(family)} rows, cols: {list(family.columns[:10])}")
print(f"  weight: {len(weight)} rows, cols: {list(weight.columns[:10])}")
print()

# ==============================================================================
# 2. Build CESD-10 Depression Score (Dependent Variable)
# ==============================================================================
print("Building CES-D 10 depression score...")

# CES-D 10 items in CHARLS: dc002-dc012 (10 items, skipping dc007 if not in 10-item version)
# Positive items (reverse coded): dc006 (felt happy), dc010 (enjoyed life)
# Negative items (direct): dc002, dc003, dc004, dc005, dc008, dc009, dc011, dc012

# Find CESD columns
cesd_cols = [c for c in health.columns if c.startswith('dc00') or c.startswith('dc01')]
print(f"  CES-D candidate columns: {cesd_cols}")

# Standard CHARLS CESD10 variable names
# 1=Rarely (<1 day), 2=Some days (1-2 days), 3=Occasionally (3-4 days), 4=Most of the time (5-7 days)
# Score: 1->0, 2->1, 3->2, 4->3 for negative items; reversed for positive items

negative_items = ['dc002', 'dc003', 'dc004', 'dc005', 'dc008', 'dc009', 'dc011', 'dc012']
positive_items = ['dc006', 'dc010']

# Check which columns exist
neg_exist = [c for c in negative_items if c in health.columns]
pos_exist = [c for c in positive_items if c in health.columns]
print(f"  Negative items found: {neg_exist}")
print(f"  Positive items found: {pos_exist}")

h = health.copy()

# Recode: 1->0, 2->1, 3->2, 4->3
def recode_neg(x):
    mapping = {1: 0, 2: 1, 3: 2, 4: 3}
    return pd.to_numeric(x, errors='coerce').map(mapping)

def recode_pos(x):
    mapping = {1: 3, 2: 2, 3: 1, 4: 0}  # reversed
    return pd.to_numeric(x, errors='coerce').map(mapping)

for col in neg_exist:
    h[f'{col}_r'] = recode_neg(h[col])
for col in pos_exist:
    h[f'{col}_r'] = recode_pos(h[col])

scored_cols = [f'{col}_r' for col in neg_exist + pos_exist]
h['cesd_score'] = h[scored_cols].sum(axis=1, min_count=len(scored_cols))
h['cesd_high'] = (h['cesd_score'] >= 10).astype(float)  # binary: score >= 10

print(f"  CES-D score range: {h['cesd_score'].min():.0f} - {h['cesd_score'].max():.0f}")
print(f"  CES-D high (>=10): {h['cesd_high'].mean()*100:.1f}%")

# Also calculate mean depression rate: score >= 10 → "High" depressive symptoms
# The paper reports ~30% men and ~43% women

# ==============================================================================
# 3. Clean Demographic Variables (Independent Variables / Controls)
# ==============================================================================
print("\nCleaning demographic variables...")
d = demo.copy()

# Gender: ba000_w2_4 or rgender — check
gender_col = None
for c in ['ba000_w2_4', 'rgender', 'ba000_w2', 'gender']:
    if c in d.columns:
        gender_col = c; break
print(f"  Gender column: {gender_col}")

# Age: ba004_w2_4 or rage
age_col = None
for c in ['ba004_w2_4', 'rage', 'ba004_w2', 'age']:
    if c in d.columns:
        age_col = c; break
print(f"  Age column: {age_col}")

# Education: bd001_w2_4 or reduc
edu_col = None
for c in ['bd001_w2_4', 'reduc', 'bd001_w2', 'edu']:
    if c in d.columns:
        edu_col = c; break
print(f"  Education column: {edu_col}")

# Hukou (urban/rural registration)
hukou_col = None
for c in ['bd002_w2_4', 'rhukou', 'bd002_w2', 'hukou']:
    if c in d.columns:
        hukou_col = c; break
print(f"  Hukou column: {hukou_col}")

# Marital status
marital_col = None
for c in ['be001_w2_4', 'rmarital', 'be001_w2', 'marital']:
    if c in d.columns:
        marital_col = c; break
print(f"  Marital column: {marital_col}")

# Print column listing to understand structure
print("\n  All demographic columns:")
print(list(d.columns))

# ==============================================================================
# 4. Merge Datasets on householdID + personID
# ==============================================================================
print("\nMerging datasets...")

# Find ID columns
id_cols_demo = [c for c in d.columns if 'ID' in c.upper() or 'id' in c.lower()]
id_cols_health = [c for c in health.columns if 'ID' in c.upper() or 'id' in c.lower()]
print(f"  ID cols in demo: {id_cols_demo[:5]}")
print(f"  ID cols in health: {id_cols_health[:5]}")

# Determine merge key
# Usually: householdID + personID, or combined 'ID'
merge_key = None
for key in ['ID', 'id', 'personID', 'person_id']:
    if key in d.columns and key in h.columns:
        merge_key = key; break

if not merge_key:
    # Try combined household+person
    for hkey in ['hhid', 'householdID', 'HHID']:
        for pkey in ['pn', 'personID', 'PN']:
            if hkey in d.columns and pkey in d.columns:
                d['_merge_key'] = d[hkey].astype(str) + '_' + d[pkey].astype(str)
                if hkey in h.columns and pkey in h.columns:
                    h['_merge_key'] = h[hkey].astype(str) + '_' + h[pkey].astype(str)
                    merge_key = '_merge_key'
                    break
        if merge_key: break

print(f"  Merge key: {merge_key}")

# Save column info to understand data structure
col_info = {
    'demographic_cols': list(d.columns),
    'health_cols': list(health.columns),
    'income_cols': list(income.columns),
    'family_cols': list(family.columns),
    'weight_cols': list(weight.columns),
}
import json
with open(f"{OUT}/column_info.json", 'w') as f:
    json.dump(col_info, f, indent=2)

print(f"\n  Column info saved to {OUT}/column_info.json")
print("\n=== STEP 1 COMPLETE: Data loaded and inspected ===")
print("Please check output/column_info.json to identify exact variable names.")
print("Run phase 2 after reviewing column names.")
