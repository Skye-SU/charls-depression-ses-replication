import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
import matplotlib.pyplot as plt
import os
from pathlib import Path

matplotlib.use('Agg')
print("=== Phase 5: Coefficient Plot ===")

PROJECT_DIR = Path(__file__).resolve().parent
INP = str(PROJECT_DIR / "data/cleaned_charls_phase1.csv")
OUT = str(PROJECT_DIR / "output")

df = pd.read_csv(INP).dropna(subset=['cesd_score', 'female', 'communityID', 'age']).copy()

# Use age dummies + education + PCE + demographic controls (matching Table 3)
controls = [
    'age_50_54', 'age_55_59', 'age_60_64', 'age_65_69', 'age_70_74', 'age_75_plus',
    'female', 'edu_can_read', 'edu_primary', 'edu_junior_high_plus',
    'rural', 'married', 'widowed',
    'log_pce_low', 'log_pce_high'
]
valid_controls = [c for c in controls if c in df.columns and df[c].notna().any()]

formula = "cesd_score ~ " + " + ".join(valid_controls)
df_reg = df.dropna(subset=valid_controls + ['cesd_score']).copy()
model = smf.ols(formula, data=df_reg).fit(cov_type='cluster', cov_kwds={'groups': df_reg['communityID']})

# Extract coefficients for key SES variables
params = model.params
conf = model.conf_int()
conf['coef'] = params

# Variables to plot (Lei et al. key findings)
vars_to_plot = [
    'edu_can_read', 'edu_primary', 'edu_junior_high_plus',
    'log_pce_low', 'female', 'rural', 'married'
]
labels = [
    'Can Read/Write', 'Finished Primary', 'Junior High+',
    'Log PCE (below median)', 'Female', 'Rural', 'Married'
]

plot_data = []
for var, lab in zip(vars_to_plot, labels):
    if var in conf.index:
        plot_data.append({
            'label': lab,
            'coef': conf.loc[var, 'coef'],
            'lower': conf.loc[var, 0],
            'upper': conf.loc[var, 1]
        })

df_plot = pd.DataFrame(plot_data).sort_values('coef', ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#F8F9FA')
ax.set_facecolor('#FFFFFF')

# Plot error bars and points
for i, row in enumerate(df_plot.itertuples()):
    color = '#A94044' if row.coef > 0 else '#2B4666'
    ax.errorbar(row.coef, i, xerr=[[row.coef - row.lower], [row.upper - row.coef]], 
                fmt='o', color=color, markersize=8, capsize=5, capthick=2, elinewidth=2)

ax.axvline(0, color='black', linestyle='--', linewidth=1)
ax.set_yticks(range(len(df_plot)))
ax.set_yticklabels(df_plot['label'], fontsize=11, fontweight='500', color='#333')
ax.set_xlabel('Effect on CES-D 10 Score (OLS Coefficients, 95% CI)', fontsize=12, fontweight='bold', labelpad=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#DDDDDD')
ax.spines['bottom'].set_color('#DDDDDD')

plt.title('Figure 2: SES Determinants of Geriatric Depression\n(Community-Clustered 95% CIs, Lei et al. 2014 Replication)', fontsize=13, fontweight='bold', pad=20)
plt.grid(axis='x', linestyle='--', alpha=0.3)

plt.tight_layout()
fig_path = f"{OUT}/figure2_coefficient_plot.png"
plt.savefig(fig_path, dpi=300, facecolor='#F8F9FA')
plt.close()

print(f"✅ Phase 5 Complete! Coefficient plot saved to {fig_path}")
