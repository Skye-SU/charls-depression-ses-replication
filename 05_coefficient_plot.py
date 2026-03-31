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

controls = ['age', 'age_sq', 'female', 'edu_primary', 'edu_middle', 'edu_high', 'rural_hukou', 'married', 'widowed', 'log_pce', 'chronic_disease']
valid_controls = [c for c in controls if c in df.columns]

formula = "cesd_score ~ " + " + ".join(valid_controls)
model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['communityID']})

# Extract coefficients
params = model.params
conf = model.conf_int()
conf['coef'] = params

vars_to_plot = ['edu_primary', 'edu_middle', 'edu_high', 'log_pce', 'female', 'rural_hukou', 'married']
labels = ['Primary Edu', 'Middle Edu', 'High Edu+', 'Log PCE (Wealth Proxy)', 'Female', 'Rural Hukou', 'Married']

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
ax.set_xlabel('Effect on CES-D 10 Score (OLS Coefficients)', fontsize=12, fontweight='bold', labelpad=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#DDDDDD')
ax.spines['bottom'].set_color('#DDDDDD')

plt.title('Figure 2: Determinants of Geriatric Depression\n(Community-Clustered 95% CIs)', fontsize=14, fontweight='bold', pad=20)
plt.grid(axis='x', linestyle='--', alpha=0.3)

plt.tight_layout()
fig_path = f"{OUT}/figure2_coefficient_plot.png"
plt.savefig(fig_path, dpi=300, facecolor='#F8F9FA')
plt.close()

print(f"✅ Generated Coefficient plot at {fig_path}")
