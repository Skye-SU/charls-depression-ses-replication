import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import seaborn as sns

matplotlib.use('Agg')
print("=== Phase 5: Visualizations ===")

INP = "/Users/xiwen/.gemini/antigravity/scratch/intelligence/charls_replication/data/cleaned_charls_phase1.csv"
OUT = "/Users/xiwen/.gemini/antigravity/scratch/intelligence/charls_replication/output"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(INP).dropna(subset=['cesd_score', 'female'])

# Figure 1: CES-D Density Distribution by Gender
male_scores = df[df['female'] == 0]['cesd_score']
female_scores = df[df['female'] == 1]['cesd_score']

fig, ax1 = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor('#F8F9FA')
ax1.set_facecolor('#FFFFFF')

bins = range(0, 32)
ax1.hist(male_scores, bins=bins, alpha=0.6, color='#2B4666', label=f'Men (n={len(male_scores):,})', density=True, edgecolor='white')
ax1.hist(female_scores, bins=bins, alpha=0.55, color='#A94044', label=f'Women (n={len(female_scores):,})', density=True, edgecolor='white')

ax1.axvline(x=10, color='#333333', linestyle='--', linewidth=1.5, label='Cut-off (CES-D >= 10)')

ax1.set_xlabel('CES-D 10 Score', fontsize=12, fontweight='500', labelpad=10)
ax1.set_ylabel('Density', fontsize=12, fontweight='500', labelpad=10)
ax1.set_title('Figure 1: CES-D Score Distribution by Gender\n(Replication of Lei et al. 2014)', fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=10, loc='upper right', frameon=True, facecolor='white', edgecolor='#EEEEEE')

# Remove top/right spines
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#DDDDDD')
ax1.spines['bottom'].set_color('#DDDDDD')
ax1.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
fig_path = f"{OUT}/figure1_cesd_histogram_final.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()

print(f"✅ Phase 5 Complete! Visualization saved to {fig_path}")
