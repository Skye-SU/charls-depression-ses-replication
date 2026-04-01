import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

print("=== Phase 7: Coefficient Forest Plot Generation ===")

PROJECT_DIR = Path(__file__).resolve().parent
INP = str(PROJECT_DIR / "output/coefficient_shrinkage_data.csv")
OUT = str(PROJECT_DIR / "output")

df = pd.read_csv(INP)

# Vibe Coding Visualization setup
plt.style.use('dark_background')
sns.set_theme(style='darkgrid', palette='deep')

# Custom palette for Models
model_colors = {
    'Model 1 (Base)': '#4B9CD3',        # Base
    'Model 2 (+Childhood)': '#FFB81C',  # Step 1
    'Model 3 (+Family)': '#E87722',     # Step 2
    'Model 4 (+Health)': '#E85B5B'      # Step 3
}

fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='#1E1E1E')
fig.suptitle('Attenuation of SES Effects on Log CES-D Score\n(Stepwise Addition of Health & Family Shocks)', fontsize=18, color='white', y=0.98)

for ax, gender in zip(axes, ['Male', 'Female']):
    data = df[df['Gender'] == gender]
    
    # We plot the coefficient of 'edu_middle' as the demonstration since 'edu_high' has large variance.
    # Alternatively, we can plot 'log_pce_low'. Let's plot both side-by-side or just focus on edu_middle and log_pce_low.
    # For a clean portfolio plot, let's track the shrinkage of Junior High Education (edu_middle)
    
    y_pos = np.arange(len(data))
    coefs = data['edu_primary_coef'].values
    errors = 1.96 * data['edu_primary_se'].values
    
    ax.set_facecolor('#1E1E1E')
    ax.grid(color='#333333', linestyle='--', linewidth=0.5)
    
    ax.axvline(x=0, color='#666666', linestyle='-', linewidth=2, zorder=1)
    
    for i in range(len(data)):
        model_name = data.iloc[i]['Model']
        c = model_colors[model_name]
        
        # Error bar
        ax.errorbar(coefs[i], len(data)-1-i, xerr=errors[i], fmt='o', color=c, 
                    elinewidth=3, capsize=8, capthick=3, markersize=14, zorder=3)
        
        # Shrinkage arrow connecting previous to current (if i > 0)
        if i > 0:
            prev_coef = coefs[i-1]
            ax.annotate('', xy=(coefs[i], len(data)-1-i), xytext=(prev_coef, len(data)-i),
                        arrowprops=dict(arrowstyle="->", color='#888888', lw=2, alpha=0.6, ls='dotted'), zorder=2)
            
    ax.set_yticks(np.arange(len(data)))
    # Reverse so Base is at the top
    ax.set_yticklabels(data['Model'].values[::-1], color='white', fontsize=12)
    ax.set_title(f'{gender} Sample: Effect of Primary Education', fontsize=14, color='white', pad=15)
    
    ax.set_xlabel('Coefficient on Finished Primary Education\n(Negative = Protective Effect against Depression)', color='white')
    ax.tick_params(colors='white')

    # Spines
    for spine in ax.spines.values():
        spine.set_color('#444444')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plot_path = f"{OUT}/figure2_stepwise_shrinkage_forest_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#1E1E1E')
print(f"✅ Vibe Visualization Saved: {plot_path}")
