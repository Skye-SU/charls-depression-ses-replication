# CHARLS 2011 Empirical Replication (Lei et al. 2014)

![CES-D Distribution](output/figure1_cesd_histogram_final.png)

## Overview
This repository contains a **modular academic replication pipeline** for the highly cited paper:
> *Lei, X., Sun, X., Strauss, J., Zhang, P., & Zhao, Y. (2014). Depressive symptoms and SES among the mid-aged and elderly in China: Evidence from the China Health and Retirement Longitudinal Study national baseline. Social Science & Medicine, 120, 224-232.*

Built with an **AI-augmented vibe coding workflow**, this project demonstrates proficiency in:
- Processing large-scale longitudinal health microdata (CHARLS `.dta` modules)
- Constructing psychometric scales (CES-D 10 with reverse scoring)
- Running high-dimensional fixed effects regressions (`linearmodels.PanelOLS`)
- Producing publication-grade visualizations

## Pipeline Architecture

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | `01_data_cleaning.py` | Merges 4 CHARLS modules, filters Age ≥ 45, constructs CES-D 10 + SES variables |
| 2 | `02_descriptive_stats.py` | Cronbach's Alpha, gender-stratified CES-D distributions |
| 3 | `03_regression_analysis.py` | PanelOLS with County/Community fixed effects, robust clustered SE |
| 4 | `04_visualization.py` | CES-D density histograms by gender |
| 5 | `05_coefficient_plot.py` | OLS coefficient forest plot with community-clustered 95% CIs |
| 6 | `06_stepwise_regression.py` | Stepwise addition of biomarker/family/health controls (Tables 4-6) |
| 7 | `07_coefficient_shrinkage_plot.py` | Coefficient attenuation visualization |

## Key Replication Results

| Metric | This Replication | Original Paper | Status |
|--------|-----------------|----------------|--------|
| Male CES-D Mean | **7.46** | ~7.3 | ✅ |
| Female CES-D Mean | **9.46** | ~9.1 | ✅ |
| Education gradient | Monotonic protective | Monotonic protective | ✅ |
| Female coefficient | +1.32*** | Positive, significant | ✅ |
| Rural Hukou | +1.01** | Positive, significant | ✅ |

## Scope & Limitations

This replication focuses on **CES-D scale construction, descriptive statistics, and baseline SES regressions** (Tables 1-3 of the original paper).

The following variables require additional CHARLS modules that are not yet integrated into the pipeline. They are explicitly marked as `TODO` / `NaN` in the code:
- **Per-capita consumption expenditure (PCE)**: Requires `household_income` module
- **Lower leg length**: Requires `biomarker` module
- **Child mortality shock**: Requires `family_information` module

Tables 4-6 (stepwise endogenous controls) are structurally implemented but depend on the above variables for complete results.

## Setup & Execution

```bash
# 1. Install dependencies
pip install pandas numpy pyreadstat statsmodels linearmodels matplotlib seaborn

# 2. Set your CHARLS data path (default: ~/Documents/CHARLS)
export CHARLS_DATA="/path/to/your/CHARLS"

# 3. Run the pipeline sequentially
python 01_data_cleaning.py
python 02_descriptive_stats.py
python 03_regression_analysis.py
python 04_visualization.py
python 05_coefficient_plot.py
```

## Disclaimer
This is an independent empirical exercise built for research portfolio purposes. Original dataset belongs to the China Health and Retirement Longitudinal Study (CHARLS) administered by Peking University.
