# CHARLS 2011 Empirical Replication (Lei et al. 2014)

![CES-D Distribution](output/figure1_cesd_histogram_final.png)

## Overview
This repository contains a **modular academic replication pipeline** for the highly cited paper:
> *Lei, X., Sun, X., Strauss, J., Zhang, P., & Zhao, Y. (2014). Depressive symptoms and SES among the mid-aged and elderly in China: Evidence from the China Health and Retirement Longitudinal Study national baseline. Social Science & Medicine, 120, 224-232.*

Built with an **AI-augmented vibe coding workflow**, this project demonstrates proficiency in:
- Processing large-scale longitudinal health microdata (6 CHARLS `.dta` modules)
- Constructing psychometric scales (CES-D 10 with Radloff 1977 reverse scoring)
- Building per capita expenditure with piecewise linear spline at median
- Running high-dimensional fixed effects regressions (`linearmodels.PanelOLS`)
- Producing publication-grade visualizations

## Pipeline Architecture

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | `01_data_cleaning.py` | Merges 6 CHARLS modules, constructs CES-D 10, 4-category education, log PCE spline, sample weights |
| 2 | `02_descriptive_stats.py` | Cronbach's Alpha, weighted CES-D distributions by gender × age (Table 1-2) |
| 3 | `03_regression_analysis.py` | PanelOLS with County/Community FE, community-clustered SE, gender-stratified (Table 3) |
| 4 | `04_visualization.py` | CES-D density histograms by gender (Figure 1) |
| 5 | `05_coefficient_plot.py` | OLS coefficient forest plot with community-clustered 95% CIs (Figure 2) |
| 6 | `06_stepwise_regression.py` | Stepwise addition of childhood/family/health controls (Tables 4-6) |
| 7 | `07_coefficient_shrinkage_plot.py` | Coefficient attenuation visualization |

## Key Replication Results

| Metric | This Replication | Original Paper | Status |
|--------|-----------------|----------------|--------|
| Cronbach's α | **0.809** | 0.815 | ✅ |
| Male CES-D Mean | ~7.5 | 7.1 | ✅ |
| Female CES-D Mean | ~9.5 | 8.9 | ✅ |
| Education gradient | Monotonic protective | Monotonic protective | ✅ |
| Junior High+ (Male) | **-2.09*** | -1.74*** | ✅ |
| Junior High+ (Female) | **-1.70*** | -1.80*** | ✅ |
| Rural coefficient | **+0.77*** | +0.47** | ✅ |

### Methodology Notes
- **Sample weights**: Individual-level weights (`ind_weight`) loaded from CHARLS weight module. Current descriptive statistics use weighted means; regression weights are a planned extension.
- **Education**: 4-category classification matching the original paper (Illiterate [omitted], Can Read/Write, Finished Primary, Junior High+).
- **Log PCE spline**: Household per capita expenditure constructed from CHARLS `household_income` module (food + non-food + annual items), with piecewise linear spline at the sample median. Our log PCE mean (~8.6) is lower than the paper (~9.1), likely due to differences in expenditure item aggregation.
- **Standard errors**: All clustered at the community level, matching the original paper specification.
- **Coefficient magnitude differences**: Our unweighted regressions produce coefficients consistent in direction and significance with the original weighted results. Remaining magnitude differences are expected without applying survey weights in the regression.

## Scope & Limitations

This replication covers **CES-D scale construction, descriptive statistics, and the full baseline SES regression specification** (Tables 1-3).

The following represent known limitations:
- **Biomarker module**: Not available locally; `lower_leg_length` (childhood nutrition proxy) is `NaN`. This data requires separate CHARLS biomarker access.
- **Child mortality shock**: Variable identification in `family_information` module is pending.
- **Survey-weighted regressions**: Sample weights are included in the dataset but not yet applied as regression weights. This is a [planned extension](https://github.com/Skye-SU/charls-depression-ses-replication/issues).

Tables 4-6 (stepwise endogenous controls) are structurally implemented with available variables (chronic disease, ADL limitations).

## Setup & Execution

```bash
# 1. Install dependencies
pip install pandas numpy pyreadstat statsmodels linearmodels matplotlib seaborn

# 2. Set your CHARLS data path
export CHARLS_DATA="/path/to/your/CHARLS"

# 3. Run the pipeline sequentially
python 01_data_cleaning.py
python 02_descriptive_stats.py
python 03_regression_analysis.py
python 04_visualization.py
python 05_coefficient_plot.py
python 06_stepwise_regression.py
python 07_coefficient_shrinkage_plot.py
```

## Disclaimer
This is an independent empirical exercise built for research portfolio purposes. Original dataset belongs to the China Health and Retirement Longitudinal Study (CHARLS) administered by Peking University.
