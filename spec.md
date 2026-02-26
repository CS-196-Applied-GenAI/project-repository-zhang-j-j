# QuickEDA Tool Specification

## Overview
QuickEDA is a Python-based data science tool designed to perform **high-level exploratory data analysis (EDA)** on tabular datasets quickly and generate a **concise HTML report** with key takeaways and baseline model insights. The tool is aimed at **experienced data scientists** looking to save time during the initial stages of dataset exploration.

---

## Target Users
- Experienced data scientists
- Goal: accelerate dataset understanding and initial model development

---

## Supported Data
- Tabular datasets only (rows × columns)
- Numeric and categorical columns
- Dataset size: up to ~50,000 rows
- One dataset per `EDAReport` instance

---

## Inputs
- Accepts either:
  - pandas DataFrame
  - File path (CSV, Parquet, etc.)
- Optional target column for supervised analysis
- Optional `config` dictionary or keyword arguments to override defaults:
  - `problem_type` (classification/regression)
  - `target` column
  - Models to run
  - Number of top features to highlight
  - Random seed / train-test split ratio

---

## Outputs
- Primary output: **HTML report**
- Report includes:
  1. **Title & Overview** – dataset name, shape, target, problem type
  2. **Key Takeaways / Insights** – actionable points, top predictors, warnings, suggested feature engineering (optional and clearly labeled as recommendations)
  3. **Data Summary** – missing values summary, constant columns, distributions
  4. **Feature Relationships** – predictor-target correlations, significant predictor-predictor correlations highlighting potential multicollinearity
  5. **Baseline Model Performance** – table with evaluation metrics for linear and tree-based models, feature importance plots
  6. **Appendix** – library versions, reproducibility info, optional extra plots

- **Conciseness principles:** only most important plots displayed, textual summaries for other insights
- **Plots included:** top feature distributions, correlation heatmap (highlighting significant correlations), baseline model feature importance, outlier summaries
- **Fitted baseline models** stored in the object for later reuse

---

## Baseline Models
- **Classification:** Logistic Regression + Random Forest Classifier
- **Regression:** Linear Regression + Random Forest Regressor
- Minimal preprocessing only:
  - Numeric: median imputation
  - Categorical: most frequent imputation + one-hot encoding
  - Drop constant columns
  - No advanced feature engineering
  - Scaling only if required by model
- Train/test split: simple 80/20 split with fixed random seed (configurable)
- Metrics:
  - Classification: Accuracy, ROC-AUC, F1 score
  - Regression: RMSE, R²

---

## Data Analysis
- Missing data summary (column-wise) with suggested handling
- Numeric feature distributions + outlier detection (IQR / ±3σ)
- Categorical feature analysis: flag unusual distributions or rare categories
- Compute all pairwise correlations internally, report only significant ones
- Feature importance from baseline models highlighted in report

---

## Object-Oriented API
```python
class EDAReport:
    def __init__(self, data, target=None, problem_type=None, config=None):
        """Initialize with dataset and optional target, problem type, and configuration."""

    def analyze_data(self):
        """Compute statistics, correlations, missing values, outliers, and categorical issues."""

    def train_baseline_models(self):
        """Perform minimal preprocessing and train linear + tree-based models."""

    def generate_report(self, output_path):
        """Generate the HTML report with textual summaries and key static plots."""
```
- Optional private helper methods: `_compute_feature_importance`, `_plot_distributions`, `_detect_outliers`
- Stores intermediate results useful for modeling: feature-target correlations, predictor-predictor correlations, baseline model importances, missing value summary
- Optional logging/progress messages during analysis and model training

---

## Reproducibility
- Store all relevant details in object and HTML report:
  - Random seed for train/test split and model training
  - Problem type and target column
  - Minimal preprocessing choices
  - Library versions (Python, pandas, scikit-learn, matplotlib/seaborn) in appendix/sidebar

---

## Error Handling
- Fatal errors: clearly raised (empty dataset, missing target, incompatible data types, impossible splits)
- Non-fatal issues: reported as warnings in HTML report (high missingness, outliers, rare categories)

---

## Implementation Notes
- Python-only, standalone module (no pip package required)
- Dependencies: pandas, scikit-learn, matplotlib/seaborn, Jinja2 (minimal templating)
- No caching except storing fitted baseline models for reuse
- Only one dataset per run
- HTML report emphasizes clarity and actionable insights, not exhaustive plots
- Feature engineering suggestions included but **clearly labeled as recommendations only**

