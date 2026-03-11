# QuickEDA Backend Implementation Plan

## Overview
This document outlines a **step-by-step blueprint** for building the backend of the QuickEDA tool, based on the previously defined specification. The goal is to create a **modular, testable, and incremental implementation plan** that can safely guide development.

---

## Phase 1: Project Setup & Skeleton

### Step 1: Project Structure
- Create a folder structure:
  - `quickeda/`
    - `__init__.py`
    - `eda_report.py` (main class)
    - `utils.py` (helper functions)
    - `plots.py` (plotting helpers)
    - `templates/` (HTML templates for Jinja2)
  - `tests/` (unit tests)
- Initialize git repository
- Set up a virtual environment
- Install dependencies: pandas, scikit-learn, matplotlib, seaborn, Jinja2

### Step 2: Base Class Skeleton
- Implement `EDAReport` class with **empty methods**:
  - `__init__`
  - `analyze_data`
  - `train_baseline_models`
  - `generate_report`
- Include **basic logging placeholders**
- Write minimal **unit tests** to verify object creation

---

## Phase 2: Data Input & Validation

### Step 3: Data Loading
- Accept pandas DataFrame or file path (CSV/Parquet)
- Load dataset into class attribute
- Unit test: verify correct loading and DataFrame creation

### Step 4: Basic Validation
- Check dataset is non-empty
- Check target column exists (if provided)
- Check types of numeric and categorical columns
- Validate train/test split feasibility
- Unit test: test error handling for missing/invalid input

---

## Phase 3: Data Analysis & Summaries

### Step 5: Missing Data Analysis
- Compute column-wise missing percentages
- Flag columns above threshold (configurable or default)
- Generate suggested handling (median/mode imputation)
- Unit test: validate summary accuracy

### Step 6: Numeric Feature Analysis
- Compute statistics (mean, median, std, min, max)
- Detect outliers using IQR or ±3σ
- Generate distribution summary plots for top features
- Unit test: ensure outlier detection works as expected

### Step 7: Categorical Feature Analysis
- Summarize category counts and proportions
- Highlight unusual distributions or rare categories
- Generate bar plots for key categorical features
- Unit test: validate rare category detection

### Step 8: Correlations & Feature Relationships
- Compute all pairwise correlations
- Identify significant predictor-target correlations
- Identify high predictor-predictor correlations (multicollinearity)
- Unit test: check that significant correlations are correctly identified

---

## Phase 4: Baseline Models & Preprocessing

### Step 9: Minimal Preprocessing
- Numeric: median imputation
- Categorical: most-frequent imputation + one-hot encoding
- Drop constant columns
- Unit test: verify preprocessing transforms

### Step 10: Model Training
- Train linear model (Logistic/Linear Regression) and tree-based model (Random Forest)
- Use simple train/test split (default 80/20)
- Store fitted models in object
- Unit test: confirm model training works and metrics can be accessed

### Step 11: Model Evaluation
- Compute evaluation metrics:
  - Classification: Accuracy, ROC-AUC, F1 score
  - Regression: RMSE, R²
- Store metrics in object for report
- Unit test: metrics correctness

### Step 12: Feature Importance
- Extract feature importance from both models
- Rank and store in object
- Unit test: validate top feature extraction

---

## Phase 5: HTML Report Generation

### Step 13: Template Setup
- Create Jinja2 HTML template with placeholders for:
  - Title/overview
  - Key Takeaways
  - Data Summary
  - Feature Relationships
  - Baseline Model Performance
  - Appendix (reproducibility, library versions)
- Unit test: verify template renders with dummy data

### Step 14: Populate Report
- Fill template with:
  - Computed statistics, correlations, missing data
  - Top plots (static images)
  - Model metrics and feature importance tables
  - Suggested feature engineering recommendations
  - Reproducibility info
- Unit test: confirm report file is generated and HTML is valid

### Step 15: Final HTML Output
- Save report to `output_path` provided by user
- Include logging message confirming successful generation
- Unit test: file existence and basic content check

---

## Phase 6: Logging, Reproducibility, & Versioning

### Step 16: Logging
- Add logging messages for all major steps
- Allow configurable verbosity
- Unit test: check logs are printed as expected

### Step 17: Reproducibility
- Store random seeds, preprocessing choices, target column, problem type in object
- Include these details in HTML appendix
- Unit test: verify reproducibility information correctness

### Step 18: Library Versions
- Collect versions of Python, pandas, scikit-learn, matplotlib, seaborn
- Include in HTML appendix
- Unit test: validate version information captured

---

## Phase 7: Testing & Iterative Refinement

### Step 19: Unit Testing
- Test all individual methods for:
  - Correct outputs
  - Handling edge cases (empty data, high missingness, rare categories)
  - Minimal preprocessing transformations
  - Model training and metric computation
  - Report generation

### Step 20: Integration Testing
- Run full pipeline on example dataset
- Verify report accuracy, plots, and metrics
- Confirm logs and reproducibility info included

### Step 21: Iterative Improvements
- Optimize plot readability and report layout
- Ensure concise summaries and key takeaways are prominent
- Verify that suggestions are clearly labeled as recommendations

---

## Phase 8: Documentation & Examples

### Step 22: README & Usage
- Provide instructions for:
  - Installing dependencies
  - Creating `EDAReport` instance
  - Running full analysis and generating HTML
  - Optional config overrides

### Step 23: Example Notebooks
- Create small example datasets
- Demonstrate EDAReport usage
- Validate plots, metrics, and report sections

---

## Phase 9: Iterative Chunking

Each phase can be broken into **small iterative chunks** for implementation, building on previous steps:

1. Skeleton + project setup (Phase 1)
2. Data loading + validation (Phase 2)
3. Missing data summary (Step 5)
4. Numeric feature analysis (Step 6)
5. Categorical feature analysis (Step 7)
6. Correlations & feature relationships (Step 8)
7. Preprocessing (Step 9)
8. Model training & evaluation (Steps 10-11)
9. Feature importance extraction (Step 12)
10. Report template setup (Step 13)
11. Populate report with data (Step 14)
12. Save HTML (Step 15)
13. Logging & reproducibility info (Steps 16-18)
14. Unit + integration testing (Steps 19-20)
15. Iterative improvements and examples (Steps 21-23)

Each chunk is **small enough to implement safely** with strong testing but **large enough to move the project forward**, and all chunks build logically on previous steps.

---

## Status
This plan is ready to guide the backend development of QuickEDA, stored as `plan.md`, and can be iteratively refined as development progresses.

