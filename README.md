[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/wOs3Tno3)

# QuickEDA

A Python tool that performs **high-level exploratory data analysis (EDA)** on tabular datasets and generates a self-contained **HTML report** with key insights, statistical summaries, visualizations, and baseline model performance.

---

## Features

- **Data loading** — accepts a pandas `DataFrame` or a file path (CSV or Parquet)
- **Automated analysis** — missing-value summary with handling suggestions, numeric statistics, outlier detection (IQR & ±3σ), categorical summaries, and pairwise correlations
- **Baseline models** — trains Logistic/Linear Regression and Random Forest automatically; reports Accuracy, ROC-AUC, F1 (classification) or RMSE, R² (regression)
- **Feature importance** — ranks features by both linear coefficients and tree importance
- **HTML report** — single self-contained file with embedded plots, tables, key takeaways, and an appendix with library versions and reproducibility info

---

## Installation

```bash
# Clone the repo and enter the directory
git clone <repo-url>
cd project-repository-zhang-j-j

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

```python
from quickeda import EDAReport

# 1. Create report object — accepts a file path or a DataFrame
report = EDAReport(
    data='path/to/dataset.csv',
    target='my_target_column',       # optional
    problem_type='classification',   # or 'regression'; inferred if omitted
    random_seed=42,
    train_test_split_ratio=0.8,
    num_top_features=10,
)

# 2. Run exploratory analysis
report.analyze_data()

# 3. Train baseline models  (requires target)
report.train_baseline_models()

# 4. Generate HTML report
report.generate_report('output/report.html')
```

`generate_report()` will call `analyze_data()` automatically if you haven't done so.

---

## Configuration

All options can be passed as keyword arguments to `EDAReport.__init__`:

| Parameter | Default | Description |
|---|---|---|
| `target` | `None` | Target column name; enables supervised analysis |
| `problem_type` | `None` | `'classification'` or `'regression'`; auto-inferred if omitted |
| `random_seed` | `42` | Seed for train/test split and model training |
| `train_test_split_ratio` | `0.8` | Fraction of data used for training |
| `num_top_features` | `10` | Number of top features shown in plots and tables |
| `missing_threshold` | `0.5` | Column missing-rate above which a warning is raised |

---

## HTML Report Sections

| Section | Content |
|---|---|
| **Key Takeaways** | Auto-generated insights and warnings (imbalance, multicollinearity, top predictors) |
| **Dataset Overview** | Shape, feature counts, target, problem type |
| **Data Summary** | Missing-value table, numeric statistics, categorical summary |
| **Feature Distributions** | Histogram plots for top numeric and categorical features |
| **Feature Relationships** | Target correlation table & plot, correlation heatmap, high collinearity table |
| **Baseline Model Performance** | Metrics table (train & test) for linear and tree models; feature importance plots |
| **Appendix** | Preprocessing choices, random seed, train/test split, library versions |

---

## Running the Interactive Analysis Script

```bash
# Interactive mode — prompts for file path, target, and problem type
python real_tests/analyze_dataset.py

# Non-interactive mode
python real_tests/analyze_dataset.py path/to/data.csv \
    --target my_col --problem-type classification
```

---

## Running Tests

```bash
# All tests
pytest

# With coverage report
pytest --cov=quickeda --cov-report=term-missing

# A single test file
pytest tests/test_phase5_report.py -v
```

Current test suite: **150 tests across 6 test modules**, covering Phases 1–6.

---

## Project Structure

```
quickeda/
├── __init__.py          # Package exports
├── eda_report.py        # EDAReport class (main implementation)
├── plots.py             # Matplotlib/Seaborn plot generators (base64 PNG output)
├── utils.py             # Library version collection, key-takeaway generation
└── templates/
    └── report.html      # Jinja2 HTML template

tests/
├── test_eda_report.py               # Phase 1–2: class skeleton & instantiation
├── test_data_loading_validation.py  # Phase 2: loading, validation
├── test_phase3_analysis.py          # Phase 3: missing data, numeric, categorical, correlations
├── test_phase4_models.py            # Phase 4: preprocessing, model training, evaluation, importance
└── test_phase5_report.py            # Phase 5–6: report generation, logging, reproducibility, versions

real_tests/
├── analyze_dataset.py   # CLI / interactive analysis & report script
├── creditcard.csv       # Example dataset
└── diabetes.csv         # Example dataset
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Preprocessing, model training, metrics |
| `matplotlib` | Plot generation |
| `seaborn` | Styled statistical plots |
| `Jinja2` | HTML templating |
| `pyarrow` | Parquet file format support |

