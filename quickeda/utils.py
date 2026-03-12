"""Utility functions for QuickEDA."""

import sys
import importlib
import logging
from typing import TYPE_CHECKING, Dict, List, Any

if TYPE_CHECKING:
    from .eda_report import EDAReport

logger = logging.getLogger(__name__)


def get_library_versions() -> Dict[str, str]:
    """
    Collect versions of key libraries used by QuickEDA.

    Returns
    -------
    dict
        Mapping from library display name to version string.
    """
    versions: Dict[str, str] = {'python': sys.version.split()[0]}
    libs = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('jinja2', 'jinja2'),
    ]
    for display_name, module_name in libs:
        try:
            mod = importlib.import_module(module_name)
            versions[display_name] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            versions[display_name] = 'not installed'
    return versions


def generate_key_takeaways(report: 'EDAReport') -> Dict[str, List[str]]:
    """
    Derive key insights and warnings from a populated EDAReport.

    Parameters
    ----------
    report : EDAReport
        Must have analyze_data() already called.

    Returns
    -------
    dict with keys 'insights' (list of str) and 'warnings' (list of str).
    """
    insights: List[str] = []
    warnings: List[str] = []

    # Dataset shape
    n_rows, n_cols = report.df.shape
    insights.append(f"Dataset has {n_rows:,} rows and {n_cols} columns.")

    # Missing data
    if report.missing_values_summary:
        any_missing = list(report.missing_values_summary.keys())
        high_missing = [
            col for col, info in report.missing_values_summary.items()
            if info['percentage'] >= report.missing_threshold
        ]
        insights.append(f"{len(any_missing)} column(s) contain missing values.")
        if high_missing:
            col_list = ', '.join(high_missing[:5])
            suffix = '...' if len(high_missing) > 5 else ''
            warnings.append(
                f"{len(high_missing)} column(s) with >50% missing values: {col_list}{suffix}"
            )
    else:
        insights.append("No missing values detected.")

    # Top predictors
    if report.target_correlations:
        top_feature, top_corr = next(iter(report.target_correlations.items()))
        insights.append(
            f'Top predictor for "{report.target}": "{top_feature}" (|r| = {abs(top_corr):.3f}).'
        )
        top3 = list(report.target_correlations.items())[:3]
        feat_list = ', '.join(f'"{f}" ({c:.3f})' for f, c in top3)
        insights.append(f"Top correlated features with target: {feat_list}.")

    # Multicollinearity
    if report.high_correlations:
        warnings.append(
            f"{len(report.high_correlations)} predictor pair(s) have |r| > 0.7 "
            f"(potential multicollinearity)."
        )

    # Class imbalance for classification
    if report.problem_type == 'classification' and report.target:
        try:
            target_vals = report.df[report.target].value_counts(normalize=True)
            min_class_frac = float(target_vals.min())
            if min_class_frac < 0.10:
                warnings.append(
                    f"Severe class imbalance: minority class is {min_class_frac:.1%} of data."
                )
            elif min_class_frac < 0.20:
                warnings.append(
                    f"Moderate class imbalance: minority class is {min_class_frac:.1%} of data."
                )
        except Exception:
            pass

    # Rare categories
    rare_cols = [
        col for col, info in report.categorical_summary.items()
        if info.get('rare_category_count', 0) > 0
    ]
    if rare_cols:
        col_list = ', '.join(rare_cols[:5])
        suffix = '...' if len(rare_cols) > 5 else ''
        warnings.append(
            f"{len(rare_cols)} categorical column(s) contain rare categories (<5%): {col_list}{suffix}"
        )

    # Outliers
    total_outliers = sum(
        info['iqr_method']['count']
        for info in report.outliers_summary.values()
    )
    if total_outliers > 0:
        insights.append(f"Total IQR-method outliers across all numeric features: {total_outliers:,}.")

    # Model performance summary
    if report.metrics and 'tree_model' in report.metrics:
        tree_test = report.metrics['tree_model'].get('test', {})
        if report.problem_type == 'classification':
            acc = tree_test.get('accuracy')
            auc = tree_test.get('roc_auc')
            if acc is not None:
                insights.append(f"Random Forest baseline test accuracy: {acc:.2%}.")
            if auc is not None:
                insights.append(f"Random Forest baseline ROC-AUC: {auc:.3f}.")
        else:
            r2 = tree_test.get('r2')
            rmse = tree_test.get('rmse')
            if r2 is not None:
                insights.append(f"Random Forest baseline test R²: {r2:.3f}.")
            if rmse is not None:
                insights.append(f"Random Forest baseline test RMSE: {rmse:.4f}.")

    return {'insights': insights, 'warnings': warnings}
