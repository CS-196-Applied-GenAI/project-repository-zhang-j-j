"""Plotting helpers for generating visualizations."""

import io
import base64
import logging
from typing import Dict, List, Any

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted")


def _plot_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=96)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


def generate_distribution_plots(
    df: pd.DataFrame,
    numeric_cols: List[str],
    top_n: int = 10,
) -> Dict[str, str]:
    """
    Generate histogram distribution plots for numeric features.

    Parameters
    ----------
    df : pd.DataFrame
    numeric_cols : list of str
    top_n : int

    Returns
    -------
    dict
        Mapping from column name to base64-encoded PNG string.
    """
    plots: Dict[str, str] = {}
    for col in numeric_cols[:top_n]:
        try:
            data = df[col].dropna()
            if data.empty:
                continue
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(data, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
            ax.set_title(col, fontsize=11, fontweight='bold')
            ax.set_xlabel('Value', fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            ax.tick_params(labelsize=8)
            fig.tight_layout()
            plots[col] = _plot_to_base64(fig)
        except Exception as exc:
            logger.warning(f"Could not generate distribution plot for '{col}': {exc}")
    return plots


def generate_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    top_n: int = 20,
) -> str:
    """
    Generate a lower-triangle correlation heatmap.

    Parameters
    ----------
    correlation_matrix : pd.DataFrame
    top_n : int
        Maximum number of features to display (selects by mean absolute correlation).

    Returns
    -------
    str
        Base64-encoded PNG, or empty string on failure.
    """
    try:
        if correlation_matrix.empty:
            return ''
        if len(correlation_matrix) > top_n:
            mean_abs = correlation_matrix.abs().mean().sort_values(ascending=False)
            top_cols = mean_abs.head(top_n).index.tolist()
            corr = correlation_matrix.loc[top_cols, top_cols]
        else:
            corr = correlation_matrix

        n = len(corr)
        fig_size = max(6, min(14, n * 0.65))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            mask=mask, annot=(n <= 15), fmt='.2f',
            annot_kws={'size': 7}, linewidths=0.4, square=True,
            cbar_kws={'shrink': 0.8},
        )
        ax.set_title('Feature Correlation Heatmap', fontsize=13, fontweight='bold', pad=10)
        ax.tick_params(labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        fig.tight_layout()
        return _plot_to_base64(fig)
    except Exception as exc:
        logger.warning(f"Could not generate correlation heatmap: {exc}")
        return ''


def generate_feature_importance_plot(
    importance_list: List[Dict[str, Any]],
    model_name: str,
    top_n: int = 15,
) -> str:
    """
    Generate a horizontal bar chart for feature importance.

    Parameters
    ----------
    importance_list : list of dicts with 'feature' and 'importance' keys (sorted descending)
    model_name : str
    top_n : int

    Returns
    -------
    str
        Base64-encoded PNG, or empty string on failure.
    """
    try:
        items = importance_list[:top_n]
        if not items:
            return ''

        features = [item['feature'] for item in items][::-1]
        importances = [item['importance'] for item in items][::-1]

        fig, ax = plt.subplots(figsize=(6, max(3, len(features) * 0.4)))
        bars = ax.barh(features, importances, color='steelblue', edgecolor='white', height=0.65)
        max_imp = max(importances) if importances else 1.0
        for bar, val in zip(bars, importances):
            ax.text(
                bar.get_width() + max_imp * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', ha='left', fontsize=7.5,
            )
        ax.set_xlabel('Importance', fontsize=9)
        ax.set_title(f'{model_name} — Top {len(features)} Features', fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, max_imp * 1.18)
        fig.tight_layout()
        return _plot_to_base64(fig)
    except Exception as exc:
        logger.warning(f"Could not generate feature importance plot for '{model_name}': {exc}")
        return ''


def generate_target_correlation_plot(
    target_correlations: Dict[str, float],
    target_name: str,
    top_n: int = 15,
) -> str:
    """
    Generate a horizontal bar chart of feature-target correlations.

    Returns
    -------
    str
        Base64-encoded PNG, or empty string on failure.
    """
    try:
        if not target_correlations:
            return ''
        items = list(target_correlations.items())[:top_n]
        features = [i[0] for i in items][::-1]
        corrs = [i[1] for i in items][::-1]
        colors = ['steelblue' if c >= 0 else 'salmon' for c in corrs]

        fig, ax = plt.subplots(figsize=(6, max(3, len(features) * 0.4)))
        ax.barh(features, corrs, color=colors, edgecolor='white', height=0.65)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel('Pearson Correlation', fontsize=9)
        ax.set_title(f'Feature Correlations with "{target_name}"', fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=8)
        ax.set_xlim(-1, 1)
        fig.tight_layout()
        return _plot_to_base64(fig)
    except Exception as exc:
        logger.warning(f"Could not generate target correlation plot: {exc}")
        return ''


def generate_categorical_plots(
    df: pd.DataFrame,
    categorical_summary: Dict[str, Any],
    top_n: int = 5,
) -> Dict[str, str]:
    """
    Generate horizontal bar plots for categorical features.

    Returns
    -------
    dict
        Mapping from column name to base64-encoded PNG string.
    """
    plots: Dict[str, str] = {}
    for col in list(categorical_summary.keys())[:top_n]:
        try:
            if col not in df.columns:
                continue
            vc = df[col].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(5, max(2.5, len(vc) * 0.32)))
            vc[::-1].plot(kind='barh', ax=ax, color='teal', edgecolor='white')
            ax.set_title(f'"{col}" Value Counts', fontsize=11, fontweight='bold')
            ax.set_xlabel('Count', fontsize=9)
            ax.tick_params(labelsize=8)
            plt.setp(ax.get_yticklabels(), rotation=0)
            fig.tight_layout()
            plots[col] = _plot_to_base64(fig)
        except Exception as exc:
            logger.warning(f"Could not generate categorical plot for '{col}': {exc}")
    return plots
