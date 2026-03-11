"""
Tests for Phase 5: HTML Report Generation, and Phase 6: Logging/Reproducibility/Versioning.
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np

from quickeda import EDAReport
from quickeda import utils


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_classification_df():
    np.random.seed(0)
    n = 120
    df = pd.DataFrame({
        'feat_a': np.random.randn(n),
        'feat_b': np.random.randn(n),
        'feat_c': np.random.choice(['X', 'Y', 'Z'], n),
        'target': np.random.randint(0, 2, n),
    })
    return df


@pytest.fixture
def simple_regression_df():
    np.random.seed(1)
    n = 120
    df = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'target': np.random.randn(n),
    })
    return df


@pytest.fixture
def report_cls(simple_classification_df):
    r = EDAReport(
        simple_classification_df,
        target='target',
        problem_type='classification',
        random_seed=42,
    )
    r.analyze_data()
    r.train_baseline_models()
    return r


@pytest.fixture
def report_reg(simple_regression_df):
    r = EDAReport(
        simple_regression_df,
        target='target',
        problem_type='regression',
        random_seed=7,
    )
    r.analyze_data()
    r.train_baseline_models()
    return r


@pytest.fixture
def report_no_target(simple_classification_df):
    r = EDAReport(simple_classification_df)
    r.analyze_data()
    return r


@pytest.fixture
def tmp_html(tmp_path):
    return str(tmp_path / 'report.html')


# ── Step 13: Template Setup ────────────────────────────────────────────────────

class TestTemplateSetup:
    def test_template_file_exists(self):
        """Template file must exist in the package."""
        import quickeda
        template_path = os.path.join(
            os.path.dirname(quickeda.__file__), 'templates', 'report.html'
        )
        assert os.path.isfile(template_path), "templates/report.html not found"

    def test_template_renders_with_minimal_context(self, tmp_html, simple_classification_df):
        """Template should render with a minimal EDAReport (no models)."""
        r = EDAReport(simple_classification_df)
        r.analyze_data()
        r.generate_report(tmp_html)
        assert os.path.isfile(tmp_html)
        assert os.path.getsize(tmp_html) > 1000

    def test_template_renders_with_full_context(self, report_cls, tmp_html):
        """Template renders with a fully populated report."""
        report_cls.generate_report(tmp_html)
        assert os.path.isfile(tmp_html)
        assert os.path.getsize(tmp_html) > 5000


# ── Step 14: Populate Report ──────────────────────────────────────────────────

class TestPopulateReport:
    def test_report_contains_dataset_overview(self, report_cls, tmp_html):
        report_cls.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'Dataset Overview' in html

    def test_report_contains_key_takeaways(self, report_cls, tmp_html):
        report_cls.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'Key Takeaways' in html

    def test_report_contains_data_summary(self, report_cls, tmp_html):
        report_cls.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'Data Summary' in html

    def test_report_contains_feature_distributions(self, report_cls, tmp_html):
        report_cls.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'Feature Distributions' in html

    def test_report_contains_feature_relationships(self, report_cls, tmp_html):
        report_cls.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'Feature Relationships' in html

    def test_report_contains_baseline_models_section(self, report_cls, tmp_html):
        report_cls.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'Baseline Model Performance' in html

    def test_report_contains_appendix(self, report_cls, tmp_html):
        report_cls.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'Appendix' in html

    def test_report_embeds_distribution_plots(self, report_cls, tmp_html):
        """Distribution images should be embedded as base64 PNGs."""
        report_cls.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'data:image/png;base64,' in html

    def test_report_includes_feature_importance_plots(self, report_cls, tmp_html):
        report_cls.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'Feature Importance' in html

    def test_classification_report_shows_accuracy(self, report_cls, tmp_html):
        report_cls.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'Accuracy' in html

    def test_classification_report_shows_roc_auc(self, report_cls, tmp_html):
        report_cls.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'ROC-AUC' in html

    def test_regression_report_shows_rmse(self, report_reg, tmp_html):
        report_reg.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'RMSE' in html

    def test_regression_report_shows_r2(self, report_reg, tmp_html):
        report_reg.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'R²' in html

    def test_report_shows_logistic_regression_label(self, report_cls, tmp_html):
        report_cls.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'Logistic Regression' in html

    def test_report_shows_random_forest_label(self, report_cls, tmp_html):
        report_cls.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'Random Forest' in html

    def test_report_no_target_skips_model_section(self, report_no_target, tmp_html):
        """Without a target, the Baseline Model section must not appear."""
        report_no_target.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'Baseline Model Performance' not in html

    def test_report_missing_values_section(self, tmp_html):
        """Report correctly shows columns with missing values."""
        df = pd.DataFrame({
            'good': [1, 2, 3, 4, 5],
            'bad':  [1, None, None, None, 5],
            'target': [0, 1, 0, 1, 0],
        })
        r = EDAReport(df, target='target', problem_type='classification')
        r.analyze_data()
        r.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'Missing Values' in html
        assert 'bad' in html

    def test_report_high_correlations_section(self, tmp_html):
        """Report shows High Predictor-Predictor Correlations when they exist."""
        np.random.seed(0)
        base = np.random.randn(100)
        df = pd.DataFrame({
            'a': base,
            'b': base + np.random.randn(100) * 0.05,  # |r| > 0.7 with a
            'target': np.random.randint(0, 2, 100),
        })
        r = EDAReport(df, target='target', problem_type='classification')
        r.analyze_data()
        r.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'Predictor-Predictor Correlations' in html


# ── Step 15: Final HTML Output ────────────────────────────────────────────────

class TestFinalHTMLOutput:
    def test_output_file_created(self, report_cls, tmp_html):
        """generate_report() creates the output file."""
        report_cls.generate_report(tmp_html)
        assert os.path.isfile(tmp_html)

    def test_output_is_valid_html(self, report_cls, tmp_html):
        """Output must start with a DOCTYPE declaration and include html tags."""
        report_cls.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read().strip()
        assert html.startswith('<!DOCTYPE html')
        assert '<html' in html
        assert '</html>' in html

    def test_output_directory_created_automatically(self, tmp_path, report_cls):
        """Nested output directories are created automatically."""
        nested = str(tmp_path / 'nested' / 'dir' / 'report.html')
        report_cls.generate_report(nested)
        assert os.path.isfile(nested)

    def test_report_is_utf8_encoded(self, report_cls, tmp_html):
        report_cls.generate_report(tmp_html)
        # Should not raise UnicodeDecodeError
        with open(tmp_html, encoding='utf-8') as f:
            f.read()

    def test_auto_analyze_before_generating(self, simple_classification_df, tmp_html):
        """generate_report() calls analyze_data() automatically if needed."""
        r = EDAReport(
            simple_classification_df,
            target='target',
            problem_type='classification',
        )
        # Do NOT call analyze_data() manually
        r.generate_report(tmp_html)
        assert os.path.isfile(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'feat_a' in html

    def test_logging_message_on_success(self, report_cls, tmp_html, caplog):
        """generate_report() logs a success message."""
        import logging
        with caplog.at_level(logging.INFO, logger='quickeda.eda_report'):
            report_cls.generate_report(tmp_html)
        assert any('Report saved' in m for m in caplog.messages)


# ── Step 16: Logging ──────────────────────────────────────────────────────────

class TestLogging:
    def test_analyze_data_logs_info(self, simple_classification_df, caplog):
        import logging
        r = EDAReport(simple_classification_df)
        with caplog.at_level(logging.INFO, logger='quickeda.eda_report'):
            r.analyze_data()
        assert len(caplog.records) > 0

    def test_train_models_logs_info(self, simple_classification_df, caplog):
        import logging
        r = EDAReport(
            simple_classification_df,
            target='target',
            problem_type='classification',
        )
        r.analyze_data()
        with caplog.at_level(logging.INFO, logger='quickeda.eda_report'):
            r.train_baseline_models()
        assert any('Training' in m or 'complete' in m.lower() for m in caplog.messages)


# ── Step 17 & 18: Reproducibility & Library Versions ─────────────────────────

class TestReproducibility:
    def test_reproducibility_info_in_report(self, report_cls, tmp_html):
        """Appendix must include random seed and split info."""
        report_cls.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'Random Seed' in html
        assert '42' in html          # random_seed
        assert '80%' in html         # train split

    def test_preprocessing_choices_in_report(self, report_cls, tmp_html):
        report_cls.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'Median' in html
        assert 'One-hot' in html or 'one-hot' in html

    def test_library_versions_in_report(self, report_cls, tmp_html):
        report_cls.generate_report(tmp_html)
        html = open(tmp_html, encoding='utf-8').read()
        assert 'pandas' in html
        assert 'scikit-learn' in html
        assert 'python' in html.lower()

    def test_get_library_versions_returns_dict(self):
        versions = utils.get_library_versions()
        assert isinstance(versions, dict)
        assert 'python' in versions
        assert 'pandas' in versions
        assert 'scikit-learn' in versions

    def test_library_versions_are_strings(self):
        versions = utils.get_library_versions()
        for k, v in versions.items():
            assert isinstance(v, str), f"Version for {k} is not a string"

    def test_python_version_present(self):
        versions = utils.get_library_versions()
        import sys
        major_minor = f'{sys.version_info.major}.{sys.version_info.minor}'
        assert major_minor in versions['python']

    def test_reproducibility_same_seed_same_split(self, simple_classification_df):
        """Two reports with same random_seed should have identical metrics."""
        def run(seed):
            r = EDAReport(
                simple_classification_df,
                target='target',
                problem_type='classification',
                random_seed=seed,
            )
            r.analyze_data()
            r.train_baseline_models()
            return r.metrics['tree_model']['test']['accuracy']

        assert run(99) == run(99)

    def test_different_seeds_may_differ(self, simple_classification_df):
        """Two different seeds don't always produce identical results."""
        def run(seed):
            r = EDAReport(
                simple_classification_df,
                target='target',
                problem_type='classification',
                random_seed=seed,
            )
            r.analyze_data()
            r.train_baseline_models()
            return r.metrics['tree_model']['test']['accuracy']

        # With only 120 rows the small split may sometimes agree; not guaranteed
        # to differ, so just check the function runs cleanly for both.
        a1 = run(1)
        a2 = run(999)
        assert isinstance(a1, float)
        assert isinstance(a2, float)


# ── Utils: generate_key_takeaways ─────────────────────────────────────────────

class TestGenerateKeyTakeaways:
    def test_returns_dict_with_insights_and_warnings(self, report_cls):
        result = utils.generate_key_takeaways(report_cls)
        assert 'insights' in result
        assert 'warnings' in result
        assert isinstance(result['insights'], list)
        assert isinstance(result['warnings'], list)

    def test_insights_mention_row_count(self, report_cls):
        result = utils.generate_key_takeaways(report_cls)
        combined = ' '.join(result['insights'])
        assert 'rows' in combined

    def test_insights_mention_top_predictor(self, report_cls):
        result = utils.generate_key_takeaways(report_cls)
        combined = ' '.join(result['insights'])
        assert 'predictor' in combined.lower() or 'target' in combined.lower()

    def test_class_imbalance_warning(self):
        """Severe class imbalance should trigger a warning."""
        np.random.seed(42)
        majority = [0] * 950 + [1] * 50
        df = pd.DataFrame({
            'x': np.random.randn(1000),
            'target': majority,
        })
        r = EDAReport(df, target='target', problem_type='classification')
        r.analyze_data()
        result = utils.generate_key_takeaways(r)
        combined = ' '.join(result['warnings'])
        assert 'imbalance' in combined.lower() or 'balanced' in combined.lower() or 'class' in combined.lower()

    def test_no_missing_values_in_insights(self, report_cls):
        """When there are no missing values, an insight should say so."""
        # simple_classification_df has no NaN
        result = utils.generate_key_takeaways(report_cls)
        combined = ' '.join(result['insights'])
        assert 'no missing' in combined.lower() or 'missing values' in combined.lower()
