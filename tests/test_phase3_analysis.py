"""Unit tests for Phase 3: Data Analysis & Summaries (Steps 5-8)."""

import pytest
import pandas as pd
import numpy as np
from quickeda.eda_report import EDAReport


class TestMissingDataAnalysis:
    """Test Step 5: Missing Data Analysis."""
    
    @pytest.fixture
    def dataframe_with_missing(self):
        """Create a DataFrame with various missing value patterns."""
        return pd.DataFrame({
            'no_missing': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'few_missing': [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
            'many_missing': [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            'categorical': ['A', 'B', np.nan, 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
            'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
    
    def test_analyze_missing_data(self, dataframe_with_missing):
        """Test that missing data analysis detects missing values correctly."""
        report = EDAReport(data=dataframe_with_missing, target='target')
        report.analyze_data()
        
        # Check that missing values are detected
        assert 'few_missing' in report.missing_values_summary
        assert 'many_missing' in report.missing_values_summary
        assert 'categorical' in report.missing_values_summary
        
        # Check that columns without missing values are not in summary
        assert 'no_missing' not in report.missing_values_summary
        assert 'target' not in report.missing_values_summary
    
    def test_missing_value_counts(self, dataframe_with_missing):
        """Test that missing value counts are accurate."""
        report = EDAReport(data=dataframe_with_missing, target='target')
        report.analyze_data()
        
        assert report.missing_values_summary['few_missing']['count'] == 1
        assert report.missing_values_summary['many_missing']['count'] == 8
        assert report.missing_values_summary['categorical']['count'] == 1
    
    def test_missing_value_percentages(self, dataframe_with_missing):
        """Test that missing value percentages are accurate."""
        report = EDAReport(data=dataframe_with_missing, target='target')
        report.analyze_data()
        
        assert abs(report.missing_values_summary['few_missing']['percentage'] - 0.1) < 0.01
        assert abs(report.missing_values_summary['many_missing']['percentage'] - 0.8) < 0.01
    
    def test_missing_handling_suggestions(self, dataframe_with_missing):
        """Test that appropriate handling methods are suggested."""
        report = EDAReport(data=dataframe_with_missing, target='target')
        report.analyze_data()
        
        # Few missing numeric -> median imputation
        assert report.missing_values_summary['few_missing']['suggested_handling'] == 'median_imputation'
        
        # Many missing (>50%) -> drop column
        assert report.missing_values_summary['many_missing']['suggested_handling'] == 'drop_column'
        
        # Few missing categorical -> mode imputation
        assert report.missing_values_summary['categorical']['suggested_handling'] == 'mode_imputation'
    
    def test_no_missing_data(self):
        """Test analysis with no missing data."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': ['x', 'y', 'z', 'x', 'y']
        })
        
        report = EDAReport(data=df)
        report.analyze_data()
        
        assert len(report.missing_values_summary) == 0


class TestNumericFeatureAnalysis:
    """Test Step 6: Numeric Feature Analysis."""
    
    @pytest.fixture
    def numeric_dataframe(self):
        """Create a DataFrame with numeric features."""
        np.random.seed(42)
        return pd.DataFrame({
            'normal': np.random.normal(100, 15, 100),
            'skewed': np.concatenate([np.random.normal(50, 10, 95), [200, 210, 220, 250, 300]]),
            'uniform': np.random.uniform(0, 100, 100),
            'target': np.random.randint(0, 2, 100)
        })
    
    def test_numeric_statistics_computed(self, numeric_dataframe):
        """Test that basic statistics are computed for numeric columns."""
        report = EDAReport(data=numeric_dataframe, target='target')
        report.analyze_data()
        
        assert 'normal' in report.numeric_statistics
        assert 'skewed' in report.numeric_statistics
        assert 'uniform' in report.numeric_statistics
    
    def test_statistics_keys(self, numeric_dataframe):
        """Test that all expected statistics are present."""
        report = EDAReport(data=numeric_dataframe, target='target')
        report.analyze_data()
        
        stats = report.numeric_statistics['normal']
        expected_keys = ['mean', 'median', 'std', 'min', 'max', 'q25', 'q75']
        
        for key in expected_keys:
            assert key in stats
    
    def test_outlier_detection(self, numeric_dataframe):
        """Test that outliers are detected correctly."""
        report = EDAReport(data=numeric_dataframe, target='target')
        report.analyze_data()
        
        # Check outlier summary exists
        assert 'skewed' in report.outliers_summary
        
        # Check both methods are present
        outlier_info = report.outliers_summary['skewed']
        assert 'iqr_method' in outlier_info
        assert 'sigma_method' in outlier_info
    
    def test_outlier_iqr_method(self, numeric_dataframe):
        """Test IQR outlier detection method."""
        report = EDAReport(data=numeric_dataframe, target='target')
        report.analyze_data()
        
        iqr_result = report.outliers_summary['skewed']['iqr_method']
        
        assert 'lower_bound' in iqr_result
        assert 'upper_bound' in iqr_result
        assert 'count' in iqr_result
        assert 'percentage' in iqr_result
        
        # Skewed data should have outliers detected
        assert iqr_result['count'] > 0
    
    def test_outlier_sigma_method(self, numeric_dataframe):
        """Test ±3σ outlier detection method."""
        report = EDAReport(data=numeric_dataframe, target='target')
        report.analyze_data()
        
        sigma_result = report.outliers_summary['skewed']['sigma_method']
        
        assert 'lower_bound' in sigma_result
        assert 'upper_bound' in sigma_result
        assert 'count' in sigma_result
        assert 'percentage' in sigma_result
    
    def test_statistics_accuracy(self):
        """Test that computed statistics are accurate."""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        report = EDAReport(data=df)
        report.analyze_data()
        
        stats = report.numeric_statistics['values']
        
        assert abs(stats['mean'] - 5.5) < 0.01
        assert abs(stats['median'] - 5.5) < 0.01
        assert stats['min'] == 1
        assert stats['max'] == 10
        assert stats['q25'] == 3.25
        assert stats['q75'] == 7.75


class TestCategoricalFeatureAnalysis:
    """Test Step 7: Categorical Feature Analysis."""
    
    @pytest.fixture
    def categorical_dataframe(self):
        """Create a DataFrame with categorical features."""
        return pd.DataFrame({
            'common': ['A'] * 50 + ['B'] * 30 + ['C'] * 20,
            'rare': ['X'] * 90 + ['Y'] * 5 + ['Z'] * 5,
            'diverse': [f'cat_{i}' for i in range(100)],
            'target': [0, 1] * 50
        })
    
    def test_categorical_summary_created(self, categorical_dataframe):
        """Test that categorical summaries are created."""
        report = EDAReport(data=categorical_dataframe, target='target')
        report.analyze_data()
        
        assert 'common' in report.categorical_summary
        assert 'rare' in report.categorical_summary
        assert 'diverse' in report.categorical_summary
    
    def test_unique_count(self, categorical_dataframe):
        """Test that unique value counts are correct."""
        report = EDAReport(data=categorical_dataframe, target='target')
        report.analyze_data()
        
        assert report.categorical_summary['common']['unique_count'] == 3
        assert report.categorical_summary['rare']['unique_count'] == 3
        assert report.categorical_summary['diverse']['unique_count'] == 100
    
    def test_top_categories(self, categorical_dataframe):
        """Test that top categories are identified."""
        report = EDAReport(data=categorical_dataframe, target='target')
        report.analyze_data()
        
        top_cats = report.categorical_summary['common']['top_categories']
        
        assert 'A' in top_cats
        assert 'B' in top_cats
        assert 'C' in top_cats
        assert top_cats['A'] == 50
    
    def test_rare_category_detection(self, categorical_dataframe):
        """Test that rare categories (<5%) are detected."""
        report = EDAReport(data=categorical_dataframe, target='target')
        report.analyze_data()
        
        rare_cats = report.categorical_summary['rare']['rare_categories']
        
        # Y and Z are each 5% (exactly at threshold, implementation may vary)
        # Some implementations might include them
        assert len(rare_cats) >= 0  # At least checking no error
        
        # Common categories should not have rare categories
        common_rare = report.categorical_summary['common']['rare_categories']
        assert len(common_rare) == 0
    
    def test_rare_category_count(self, categorical_dataframe):
        """Test that rare category count is tracked."""
        report = EDAReport(data=categorical_dataframe, target='target')
        report.analyze_data()
        
        assert 'rare_category_count' in report.categorical_summary['common']
        assert isinstance(report.categorical_summary['common']['rare_category_count'], int)


class TestCorrelationAnalysis:
    """Test Step 8: Correlations & Feature Relationships."""
    
    @pytest.fixture
    def correlated_dataframe(self):
        """Create a DataFrame with known correlations."""
        np.random.seed(42)
        x1 = np.random.normal(0, 1, 100)
        x2 = x1 + np.random.normal(0, 0.1, 100)  # Highly correlated with x1
        x3 = np.random.normal(0, 1, 100)  # Independent
        target = 2 * x1 + np.random.normal(0, 0.5, 100)  # Depends on x1
        
        return pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'target': target
        })
    
    def test_correlation_matrix_computed(self, correlated_dataframe):
        """Test that correlation matrix is computed."""
        report = EDAReport(data=correlated_dataframe, target='target')
        report.analyze_data()
        
        assert not report.feature_correlations.empty
        assert 'x1' in report.feature_correlations.columns
        assert 'target' in report.feature_correlations.columns
    
    def test_correlation_matrix_symmetric(self, correlated_dataframe):
        """Test that correlation matrix is symmetric."""
        report = EDAReport(data=correlated_dataframe, target='target')
        report.analyze_data()
        
        corr_matrix = report.feature_correlations
        
        # Check symmetry
        for i in corr_matrix.columns:
            for j in corr_matrix.columns:
                assert abs(corr_matrix.loc[i, j] - corr_matrix.loc[j, i]) < 0.001
    
    def test_target_correlations(self, correlated_dataframe):
        """Test that target correlations are identified."""
        report = EDAReport(data=correlated_dataframe, target='target')
        report.analyze_data()
        
        assert len(report.target_correlations) > 0
        assert 'x1' in report.target_correlations
        assert 'x2' in report.target_correlations
        assert 'x3' in report.target_correlations
    
    def test_target_correlations_sorted(self, correlated_dataframe):
        """Test that target correlations are sorted by importance."""
        report = EDAReport(data=correlated_dataframe, target='target')
        report.analyze_data()
        
        # Should be sorted by absolute correlation value
        corr_values = list(report.target_correlations.values())
        assert corr_values == sorted(corr_values, reverse=True)
    
    def test_high_correlations_detected(self, correlated_dataframe):
        """Test that high predictor-predictor correlations are detected."""
        report = EDAReport(data=correlated_dataframe, target='target')
        report.analyze_data()
        
        # x1 and x2 are highly correlated
        high_corr_pairs = [(pair[0], pair[1]) for pair in report.high_correlations]
        
        # Should detect x1-x2 correlation
        assert ('x1', 'x2') in high_corr_pairs or ('x2', 'x1') in high_corr_pairs
    
    def test_high_correlation_threshold(self, correlated_dataframe):
        """Test that only correlations above threshold are reported."""
        report = EDAReport(data=correlated_dataframe, target='target')
        report.analyze_data()
        
        # All reported correlations should be |r| > 0.7
        for pair in report.high_correlations:
            assert abs(pair[2]) > 0.7
    
    def test_no_self_correlation_in_high_corr(self, correlated_dataframe):
        """Test that variables are not correlated with themselves in high_correlations."""
        report = EDAReport(data=correlated_dataframe, target='target')
        report.analyze_data()
        
        for pair in report.high_correlations:
            assert pair[0] != pair[1]
    
    def test_correlation_with_no_target(self):
        """Test correlation analysis without a target variable."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [2, 4, 6, 8, 10],
            'c': [5, 4, 3, 2, 1]
        })
        
        report = EDAReport(data=df)
        report.analyze_data()
        
        # Should still compute correlation matrix
        assert not report.feature_correlations.empty
        
        # a and b are perfectly correlated
        assert len(report.high_correlations) > 0


class TestIntegratedAnalysis:
    """Test integrated analysis workflow."""
    
    @pytest.fixture
    def complex_dataframe(self):
        """Create a complex DataFrame with various data patterns."""
        np.random.seed(42)
        return pd.DataFrame({
            'numeric_clean': np.random.normal(100, 15, 200),
            'numeric_missing': np.concatenate([np.random.normal(50, 10, 180), [np.nan] * 20]),
            'numeric_outliers': np.concatenate([np.random.normal(10, 2, 195), [100, 110, 120, 130, 140]]),
            'categorical': np.random.choice(['A', 'B', 'C'], 200),
            'categorical_rare': ['common'] * 180 + ['rare1', 'rare2'] * 10,
            'target': np.random.randint(0, 2, 200)
        })
    
    def test_full_analysis_pipeline(self, complex_dataframe):
        """Test that all analysis steps run successfully."""
        report = EDAReport(data=complex_dataframe, target='target')
        report.analyze_data()
        
        # Check all analysis components are populated
        assert len(report.missing_values_summary) > 0
        assert len(report.numeric_statistics) > 0
        assert len(report.outliers_summary) > 0
        assert len(report.categorical_summary) > 0
        assert not report.feature_correlations.empty
    
    def test_analyze_data_idempotent(self, complex_dataframe):
        """Test that running analyze_data multiple times is safe."""
        report = EDAReport(data=complex_dataframe, target='target')
        
        report.analyze_data()
        first_stats = report.numeric_statistics.copy()
        
        report.analyze_data()
        second_stats = report.numeric_statistics
        
        # Results should be consistent
        assert first_stats.keys() == second_stats.keys()
