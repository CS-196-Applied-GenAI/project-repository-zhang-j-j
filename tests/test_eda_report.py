"""Unit tests for EDAReport class - Step 2: Basic Class Skeleton."""

import pytest
import pandas as pd
from quickeda.eda_report import EDAReport


class TestEDAReportBasics:
    """Test basic EDAReport class functionality."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a simple test DataFrame."""
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_eda_report_initialization_with_dataframe(self, sample_dataframe):
        """Test that EDAReport can be initialized with a DataFrame."""
        report = EDAReport(data=sample_dataframe)
        assert report is not None
        assert isinstance(report, EDAReport)
    
    def test_eda_report_initialization_with_target(self, sample_dataframe):
        """Test that EDAReport can be initialized with a target column."""
        report = EDAReport(data=sample_dataframe, target='target')
        assert report is not None
        assert isinstance(report, EDAReport)
    
    def test_eda_report_initialization_with_problem_type(self, sample_dataframe):
        """Test that EDAReport accepts problem_type parameter."""
        report = EDAReport(
            data=sample_dataframe,
            target='target',
            problem_type='classification'
        )
        assert report is not None
        assert isinstance(report, EDAReport)
    
    def test_eda_report_initialization_with_config(self, sample_dataframe):
        """Test that EDAReport accepts config dictionary."""
        config = {
            'problem_type': 'regression',
            'random_seed': 42,
            'train_test_split_ratio': 0.8
        }
        report = EDAReport(data=sample_dataframe, config=config)
        assert report is not None
        assert isinstance(report, EDAReport)
    
    def test_eda_report_has_required_methods(self, sample_dataframe):
        """Test that EDAReport has all required methods."""
        report = EDAReport(data=sample_dataframe)
        assert hasattr(report, 'analyze_data')
        assert hasattr(report, 'train_baseline_models')
        assert hasattr(report, 'generate_report')
        assert callable(report.analyze_data)
        assert callable(report.train_baseline_models)
        assert callable(report.generate_report)
    
    def test_eda_report_with_kwargs(self, sample_dataframe):
        """Test that EDAReport accepts keyword arguments."""
        report = EDAReport(
            data=sample_dataframe,
            target='target',
            problem_type='classification',
            random_seed=42
        )
        assert report is not None
        assert isinstance(report, EDAReport)
