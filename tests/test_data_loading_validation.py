"""Unit tests for Phase 2: Data Input & Validation (Steps 3 & 4)."""

import pytest
import pandas as pd
import tempfile
import os
from quickeda.eda_report import EDAReport


class TestDataLoading:
    """Test Step 3: Data Loading functionality."""
    
    @pytest.fixture
    def sample_csv_file(self):
        """Create a temporary CSV file for testing."""
        df = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [10.5, 20.5, 30.5, 40.5, 50.5],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 0]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f, index=False)
            temp_path = f.name
        
        yield temp_path
        os.remove(temp_path)
    
    @pytest.fixture
    def sample_parquet_file(self):
        """Create a temporary Parquet file for testing."""
        df = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [10.5, 20.5, 30.5, 40.5, 50.5],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 0]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = f.name
        
        df.to_parquet(temp_path, index=False)
        yield temp_path
        os.remove(temp_path)
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame."""
        return pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [10.5, 20.5, 30.5, 40.5, 50.5],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_load_data_from_dataframe(self, sample_dataframe):
        """Test loading data from a pandas DataFrame."""
        report = EDAReport(data=sample_dataframe)
        assert isinstance(report.df, pd.DataFrame)
        assert report.df.shape == (5, 4)
        assert list(report.df.columns) == ['numeric1', 'numeric2', 'categorical', 'target']
    
    def test_load_data_from_csv(self, sample_csv_file):
        """Test loading data from a CSV file."""
        report = EDAReport(data=sample_csv_file)
        assert isinstance(report.df, pd.DataFrame)
        assert report.df.shape == (5, 4)
    
    def test_load_data_from_parquet(self, sample_parquet_file):
        """Test loading data from a Parquet file."""
        report = EDAReport(data=sample_parquet_file)
        assert isinstance(report.df, pd.DataFrame)
        assert report.df.shape == (5, 4)
    
    def test_load_data_invalid_path(self):
        """Test loading data from non-existent file path."""
        with pytest.raises(ValueError, match="File path does not exist"):
            EDAReport(data="/nonexistent/path/file.csv")
    
    def test_load_data_unsupported_format(self, tmp_path):
        """Test loading data from unsupported file format."""
        unsupported_file = tmp_path / "file.xlsx"
        unsupported_file.write_text("dummy")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            EDAReport(data=str(unsupported_file))
    
    def test_load_data_invalid_type(self):
        """Test loading data with invalid type."""
        with pytest.raises(ValueError, match="data must be either"):
            EDAReport(data=123)
    
    def test_load_data_creates_copy(self, sample_dataframe):
        """Test that loaded DataFrame is a copy, not a reference."""
        original = sample_dataframe.copy()
        report = EDAReport(data=sample_dataframe)
        
        # Modify original
        sample_dataframe.loc[0, 'numeric1'] = 999
        
        # Verify report's df is not affected
        assert report.df.loc[0, 'numeric1'] == 1


class TestBasicValidation:
    """Test Step 4: Basic Validation functionality."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame."""
        return pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [10.5, 20.5, 30.5, 40.5, 50.5],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_validate_non_empty_dataset(self, sample_dataframe):
        """Test that non-empty dataset passes validation."""
        report = EDAReport(data=sample_dataframe)
        assert not report.df.empty
        assert len(report.df) > 0
    
    def test_validate_empty_dataset(self):
        """Test that empty dataset raises ValueError."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Dataset has no columns"):
            EDAReport(data=empty_df)
    
    def test_validate_target_column_exists(self, sample_dataframe):
        """Test that valid target column is accepted."""
        report = EDAReport(data=sample_dataframe, target='target')
        assert report.target == 'target'
    
    def test_validate_target_column_not_exists(self, sample_dataframe):
        """Test that invalid target column raises ValueError."""
        with pytest.raises(ValueError, match="Target column.*not found"):
            EDAReport(data=sample_dataframe, target='nonexistent_column')
    
    def test_validate_numeric_column_detection(self, sample_dataframe):
        """Test that numeric columns are correctly identified."""
        report = EDAReport(data=sample_dataframe)
        # Note: target column is also numeric, so it should be included
        assert set(report.numeric_columns) == {'numeric1', 'numeric2', 'target'}
    
    def test_validate_categorical_column_detection(self, sample_dataframe):
        """Test that categorical columns are correctly identified."""
        report = EDAReport(data=sample_dataframe)
        assert report.categorical_columns == ['categorical']
    
    def test_validate_mixed_column_types(self):
        """Test validation with mixed data types."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e'],
            'bool_col': [True, False, True, False, True]
        })
        
        report = EDAReport(data=df)
        assert 'int_col' in report.numeric_columns
        assert 'float_col' in report.numeric_columns
        assert 'str_col' in report.categorical_columns
    
    def test_validate_train_test_split_feasibility(self, sample_dataframe):
        """Test that valid dataset passes train/test split validation."""
        report = EDAReport(data=sample_dataframe, config={'train_test_split_ratio': 0.8})
        # Should not raise any exception
        assert report.train_test_split_ratio == 0.8
    
    def test_validate_train_test_split_too_small_dataset(self):
        """Test that dataset too small for train/test split raises ValueError."""
        tiny_df = pd.DataFrame({
            'x': [1]
        })
        
        with pytest.raises(ValueError, match="Dataset has.*rows.*needed"):
            EDAReport(data=tiny_df, config={'train_test_split_ratio': 0.95})
    
    def test_validate_no_rows(self):
        """Test that dataset with no rows raises ValueError."""
        df = pd.DataFrame(columns=['a', 'b', 'c'])
        
        with pytest.raises(ValueError, match="Dataset has no rows"):
            EDAReport(data=df)
    
    def test_validate_no_columns(self):
        """Test that dataset with no columns raises ValueError."""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Dataset has no columns"):
            EDAReport(data=df)


class TestConfigurationHandling:
    """Test configuration and initialization parameters."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame."""
        return pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_configuration_defaults(self, sample_dataframe):
        """Test that default configuration values are set."""
        report = EDAReport(data=sample_dataframe)
        assert report.random_seed == 42
        assert report.train_test_split_ratio == 0.8
        assert report.num_top_features == 10
    
    def test_configuration_from_config_dict(self, sample_dataframe):
        """Test that configuration can be set via config dictionary."""
        config = {
            'random_seed': 123,
            'train_test_split_ratio': 0.7,
            'num_top_features': 5
        }
        report = EDAReport(data=sample_dataframe, config=config)
        assert report.random_seed == 123
        assert report.train_test_split_ratio == 0.7
        assert report.num_top_features == 5
    
    def test_configuration_from_kwargs(self, sample_dataframe):
        """Test that configuration can be set via keyword arguments."""
        report = EDAReport(
            data=sample_dataframe,
            random_seed=456,
            train_test_split_ratio=0.75
        )
        assert report.random_seed == 456
        assert report.train_test_split_ratio == 0.75
    
    def test_configuration_parameter_priority(self, sample_dataframe):
        """Test that explicit parameters override config dictionary."""
        config = {'target': 'wrong_target'}
        report = EDAReport(
            data=sample_dataframe,
            target='target',
            config=config
        )
        assert report.target == 'target'
    
    def test_problem_type_configuration(self, sample_dataframe):
        """Test problem_type configuration."""
        report = EDAReport(
            data=sample_dataframe,
            problem_type='classification'
        )
        assert report.problem_type == 'classification'
    
    def test_initialization_creates_storage_attributes(self, sample_dataframe):
        """Test that initialization creates necessary storage attributes."""
        report = EDAReport(data=sample_dataframe)
        assert hasattr(report, 'missing_values_summary')
        assert hasattr(report, 'correlations')
        assert hasattr(report, 'models')
        assert hasattr(report, 'metrics')
        assert isinstance(report.missing_values_summary, dict)
        assert isinstance(report.correlations, dict)
        assert isinstance(report.models, dict)
        assert isinstance(report.metrics, dict)
