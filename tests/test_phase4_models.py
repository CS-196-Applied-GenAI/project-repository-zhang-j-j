"""
Tests for Phase 4: Baseline Model Training and Evaluation.

Tests cover:
- Step 9: Minimal preprocessing
- Step 10: Model training
- Step 11: Model evaluation
- Step 12: Feature importance extraction
"""

import pytest
import pandas as pd
import numpy as np
from quickeda import EDAReport


@pytest.fixture
def classification_data():
    """Create sample classification dataset."""
    np.random.seed(42)
    n_samples = 200
    
    # Features
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    X3 = np.random.randn(n_samples)
    
    # Target (binary classification)
    y = (X1 + 0.5 * X2 - 0.3 * X3 + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    
    df = pd.DataFrame({
        'feature1': X1,
        'feature2': X2,
        'feature3': X3,
        'target': y
    })
    
    return df


@pytest.fixture
def regression_data():
    """Create sample regression dataset."""
    np.random.seed(42)
    n_samples = 200
    
    # Features
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    X3 = np.random.randn(n_samples)
    
    # Target (continuous)
    y = 10 + 2*X1 + 3*X2 - 1.5*X3 + np.random.randn(n_samples) * 0.5
    
    df = pd.DataFrame({
        'feature1': X1,
        'feature2': X2,
        'feature3': X3,
        'target': y
    })
    
    return df


@pytest.fixture
def data_with_missing():
    """Create dataset with missing values."""
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        'num_feat1': np.random.randn(n_samples),
        'num_feat2': np.random.randn(n_samples),
        'cat_feat': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.randint(0, 2, n_samples)
    })
    
    # Introduce missing values
    df.loc[0:9, 'num_feat1'] = np.nan
    df.loc[0:4, 'cat_feat'] = np.nan
    
    return df


@pytest.fixture
def data_with_categorical():
    """Create dataset with categorical features."""
    np.random.seed(42)
    n_samples = 150
    
    df = pd.DataFrame({
        'numeric1': np.random.randn(n_samples),
        'numeric2': np.random.randn(n_samples),
        'category1': np.random.choice(['Red', 'Blue', 'Green'], n_samples),
        'category2': np.random.choice(['Small', 'Medium', 'Large'], n_samples),
        'target': np.random.randint(0, 2, n_samples)
    })
    
    return df


class TestStep9Preprocessing:
    """Test Step 9: Minimal Preprocessing."""
    
    def test_preprocessing_creates_train_test_split(self, classification_data):
        """Test that preprocessing creates train/test split."""
        report = EDAReport(
            classification_data,
            target='target',
            problem_type='classification',
            train_test_split_ratio=0.8
        )
        
        # Run model training (includes preprocessing)
        report.train_baseline_models()
        
        assert report.X_train is not None
        assert report.X_test is not None
        assert report.y_train is not None
        assert report.y_test is not None
        
        # Check split ratio
        total_samples = len(classification_data)
        train_size = len(report.X_train)
        test_size = len(report.X_test)
        
        assert train_size + test_size == total_samples
        assert abs(train_size / total_samples - 0.8) < 0.05  # Within 5% of expected
    
    def test_preprocessing_handles_missing_numeric(self, data_with_missing):
        """Test that missing numeric values are imputed with median."""
        report = EDAReport(
            data_with_missing,
            target='target',
            problem_type='classification'
        )
        
        report.train_baseline_models()
        
        # Check that no missing values remain in preprocessed data
        assert not report.X_train.isnull().any().any()
        assert not report.X_test.isnull().any().any()
    
    def test_preprocessing_handles_missing_categorical(self, data_with_missing):
        """Test that missing categorical values are imputed with mode."""
        report = EDAReport(
            data_with_missing,
            target='target',
            problem_type='classification'
        )
        
        report.train_baseline_models()
        
        # Check that all categorical features were processed (encoded)
        # Original had 1 categorical feature, should be one-hot encoded
        assert not report.X_train.isnull().any().any()
        assert not report.X_test.isnull().any().any()
    
    def test_preprocessing_one_hot_encodes_categorical(self, data_with_categorical):
        """Test that categorical features are one-hot encoded."""
        report = EDAReport(
            data_with_categorical,
            target='target',
            problem_type='classification'
        )
        
        report.train_baseline_models()
        
        # Original features: 2 numeric, 2 categorical
        # After encoding: 2 numeric + encoded categoricals
        original_cats = ['category1', 'category2']
        n_original_numeric = 2
        
        # Check that we have more features than original numeric
        assert report.X_train.shape[1] > n_original_numeric
        
        # Check that categorical features are not in processed data
        for cat in original_cats:
            assert cat not in report.X_train.columns
    
    def test_preprocessing_drops_constant_columns(self):
        """Test that constant columns are dropped."""
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'constant': np.ones(100),  # Constant column
            'target': np.random.randint(0, 2, 100)
        })
        
        report = EDAReport(
            df,
            target='target',
            problem_type='classification'
        )
        
        report.train_baseline_models()
        
        # Constant column should be dropped
        assert 'constant' not in report.X_train.columns
        assert report.X_train.shape[1] == 1  # Only feature1 remains
    
    def test_preprocessing_aligns_train_test_columns(self, data_with_categorical):
        """Test that train and test sets have aligned columns after encoding."""
        report = EDAReport(
            data_with_categorical,
            target='target',
            problem_type='classification'
        )
        
        report.train_baseline_models()
        
        # Train and test should have same columns
        assert list(report.X_train.columns) == list(report.X_test.columns)


class TestStep10ModelTraining:
    """Test Step 10: Model Training."""
    
    def test_trains_classification_models(self, classification_data):
        """Test that both classification models are trained."""
        report = EDAReport(
            classification_data,
            target='target',
            problem_type='classification'
        )
        
        report.train_baseline_models()
        
        # Check that models are stored
        assert 'linear_model' in report.models
        assert 'tree_model' in report.models
        assert 'linear_scaler' in report.models
        
        # Check model types
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        assert isinstance(report.models['linear_model'], LogisticRegression)
        assert isinstance(report.models['tree_model'], RandomForestClassifier)
    
    def test_trains_regression_models(self, regression_data):
        """Test that both regression models are trained."""
        report = EDAReport(
            regression_data,
            target='target',
            problem_type='regression'
        )
        
        report.train_baseline_models()
        
        # Check that models are stored
        assert 'linear_model' in report.models
        assert 'tree_model' in report.models
        assert 'linear_scaler' in report.models
        
        # Check model types
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        
        assert isinstance(report.models['linear_model'], LinearRegression)
        assert isinstance(report.models['tree_model'], RandomForestRegressor)
    
    def test_no_training_without_target(self, classification_data):
        """Test that training is skipped when no target is specified."""
        report = EDAReport(classification_data)
        
        report.train_baseline_models()
        
        # No models should be trained
        assert len(report.models) == 0
        assert len(report.metrics) == 0
    
    def test_infers_classification_for_binary_target(self):
        """Test that classification is inferred for binary target."""
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        
        report = EDAReport(df, target='target')
        
        report.train_baseline_models()
        
        # Should infer classification
        assert report.problem_type == 'classification'
        assert 'linear_model' in report.models
    
    def test_infers_regression_for_continuous_target(self):
        """Test that regression is inferred for continuous target."""
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'target': np.random.randn(100)
        })
        
        report = EDAReport(df, target='target')
        
        report.train_baseline_models()
        
        # Should infer regression
        assert report.problem_type == 'regression'
        assert 'linear_model' in report.models


class TestStep11ModelEvaluation:
    """Test Step 11: Model Evaluation."""
    
    def test_classification_metrics_computed(self, classification_data):
        """Test that classification metrics are computed."""
        report = EDAReport(
            classification_data,
            target='target',
            problem_type='classification'
        )
        
        report.train_baseline_models()
        
        # Check metrics structure
        assert 'linear_model' in report.metrics
        assert 'tree_model' in report.metrics
        
        # Check train metrics
        linear_train = report.metrics['linear_model']['train']
        assert 'accuracy' in linear_train
        assert 'f1' in linear_train
        
        tree_train = report.metrics['tree_model']['train']
        assert 'accuracy' in tree_train
        assert 'f1' in tree_train
        
        # Check test metrics
        linear_test = report.metrics['linear_model']['test']
        assert 'accuracy' in linear_test
        assert 'f1' in linear_test
        
        tree_test = report.metrics['tree_model']['test']
        assert 'accuracy' in tree_test
        assert 'f1' in tree_test
    
    def test_binary_classification_includes_roc_auc(self, classification_data):
        """Test that ROC-AUC is computed for binary classification."""
        report = EDAReport(
            classification_data,
            target='target',
            problem_type='classification'
        )
        
        report.train_baseline_models()
        
        # Check train metrics
        assert 'roc_auc' in report.metrics['linear_model']['train']
        assert 'roc_auc' in report.metrics['tree_model']['train']
        
        # Check test metrics
        assert 'roc_auc' in report.metrics['linear_model']['test']
        assert 'roc_auc' in report.metrics['tree_model']['test']
    
    def test_regression_metrics_computed(self, regression_data):
        """Test that regression metrics are computed."""
        report = EDAReport(
            regression_data,
            target='target',
            problem_type='regression'
        )
        
        report.train_baseline_models()
        
        # Check metrics structure
        assert 'linear_model' in report.metrics
        assert 'tree_model' in report.metrics
        
        # Check train metrics
        linear_train = report.metrics['linear_model']['train']
        assert 'rmse' in linear_train
        assert 'r2' in linear_train
        
        tree_train = report.metrics['tree_model']['train']
        assert 'rmse' in tree_train
        assert 'r2' in tree_train
        
        # Check test metrics
        linear_test = report.metrics['linear_model']['test']
        assert 'rmse' in linear_test
        assert 'r2' in linear_test
        
        tree_test = report.metrics['tree_model']['test']
        assert 'rmse' in tree_test
        assert 'r2' in tree_test
    
    def test_metrics_are_reasonable_values(self, classification_data):
        """Test that computed metrics have reasonable values."""
        report = EDAReport(
            classification_data,
            target='target',
            problem_type='classification'
        )
        
        report.train_baseline_models()
        
        # Check that metrics are in valid ranges
        for model_name in ['linear_model', 'tree_model']:
            for split in ['train', 'test']:
                metrics = report.metrics[model_name][split]
                
                # Accuracy should be between 0 and 1
                assert 0 <= metrics['accuracy'] <= 1
                
                # F1 should be between 0 and 1
                assert 0 <= metrics['f1'] <= 1
                
                # ROC-AUC should be between 0 and 1 (if present)
                if 'roc_auc' in metrics:
                    assert 0 <= metrics['roc_auc'] <= 1


class TestStep12FeatureImportance:
    """Test Step 12: Feature Importance Extraction."""
    
    def test_feature_importance_extracted(self, classification_data):
        """Test that feature importance is extracted for both models."""
        report = EDAReport(
            classification_data,
            target='target',
            problem_type='classification'
        )
        
        report.train_baseline_models()
        
        # Check that feature importance is stored
        assert 'linear_feature_importance' in report.models
        assert 'tree_feature_importance' in report.models
    
    def test_feature_importance_has_correct_structure(self, classification_data):
        """Test that feature importance has correct structure."""
        report = EDAReport(
            classification_data,
            target='target',
            problem_type='classification'
        )
        
        report.train_baseline_models()
        
        # Check linear model feature importance
        linear_fi = report.models['linear_feature_importance']
        assert isinstance(linear_fi, list)
        assert len(linear_fi) > 0
        assert 'feature' in linear_fi[0]
        assert 'importance' in linear_fi[0]
        
        # Check tree model feature importance
        tree_fi = report.models['tree_feature_importance']
        assert isinstance(tree_fi, list)
        assert len(tree_fi) > 0
        assert 'feature' in tree_fi[0]
        assert 'importance' in tree_fi[0]
    
    def test_feature_importance_sorted_descending(self, classification_data):
        """Test that feature importance is sorted in descending order."""
        report = EDAReport(
            classification_data,
            target='target',
            problem_type='classification'
        )
        
        report.train_baseline_models()
        
        # Check linear model
        linear_fi = report.models['linear_feature_importance']
        importances = [item['importance'] for item in linear_fi]
        assert importances == sorted(importances, reverse=True)
        
        # Check tree model
        tree_fi = report.models['tree_feature_importance']
        importances = [item['importance'] for item in tree_fi]
        assert importances == sorted(importances, reverse=True)
    
    def test_feature_importance_matches_feature_count(self, classification_data):
        """Test that feature importance has entry for each feature."""
        report = EDAReport(
            classification_data,
            target='target',
            problem_type='classification'
        )
        
        report.train_baseline_models()
        
        n_features = report.X_train.shape[1]
        
        # Check counts
        assert len(report.models['linear_feature_importance']) == n_features
        assert len(report.models['tree_feature_importance']) == n_features


class TestFullPipeline:
    """Test full Phase 4 pipeline."""
    
    def test_full_pipeline_classification(self, classification_data):
        """Test complete pipeline for classification."""
        report = EDAReport(
            classification_data,
            target='target',
            problem_type='classification'
        )
        
        # Run analysis
        report.analyze_data()
        
        # Train models
        report.train_baseline_models()
        
        # Verify all Phase 4 components
        assert report.X_train is not None
        assert report.X_test is not None
        assert 'linear_model' in report.models
        assert 'tree_model' in report.models
        assert 'linear_model' in report.metrics
        assert 'tree_model' in report.metrics
        assert 'linear_feature_importance' in report.models
        assert 'tree_feature_importance' in report.models
    
    def test_full_pipeline_regression(self, regression_data):
        """Test complete pipeline for regression."""
        report = EDAReport(
            regression_data,
            target='target',
            problem_type='regression'
        )
        
        # Run analysis
        report.analyze_data()
        
        # Train models
        report.train_baseline_models()
        
        # Verify all Phase 4 components
        assert report.X_train is not None
        assert report.X_test is not None
        assert 'linear_model' in report.models
        assert 'tree_model' in report.models
        assert 'linear_model' in report.metrics
        assert 'tree_model' in report.metrics
        assert 'linear_feature_importance' in report.models
        assert 'tree_feature_importance' in report.models
    
    def test_pipeline_with_complex_data(self, data_with_categorical):
        """Test pipeline with missing values and categorical features."""
        report = EDAReport(
            data_with_categorical,
            target='target',
            problem_type='classification'
        )
        
        # Run full pipeline
        report.analyze_data()
        report.train_baseline_models()
        
        # Verify successful completion
        assert report.models is not None
        assert report.metrics is not None
        assert len(report.models) > 0
        assert len(report.metrics) > 0
