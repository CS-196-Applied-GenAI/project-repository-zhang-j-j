"""Main EDAReport class for exploratory data analysis."""

import logging
from typing import Optional, Union, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import os

# Scikit-learn imports for Phase 4
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    mean_squared_error, r2_score
)


logger = logging.getLogger(__name__)


class EDAReport:
    """
    EDAReport generates automated exploratory data analysis reports.
    
    Parameters
    ----------
    data : pd.DataFrame or str
        Either a pandas DataFrame or file path (CSV/Parquet)
    target : str, optional
        Name of the target column for supervised analysis
    problem_type : str, optional
        'classification' or 'regression'. If None, will be inferred.
    config : dict, optional
        Configuration dictionary with optional keys:
        - problem_type
        - target
        - random_seed
        - train_test_split_ratio
        - num_top_features
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, str],
        target: Optional[str] = None,
        problem_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize EDAReport with dataset and configuration."""
        logger.info("Initializing EDAReport...")
        
        # Initialize configuration
        self.config = config or {}
        self.config.update(kwargs)
        
        # Set target and problem_type, with config overrides
        if target is None:
            self.target = self.config.get('target')
        else:
            self.target = target
            
        if problem_type is None:
            self.problem_type = self.config.get('problem_type')
        else:
            self.problem_type = problem_type
        
        # Set other configuration defaults
        self.random_seed = self.config.get('random_seed', 42)
        self.train_test_split_ratio = self.config.get('train_test_split_ratio', 0.8)
        self.num_top_features = self.config.get('num_top_features', 10)
        self.missing_threshold = self.config.get('missing_threshold', 0.5)  # 50%
        
        # Initialize storage for analysis results (before validation)
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.missing_values_summary: Dict[str, float] = {}
        self.correlations: Dict[str, float] = {}
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        
        # Phase 3 analysis results
        self.numeric_statistics: Dict[str, Dict[str, float]] = {}
        self.outliers_summary: Dict[str, Dict[str, Any]] = {}
        self.categorical_summary: Dict[str, Dict[str, Any]] = {}
        self.target_correlations: Dict[str, float] = {}
        self.feature_correlations: pd.DataFrame = pd.DataFrame()
        self.high_correlations: List[Tuple[str, str, float]] = []
        
        # Phase 4 model training results
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.feature_names_original: List[str] = []
        self.feature_names_processed: List[str] = []
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        
        # Load and validate data
        self.df = self._load_data(data)
        logger.info(f"Data loaded: shape {self.df.shape}")
        
        self._validate_data()
        logger.info("Data validation passed")
    
    def _load_data(self, data: Union[pd.DataFrame, str]) -> pd.DataFrame:
        """
        Load data from DataFrame or file path.
        
        Parameters
        ----------
        data : pd.DataFrame or str
            Either a pandas DataFrame or file path (CSV/Parquet)
            
        Returns
        -------
        pd.DataFrame
            The loaded dataset
            
        Raises
        ------
        ValueError
            If file path is invalid or format is unsupported
        """
        if isinstance(data, pd.DataFrame):
            logger.info("Data loaded from DataFrame")
            return data.copy()
        
        elif isinstance(data, str):
            if not os.path.exists(data):
                raise ValueError(f"File path does not exist: {data}")
            
            file_ext = os.path.splitext(data)[1].lower()
            
            if file_ext == '.csv':
                logger.info(f"Loading CSV file: {data}")
                return pd.read_csv(data)
            
            elif file_ext == '.parquet':
                logger.info(f"Loading Parquet file: {data}")
                return pd.read_parquet(data)
            
            else:
                raise ValueError(
                    f"Unsupported file format: {file_ext}. "
                    "Supported formats: CSV, Parquet"
                )
        else:
            raise ValueError(
                "data must be either a pandas DataFrame or a file path (str)"
            )
    
    def _validate_data(self) -> None:
        """
        Validate the loaded dataset.
        
        Checks:
        - Dataset is not empty
        - Target column exists (if provided)
        - Identify numeric and categorical columns
        - Validate train/test split feasibility
        
        Raises
        ------
        ValueError
            If validation fails
        """
        # Check dataset structure
        if len(self.df.columns) == 0:
            raise ValueError("Dataset has no columns")
        
        if len(self.df) == 0:
            raise ValueError("Dataset has no rows")
        
        logger.info(f"Dataset shape: {self.df.shape}")
        
        # Check target column exists (if provided)
        if self.target is not None:
            if self.target not in self.df.columns:
                raise ValueError(
                    f"Target column '{self.target}' not found in dataset. "
                    f"Available columns: {list(self.df.columns)}"
                )
            logger.info(f"Target column '{self.target}' found")
        
        # Identify numeric and categorical columns
        self.numeric_columns = self.df.select_dtypes(
            include=['number']
        ).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(
            include=['object', 'category', 'string']
        ).columns.tolist()
        
        logger.info(f"Numeric columns: {self.numeric_columns}")
        logger.info(f"Categorical columns: {self.categorical_columns}")
        
        # Validate train/test split feasibility
        min_samples_needed = max(2, int(1 / (1 - self.train_test_split_ratio)))
        if len(self.df) < min_samples_needed:
            raise ValueError(
                f"Dataset has {len(self.df)} rows, but at least "
                f"{min_samples_needed} rows are needed for train/test split "
                f"with ratio {self.train_test_split_ratio}"
            )
    
    def analyze_data(self) -> None:
        """
        Compute statistics, correlations, missing values, outliers, and categorical issues.
        
        This method performs Phase 3 analysis:
        - Step 5: Missing data analysis
        - Step 6: Numeric feature analysis
        - Step 7: Categorical feature analysis
        - Step 8: Correlations & feature relationships
        """
        logger.info("Analyzing data...")
        
        # Step 5: Missing Data Analysis
        self._analyze_missing_data()
        logger.info("Missing data analysis complete")
        
        # Step 6: Numeric Feature Analysis
        if self.numeric_columns:
            self._analyze_numeric_features()
            logger.info("Numeric feature analysis complete")
        
        # Step 7: Categorical Feature Analysis
        if self.categorical_columns:
            self._analyze_categorical_features()
            logger.info("Categorical feature analysis complete")
        
        # Step 8: Correlations & Feature Relationships
        if self.numeric_columns:
            self._analyze_correlations()
            logger.info("Correlation analysis complete")
        
        logger.info("Data analysis complete")
    
    def _analyze_missing_data(self) -> None:
        """
        Step 5: Missing Data Analysis.
        
        Compute column-wise missing percentages and suggest handling methods.
        """
        # Calculate missing percentage for each column
        total_rows = len(self.df)
        
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            missing_pct = missing_count / total_rows if total_rows > 0 else 0
            
            if missing_count > 0:
                self.missing_values_summary[col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_pct),
                    'suggested_handling': self._suggest_missing_handling(col, missing_pct)
                }
        
        logger.info(f"Found missing values in {len(self.missing_values_summary)} columns")
    
    def _suggest_missing_handling(self, column: str, missing_pct: float) -> str:
        """
        Suggest handling method for missing values.
        
        Parameters
        ----------
        column : str
            Column name
        missing_pct : float
            Percentage of missing values (0-1)
            
        Returns
        -------
        str
            Suggested handling method
        """
        if missing_pct > self.missing_threshold:
            return "drop_column"
        elif column in self.numeric_columns:
            return "median_imputation"
        elif column in self.categorical_columns:
            return "mode_imputation"
        else:
            return "drop_rows"
    
    def _analyze_numeric_features(self) -> None:
        """
        Step 6: Numeric Feature Analysis.
        
        Compute statistics and detect outliers for numeric features.
        """
        for col in self.numeric_columns:
            if col in self.df.columns:
                # Compute basic statistics
                col_data = self.df[col].dropna()
                
                if len(col_data) > 0:
                    self.numeric_statistics[col] = {
                        'mean': float(col_data.mean()),
                        'median': float(col_data.median()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'q25': float(col_data.quantile(0.25)),
                        'q75': float(col_data.quantile(0.75))
                    }
                    
                    # Detect outliers
                    self.outliers_summary[col] = self._detect_outliers(col_data, col)
    
    def _detect_outliers(self, data: pd.Series, column_name: str) -> Dict[str, Any]:
        """
        Detect outliers using IQR method and ±3σ method.
        
        Parameters
        ----------
        data : pd.Series
            Column data (without NaN values)
        column_name : str
            Name of the column
            
        Returns
        -------
        dict
            Outlier detection results
        """
        # IQR method
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound_iqr = q1 - 1.5 * iqr
        upper_bound_iqr = q3 + 1.5 * iqr
        outliers_iqr = data[(data < lower_bound_iqr) | (data > upper_bound_iqr)]
        
        # ±3σ method
        mean = data.mean()
        std = data.std()
        lower_bound_sigma = mean - 3 * std
        upper_bound_sigma = mean + 3 * std
        outliers_sigma = data[(data < lower_bound_sigma) | (data > upper_bound_sigma)]
        
        return {
            'iqr_method': {
                'lower_bound': float(lower_bound_iqr),
                'upper_bound': float(upper_bound_iqr),
                'count': int(len(outliers_iqr)),
                'percentage': float(len(outliers_iqr) / len(data)) if len(data) > 0 else 0
            },
            'sigma_method': {
                'lower_bound': float(lower_bound_sigma),
                'upper_bound': float(upper_bound_sigma),
                'count': int(len(outliers_sigma)),
                'percentage': float(len(outliers_sigma) / len(data)) if len(data) > 0 else 0
            }
        }
    
    def _analyze_categorical_features(self) -> None:
        """
        Step 7: Categorical Feature Analysis.
        
        Summarize category counts and identify rare categories.
        """
        for col in self.categorical_columns:
            if col in self.df.columns:
                value_counts = self.df[col].value_counts()
                total_count = len(self.df[col].dropna())
                
                # Calculate proportions
                proportions = value_counts / total_count if total_count > 0 else value_counts
                
                # Identify rare categories (< 5% of data)
                rare_threshold = 0.05
                rare_categories = proportions[proportions < rare_threshold].index.tolist()
                
                self.categorical_summary[col] = {
                    'unique_count': int(self.df[col].nunique()),
                    'top_categories': value_counts.head(10).to_dict(),
                    'rare_categories': rare_categories,
                    'rare_category_count': len(rare_categories)
                }
    
    def _analyze_correlations(self) -> None:
        """
        Step 8: Correlations & Feature Relationships.
        
        Compute pairwise correlations and identify significant relationships.
        """
        # Select only numeric columns for correlation
        numeric_df = self.df[self.numeric_columns].select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            logger.warning("Not enough numeric columns for correlation analysis")
            return
        
        # Compute correlation matrix
        self.feature_correlations = numeric_df.corr()
        
        # Identify target correlations if target is specified and numeric
        if self.target and self.target in self.numeric_columns:
            target_corrs = self.feature_correlations[self.target].drop(self.target, errors='ignore')
            # Sort by absolute correlation value
            self.target_correlations = target_corrs.abs().sort_values(ascending=False).to_dict()
        
        # Identify high predictor-predictor correlations (multicollinearity)
        # Threshold for high correlation: |r| > 0.7
        correlation_threshold = 0.7
        
        for i in range(len(self.feature_correlations.columns)):
            for j in range(i + 1, len(self.feature_correlations.columns)):
                col_i = self.feature_correlations.columns[i]
                col_j = self.feature_correlations.columns[j]
                corr_value = self.feature_correlations.iloc[i, j]
                
                if abs(corr_value) > correlation_threshold:
                    self.high_correlations.append((col_i, col_j, float(corr_value)))
        
        logger.info(f"Found {len(self.high_correlations)} high correlation pairs")
    
    def train_baseline_models(self) -> None:
        """
        Perform minimal preprocessing and train linear + tree-based models.
        
        This method performs Phase 4 steps:
        - Step 9: Minimal preprocessing
        - Step 10: Model training
        - Step 11: Model evaluation
        - Step 12: Feature importance extraction
        """
        logger.info("Training baseline models...")
        
        if not self.target:
            logger.warning("No target specified - skipping model training")
            return
        
        if self.target not in self.df.columns:
            logger.error(f"Target column '{self.target}' not found in dataset")
            return
        
        # Infer problem type if not specified
        if not self.problem_type:
            self.problem_type = self._infer_problem_type()
            logger.info(f"Inferred problem type: {self.problem_type}")
        
        # Step 9: Minimal Preprocessing
        X_train, X_test, y_train, y_test = self._preprocess_data()
        logger.info("Data preprocessing complete")
        
        # Step 10: Model Training
        self._train_models(X_train, X_test, y_train, y_test)
        logger.info("Model training complete")
        
        # Step 11: Model Evaluation (already done in _train_models)
        logger.info("Model evaluation complete")
        
        # Step 12: Feature Importance (already extracted in _train_models)
        logger.info("Feature importance extraction complete")
        
        logger.info("Baseline model training complete")
    
    def _infer_problem_type(self) -> str:
        """
        Infer problem type from target column.
        
        Returns
        -------
        str
            'classification' or 'regression'
        """
        if self.target not in self.df.columns:
            return 'classification'
        
        target_data = self.df[self.target]
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(target_data):
            # Check number of unique values
            n_unique = target_data.nunique()
            n_total = len(target_data)
            
            # If fewer than 10 unique values or unique ratio < 5%, likely classification
            if n_unique < 10 or (n_unique / n_total) < 0.05:
                return 'classification'
            else:
                return 'regression'
        else:
            # Non-numeric targets are classification
            return 'classification'
    
    def _preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Step 9: Minimal Preprocessing.
        
        Preprocessing steps:
        - Separate features and target
        - Train/test split
        - Handle missing values (median for numeric, mode for categorical)
        - One-hot encode categorical variables
        - Drop constant columns
        - Scale features for linear models
        
        Returns
        -------
        tuple
            X_train, X_test, y_train, y_test (preprocessed)
        """
        # Separate features and target
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=1-self.train_test_split_ratio,
            random_state=self.random_seed,
            stratify=y if self.problem_type == 'classification' and y.nunique() > 1 else None
        )
        
        # Store original feature names
        self.feature_names_original = X_train.columns.tolist()
        
        # Separate numeric and categorical columns
        numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        
        # Handle missing values - Numeric: median imputation
        if numeric_cols:
            num_imputer = SimpleImputer(strategy='median')
            X_train_num = pd.DataFrame(
                num_imputer.fit_transform(X_train[numeric_cols]),
                columns=numeric_cols,
                index=X_train.index
            )
            X_test_num = pd.DataFrame(
                num_imputer.transform(X_test[numeric_cols]),
                columns=numeric_cols,
                index=X_test.index
            )
        else:
            X_train_num = pd.DataFrame(index=X_train.index)
            X_test_num = pd.DataFrame(index=X_test.index)
        
        # Handle missing values - Categorical: mode imputation + one-hot encoding
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_train_cat = pd.DataFrame(
                cat_imputer.fit_transform(X_train[categorical_cols]),
                columns=categorical_cols,
                index=X_train.index
            )
            X_test_cat = pd.DataFrame(
                cat_imputer.transform(X_test[categorical_cols]),
                columns=categorical_cols,
                index=X_test.index
            )
            
            # One-hot encode categorical variables
            X_train_cat_encoded = pd.get_dummies(X_train_cat, prefix=categorical_cols)
            X_test_cat_encoded = pd.get_dummies(X_test_cat, prefix=categorical_cols)
            
            # Align columns between train and test
            X_train_cat_encoded, X_test_cat_encoded = X_train_cat_encoded.align(
                X_test_cat_encoded, join='left', axis=1, fill_value=0
            )
        else:
            X_train_cat_encoded = pd.DataFrame(index=X_train.index)
            X_test_cat_encoded = pd.DataFrame(index=X_test.index)
        
        # Combine numeric and encoded categorical
        X_train_processed = pd.concat([X_train_num, X_train_cat_encoded], axis=1)
        X_test_processed = pd.concat([X_test_num, X_test_cat_encoded], axis=1)
        
        # Drop constant columns
        constant_cols = [col for col in X_train_processed.columns 
                        if X_train_processed[col].nunique() <= 1]
        if constant_cols:
            logger.info(f"Dropping {len(constant_cols)} constant columns")
            X_train_processed = X_train_processed.drop(columns=constant_cols)
            X_test_processed = X_test_processed.drop(columns=constant_cols)
        
        # Store feature names after preprocessing
        self.feature_names_processed = X_train_processed.columns.tolist()
        
        # Store preprocessed data for potential reuse
        self.X_train = X_train_processed
        self.X_test = X_test_processed
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def _train_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> None:
        """
        Step 10: Model Training and Step 11: Model Evaluation.
        
        Train linear and tree-based models, evaluate them, and extract feature importance.
        
        Parameters
        ----------
        X_train, X_test : pd.DataFrame
            Preprocessed feature matrices
        y_train, y_test : pd.Series
            Target variables
        """
        if self.problem_type == 'classification':
            # Linear model: Logistic Regression (with scaling)
            logger.info("Training Logistic Regression...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            linear_model = LogisticRegression(
                random_state=self.random_seed,
                max_iter=1000,
                n_jobs=-1
            )
            linear_model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred_train_linear = linear_model.predict(X_train_scaled)
            y_pred_test_linear = linear_model.predict(X_test_scaled)
            y_pred_proba_linear = linear_model.predict_proba(X_test_scaled)
            
            # Metrics
            self.metrics['linear_model'] = {
                'train': {
                    'accuracy': float(accuracy_score(y_train, y_pred_train_linear)),
                    'f1': float(f1_score(y_train, y_pred_train_linear, average='weighted'))
                },
                'test': {
                    'accuracy': float(accuracy_score(y_test, y_pred_test_linear)),
                    'f1': float(f1_score(y_test, y_pred_test_linear, average='weighted'))
                }
            }
            
            # ROC-AUC for binary classification
            if len(np.unique(y_train)) == 2:
                self.metrics['linear_model']['train']['roc_auc'] = float(
                    roc_auc_score(y_train, linear_model.predict_proba(X_train_scaled)[:, 1])
                )
                self.metrics['linear_model']['test']['roc_auc'] = float(
                    roc_auc_score(y_test, y_pred_proba_linear[:, 1])
                )
            
            # Feature importance (absolute coefficients)
            feature_importance_linear = np.abs(linear_model.coef_[0] if linear_model.coef_.ndim > 1 else linear_model.coef_)
            
            # Tree model: Random Forest
            logger.info("Training Random Forest Classifier...")
            tree_model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_seed,
                n_jobs=-1,
                max_depth=10
            )
            tree_model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train_tree = tree_model.predict(X_train)
            y_pred_test_tree = tree_model.predict(X_test)
            y_pred_proba_tree = tree_model.predict_proba(X_test)
            
            # Metrics
            self.metrics['tree_model'] = {
                'train': {
                    'accuracy': float(accuracy_score(y_train, y_pred_train_tree)),
                    'f1': float(f1_score(y_train, y_pred_train_tree, average='weighted'))
                },
                'test': {
                    'accuracy': float(accuracy_score(y_test, y_pred_test_tree)),
                    'f1': float(f1_score(y_test, y_pred_test_tree, average='weighted'))
                }
            }
            
            # ROC-AUC for binary classification
            if len(np.unique(y_train)) == 2:
                self.metrics['tree_model']['train']['roc_auc'] = float(
                    roc_auc_score(y_train, tree_model.predict_proba(X_train)[:, 1])
                )
                self.metrics['tree_model']['test']['roc_auc'] = float(
                    roc_auc_score(y_test, y_pred_proba_tree[:, 1])
                )
            
            # Feature importance
            feature_importance_tree = tree_model.feature_importances_
            
            # Store models
            self.models['linear_model'] = linear_model
            self.models['linear_scaler'] = scaler
            self.models['tree_model'] = tree_model
            
        else:  # regression
            # Linear model: Linear Regression (with scaling)
            logger.info("Training Linear Regression...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            linear_model = LinearRegression(n_jobs=-1)
            linear_model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred_train_linear = linear_model.predict(X_train_scaled)
            y_pred_test_linear = linear_model.predict(X_test_scaled)
            
            # Metrics
            self.metrics['linear_model'] = {
                'train': {
                    'rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train_linear))),
                    'r2': float(r2_score(y_train, y_pred_train_linear))
                },
                'test': {
                    'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test_linear))),
                    'r2': float(r2_score(y_test, y_pred_test_linear))
                }
            }
            
            # Feature importance (absolute coefficients)
            feature_importance_linear = np.abs(linear_model.coef_)
            
            # Tree model: Random Forest
            logger.info("Training Random Forest Regressor...")
            tree_model = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_seed,
                n_jobs=-1,
                max_depth=10
            )
            tree_model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train_tree = tree_model.predict(X_train)
            y_pred_test_tree = tree_model.predict(X_test)
            
            # Metrics
            self.metrics['tree_model'] = {
                'train': {
                    'rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train_tree))),
                    'r2': float(r2_score(y_train, y_pred_train_tree))
                },
                'test': {
                    'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test_tree))),
                    'r2': float(r2_score(y_test, y_pred_test_tree))
                }
            }
            
            # Feature importance
            feature_importance_tree = tree_model.feature_importances_
            
            # Store models
            self.models['linear_model'] = linear_model
            self.models['linear_scaler'] = scaler
            self.models['tree_model'] = tree_model
        
        # Step 12: Extract and store feature importance
        self._extract_feature_importance(feature_importance_linear, feature_importance_tree, X_train.columns)
    
    def _extract_feature_importance(
        self,
        linear_importance: np.ndarray,
        tree_importance: np.ndarray,
        feature_names: pd.Index
    ) -> None:
        """
        Step 12: Feature Importance extraction.
        
        Extract and rank feature importance from both models.
        
        Parameters
        ----------
        linear_importance : np.ndarray
            Feature importance from linear model
        tree_importance : np.ndarray
            Feature importance from tree model
        feature_names : pd.Index
            Names of features
        """
        # Linear model feature importance
        linear_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': linear_importance
        }).sort_values('importance', ascending=False)
        
        self.models['linear_feature_importance'] = linear_importance_df.to_dict('records')
        
        # Tree model feature importance
        tree_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': tree_importance
        }).sort_values('importance', ascending=False)
        
        self.models['tree_feature_importance'] = tree_importance_df.to_dict('records')
        
        logger.info(f"Feature importance extracted for {len(feature_names)} features")
    
    def generate_report(self, output_path: str) -> None:
        """
        Generate the HTML report with textual summaries and key static plots.
        
        Parameters
        ----------
        output_path : str
            Path to save the HTML report
        """
        logger.info(f"Generating report to {output_path}...")
        # TODO: Implement generate_report
        pass
