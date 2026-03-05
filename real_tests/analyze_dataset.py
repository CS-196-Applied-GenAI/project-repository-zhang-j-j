"""
Example usage of QuickEDA for exploratory data analysis.

This script demonstrates how to use the EDAReport class to analyze
any CSV dataset with command-line arguments or interactive prompts.

Usage:
    python analyze_dataset.py [file_path] [--target TARGET] [--problem-type TYPE]
    
    If file_path is not provided, you will be prompted to enter it.

Examples:
    python analyze_dataset.py creditcard.csv --target Class --problem-type classification
    python analyze_dataset.py diabetes.csv --target Outcome
    python analyze_dataset.py  # Will prompt for inputs
"""

import sys
import os
import argparse

# Add parent directory to path to import quickeda
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quickeda import EDAReport
import pandas as pd


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run QuickEDA analysis on a CSV dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s creditcard.csv --target Class --problem-type classification
  %(prog)s diabetes.csv --target Outcome
  %(prog)s  # Will prompt for file path
        """
    )
    
    parser.add_argument(
        'file_path',
        nargs='?',  # Make file_path optional
        help='Path to the CSV file to analyze'
    )
    
    parser.add_argument(
        '--target',
        '-t',
        help='Name of the target column for supervised analysis',
        default=None
    )
    
    parser.add_argument(
        '--problem-type',
        '-p',
        choices=['classification', 'regression'],
        help='Type of ML problem (classification or regression)',
        default=None
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--top-features',
        type=int,
        default=10,
        help='Number of top features to highlight (default: 10)'
    )
    
    return parser.parse_args()


def prompt_for_file():
    """Prompt user for file path."""
    print()
    print("=" * 80)
    print("QuickEDA Interactive Mode")
    print("=" * 80)
    print()
    
    while True:
        file_path = input("Enter the path to your CSV file: ").strip()
        
        if not file_path:
            print("ERROR: File path cannot be empty.")
            continue
        
        # Resolve file path
        if os.path.isabs(file_path):
            data_path = file_path
        else:
            # Try relative to script directory first
            script_dir_path = os.path.join(os.path.dirname(__file__), file_path)
            if os.path.exists(script_dir_path):
                data_path = script_dir_path
            else:
                # Try relative to current working directory
                data_path = file_path
        
        # Check if file exists
        if not os.path.exists(data_path):
            print(f"ERROR: File not found: {file_path}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                sys.exit(1)
            continue
        
        return data_path


def prompt_for_target(df):
    """Prompt user for target column."""
    print()
    print(f"Dataset has {len(df.columns)} columns:")
    print(", ".join(df.columns.tolist()))
    print()
    
    target = input("Enter target column name (or press Enter to skip): ").strip()
    
    if not target:
        return None
    
    if target not in df.columns:
        print(f"WARNING: Column '{target}' not found in dataset.")
        use_anyway = input("Continue without target? (y/n): ").strip().lower()
        if use_anyway == 'y':
            return None
        else:
            return prompt_for_target(df)
    
    return target


def prompt_for_problem_type():
    """Prompt user for problem type."""
    print()
    problem_type = input("Problem type (classification/regression/skip): ").strip().lower()
    
    if problem_type in ['classification', 'regression']:
        return problem_type
    
    return None


def main():
    """Run EDA analysis on the specified dataset."""
    
    # Parse command-line arguments
    args = parse_args()
    
    # Get file path (from args or prompt)
    if args.file_path:
        # File path provided via command line
        if os.path.isabs(args.file_path):
            data_path = args.file_path
        else:
            # Try relative to script directory first
            script_dir_path = os.path.join(os.path.dirname(__file__), args.file_path)
            if os.path.exists(script_dir_path):
                data_path = script_dir_path
            else:
                # Try relative to current working directory
                data_path = args.file_path
        
        # Check if file exists
        if not os.path.exists(data_path):
            print(f"ERROR: File not found: {args.file_path}")
            print(f"Searched at: {data_path}")
            sys.exit(1)
    else:
        # Prompt for file path
        data_path = prompt_for_file()
    
    dataset_name = os.path.basename(data_path)
    
    # Load dataset to check columns (if we need to prompt for target)
    target = args.target
    problem_type = args.problem_type
    
    if not target:
        # Load dataset to show columns
        try:
            temp_df = pd.read_csv(data_path)
            target = prompt_for_target(temp_df)
            
            # If target was specified via prompt, ask for problem type
            if target and not problem_type:
                problem_type = prompt_for_problem_type()
        except Exception as e:
            print(f"ERROR loading dataset: {e}")
            sys.exit(1)
    
    print("=" * 80)
    print(f"QuickEDA Analysis: {dataset_name}")
    print("=" * 80)
    print()
    
    # Load and initialize the EDA report
    print("Step 1: Loading data and initializing EDAReport...")
    print(f"  File: {data_path}")
    
    # Create EDA report with configuration
    report = EDAReport(
        data=data_path,
        target=target,
        problem_type=problem_type,
        random_seed=args.seed,
        train_test_split_ratio=0.8,
        num_top_features=args.top_features
    )
    
    print(f"✓ Data loaded successfully!")
    print(f"  - Dataset shape: {report.df.shape}")
    print(f"  - Target column: {report.target if report.target else 'None (unsupervised)'}")
    print(f"  - Problem type: {report.problem_type if report.problem_type else 'Not specified'}")
    print(f"  - Numeric columns: {len(report.numeric_columns)}")
    print(f"  - Categorical columns: {len(report.categorical_columns)}")
    print()
    
    # Run data analysis (Phase 3)
    print("Step 2: Running data analysis...")
    report.analyze_data()
    print("✓ Analysis complete!")
    print()
    
    # Display missing data analysis results
    print("-" * 80)
    print("Missing Data Analysis (Step 5)")
    print("-" * 80)
    if report.missing_values_summary:
        for col, info in report.missing_values_summary.items():
            print(f"  {col}:")
            print(f"    - Missing count: {info['count']}")
            print(f"    - Missing percentage: {info['percentage']:.2%}")
            print(f"    - Suggested handling: {info['suggested_handling']}")
    else:
        print("  ✓ No missing values detected!")
    print()
    
    # Display numeric feature statistics
    print("-" * 80)
    print("Numeric Feature Analysis (Step 6)")
    print("-" * 80)
    print(f"Analyzing {len(report.numeric_statistics)} numeric features...")
    print()
    
    # Show statistics for a few key features
    features_to_show = ['Time', 'Amount', 'Class']
    for feature in features_to_show:
        if feature in report.numeric_statistics:
            stats = report.numeric_statistics[feature]
            print(f"  {feature}:")
            print(f"    - Mean: {stats['mean']:.2f}")
            print(f"    - Median: {stats['median']:.2f}")
            print(f"    - Std: {stats['std']:.2f}")
            print(f"    - Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
            
            # Show outlier information
            if feature in report.outliers_summary:
                outlier_info = report.outliers_summary[feature]
                iqr_outliers = outlier_info['iqr_method']['count']
                sigma_outliers = outlier_info['sigma_method']['count']
                print(f"    - Outliers (IQR): {iqr_outliers} ({outlier_info['iqr_method']['percentage']:.2%})")
                print(f"    - Outliers (±3σ): {sigma_outliers} ({outlier_info['sigma_method']['percentage']:.2%})")
            print()
    
    # Display categorical feature analysis
    print("-" * 80)
    print("Categorical Feature Analysis (Step 7)")
    print("-" * 80)
    if report.categorical_summary:
        for col, info in report.categorical_summary.items():
            print(f"  {col}:")
            print(f"    - Unique values: {info['unique_count']}")
            print(f"    - Rare categories: {info['rare_category_count']}")
            print()
    else:
        print("  No categorical features detected.")
    print()
    
    # Display correlation analysis
    print("-" * 80)
    print("Correlation Analysis (Step 8)")
    print("-" * 80)
    
    # Target correlations
    if report.target_correlations and report.target:
        print(f"Top {min(10, len(report.target_correlations))} features correlated with {report.target}:")
        for i, (feature, corr) in enumerate(list(report.target_correlations.items())[:10], 1):
            print(f"  {i}. {feature}: {corr:.4f}")
        print()
    elif report.target:
        print(f"Target '{report.target}' specified but no correlations computed.")
        print()
    else:
        print("No target specified - skipping target correlation analysis.")
        print()
    
    # High predictor-predictor correlations
    if report.high_correlations:
        print(f"High predictor-predictor correlations (|r| > 0.7):")
        print(f"  Found {len(report.high_correlations)} pairs")
        for i, (feat1, feat2, corr) in enumerate(report.high_correlations[:5], 1):
            print(f"  {i}. {feat1} <-> {feat2}: {corr:.4f}")
        if len(report.high_correlations) > 5:
            print(f"  ... and {len(report.high_correlations) - 5} more pairs")
    else:
        print("  No high correlations detected (all |r| ≤ 0.7)")
    print()
    
    # Target distribution check (moved before Phase 4)
    if report.target and report.target in report.df.columns:
        target_dist = report.df[report.target].value_counts().sort_index()
        print("Target Distribution:")
        for val, count in target_dist.items():
            pct = count / len(report.df) * 100
            print(f"  {val}: {count} ({pct:.2f}%)")
        print()
    
    # Phase 4: Train baseline models
    if report.target and report.problem_type:
        print("=" * 80)
        print("Step 3: Training Baseline Models (Phase 4)")
        print("=" * 80)
        print()
        
        try:
            report.train_baseline_models()
            print("✓ Model training complete!")
            print()
            
            # Display model metrics
            print("-" * 80)
            print("Model Evaluation Metrics (Step 11)")
            print("-" * 80)
            
            if report.problem_type == 'classification':
                # Logistic Regression / Linear Model
                if 'linear_model' in report.metrics:
                    print("Linear Model (Logistic Regression):")
                    train_metrics = report.metrics['linear_model']['train']
                    test_metrics = report.metrics['linear_model']['test']
                    print(f"  Training Set:")
                    print(f"    - Accuracy: {train_metrics['accuracy']:.4f}")
                    print(f"    - F1 Score: {train_metrics['f1']:.4f}")
                    if 'roc_auc' in train_metrics:
                        print(f"    - ROC-AUC: {train_metrics['roc_auc']:.4f}")
                    print(f"  Test Set:")
                    print(f"    - Accuracy: {test_metrics['accuracy']:.4f}")
                    print(f"    - F1 Score: {test_metrics['f1']:.4f}")
                    if 'roc_auc' in test_metrics:
                        print(f"    - ROC-AUC: {test_metrics['roc_auc']:.4f}")
                    print()
                
                # Random Forest
                if 'tree_model' in report.metrics:
                    print("Tree Model (Random Forest Classifier):")
                    train_metrics = report.metrics['tree_model']['train']
                    test_metrics = report.metrics['tree_model']['test']
                    print(f"  Training Set:")
                    print(f"    - Accuracy: {train_metrics['accuracy']:.4f}")
                    print(f"    - F1 Score: {train_metrics['f1']:.4f}")
                    if 'roc_auc' in train_metrics:
                        print(f"    - ROC-AUC: {train_metrics['roc_auc']:.4f}")
                    print(f"  Test Set:")
                    print(f"    - Accuracy: {test_metrics['accuracy']:.4f}")
                    print(f"    - F1 Score: {test_metrics['f1']:.4f}")
                    if 'roc_auc' in test_metrics:
                        print(f"    - ROC-AUC: {test_metrics['roc_auc']:.4f}")
                    print()
            
            else:  # regression
                # Linear Regression
                if 'linear_model' in report.metrics:
                    print("Linear Model (Linear Regression):")
                    train_metrics = report.metrics['linear_model']['train']
                    test_metrics = report.metrics['linear_model']['test']
                    print(f"  Training Set:")
                    print(f"    - RMSE: {train_metrics['rmse']:.4f}")
                    print(f"    - R²: {train_metrics['r2']:.4f}")
                    print(f"  Test Set:")
                    print(f"    - RMSE: {test_metrics['rmse']:.4f}")
                    print(f"    - R²: {test_metrics['r2']:.4f}")
                    print()
                
                # Random Forest
                if 'tree_model' in report.metrics:
                    print("Tree Model (Random Forest Regressor):")
                    train_metrics = report.metrics['tree_model']['train']
                    test_metrics = report.metrics['tree_model']['test']
                    print(f"  Training Set:")
                    print(f"    - RMSE: {train_metrics['rmse']:.4f}")
                    print(f"    - R²: {train_metrics['r2']:.4f}")
                    print(f"  Test Set:")
                    print(f"    - RMSE: {test_metrics['rmse']:.4f}")
                    print(f"    - R²: {test_metrics['r2']:.4f}")
                    print()
            
            # Display feature importance
            print("-" * 80)
            print("Feature Importance (Step 12)")
            print("-" * 80)
            
            # Linear model feature importance
            if 'linear_feature_importance' in report.models:
                linear_fi = report.models['linear_feature_importance']
                print(f"Linear Model - Top {min(10, len(linear_fi))} Important Features:")
                for i, item in enumerate(linear_fi[:10], 1):
                    print(f"  {i}. {item['feature']}: {item['importance']:.6f}")
                print()
            
            # Tree model feature importance
            if 'tree_feature_importance' in report.models:
                tree_fi = report.models['tree_feature_importance']
                print(f"Tree Model - Top {min(10, len(tree_fi))} Important Features:")
                for i, item in enumerate(tree_fi[:10], 1):
                    print(f"  {i}. {item['feature']}: {item['importance']:.6f}")
                print()
        
        except Exception as e:
            print(f"ERROR during model training: {e}")
            import traceback
            traceback.print_exc()
        print()
    
    else:
        print()
        print("=" * 80)
        print("Skipping model training (no target or problem type specified)")
        print("=" * 80)
        print()
    
    print("=" * 80)
    print("Analysis Summary")
    print("=" * 80)
    print(f"Dataset: {report.df.shape[0]} rows × {report.df.shape[1]} columns")
    if report.target and report.problem_type:
        print(f"Target: {report.target} ({report.problem_type})")
        print(f"Models trained: {'Yes' if report.models else 'No'}")
    elif report.target:
        print(f"Target: {report.target}")
    else:
        print("Target: None (unsupervised analysis)")
    print(f"Missing values: {len(report.missing_values_summary)} columns")
    print(f"Numeric features analyzed: {len(report.numeric_statistics)}")
    print(f"Categorical features analyzed: {len(report.categorical_summary)}")
    print(f"High correlations found: {len(report.high_correlations)}")
    print()
    
    # Check target distribution
    if report.target and report.target in report.df.columns:
        target_dist = report.df[report.target].value_counts().sort_index()
        print("Target Distribution:")
        for val, count in target_dist.items():
            pct = count / len(report.df) * 100
            print(f"  {val}: {count} ({pct:.2f}%)")
        print()
    print("=" * 80)
    print("Analysis complete! Phase 3 & Phase 4 implementation verified.")
    print("=" * 80)


if __name__ == "__main__":
    main()
