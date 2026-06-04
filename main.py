"""
Example script demonstrating how to use the Customer Churn Prediction package.

This script shows the complete workflow from data loading through model evaluation.
"""

import sys
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tắt log của TensorFlow (nếu CTGAN dùng)
os.environ['PYTHONWARNINGS'] = 'ignore'
# Add src to path if running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_data, get_data_info
from src.preprocessing import (
    drop_unnecessary_columns,
    create_age_feature,
    create_tenure_feature,
    remove_age_outliers,
    create_total_spend_feature,
    handle_missing_values,
    group_rare_categories,
    create_preprocessor_pipeline,
    apply_class_balancing,
    split_and_encode,
)
from src.model import create_all_models, train_model
from src.utils import (
    get_independent_test_table,
    get_cv_table,
    print_results_table,
)


def main():
    """Run the complete ML pipeline."""
    
    print("\n" + "="*60)
    print("Customer Churn Prediction - Complete Pipeline")
    print("="*60)
    
    # ============================================================
    # STEP 1: Load Data
    # ============================================================
    print("\n[STEP 1] Loading Data...")
    df = load_data()
    get_data_info(df)
    
    # ============================================================
    # STEP 2: Data Cleaning & Feature Engineering
    # ============================================================
    print("\n[STEP 2] Data Cleaning & Feature Engineering...")
    
    df = drop_unnecessary_columns(df)
    df = create_age_feature(df)
    df = create_tenure_feature(df)
    df = remove_age_outliers(df, max_age=100)
    df = create_total_spend_feature(df)
    df = handle_missing_values(df, strategy='median_by_group')
    
    # Group rare categories
    df = group_rare_categories(df, 'Marital_Status', min_frequency=0.02)
    
    print(f"\nCleaned dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['Response'].value_counts()}")
    
    # ============================================================
    # STEP 3: Preprocessing & Encoding (required before SMOTE)
    # ============================================================
    print("\n[STEP 3] Creating Preprocessing Pipeline...")
    
    X = df.drop('Response', axis=1)
    y = df['Response']
    
    # Create and fit preprocessor on full dataset
    preprocessor = create_preprocessor_pipeline(X)
    X_processed = preprocessor.transform(X)
    
    print(f"✓ Data encoded. Shape after preprocessing: {X_processed.shape}")
    print(f"  (Categorical variables converted to numeric)")
    
    # ============================================================
    # STEP 4: Class Balancing (BorderlineSMOTE on encoded data)
    # ============================================================
    print("\n[STEP 4] Applying Class Balancing (BorderlineSMOTE)...")
    
    X_balanced, y_balanced = apply_class_balancing(
        X_processed, y,
        method='borderline_smote',
        random_state=42,
        verbose=True
    )
    
    # ============================================================
    # STEP 5: Train/Test Split
    # ============================================================
    print("\n[STEP 5] Splitting Data into Train/Test Sets...")
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced,
        test_size=0.3,
        random_state=42,
        stratify=y_balanced
    )
    
    print(f"✓ Train shape: {X_train.shape}")
    print(f"✓ Test shape: {X_test.shape}")
    
    # ============================================================
    # STEP 6: Create Models
    # ============================================================
    print("\n[STEP 6] Creating ML Models...")
    
    all_models = create_all_models(random_state=42)
    
    # ============================================================
    # STEP 7: Evaluate Models (Independent Test Set)
    # ============================================================
    print("\n[STEP 7] Evaluating Models (Independent Test Set)...")
    
    results_test = get_independent_test_table(
        all_models,
        X_train, y_train,
        X_test, y_test,
        verbose=True
    )
    
    print_results_table(results_test, "Independent Test Set Results")
    
    # ============================================================
    # STEP 8: Cross-Validation Evaluation
    # ============================================================
    print("\n[STEP 8] Evaluating Models (5-Fold Cross-Validation)...")
    
    results_cv5 = get_cv_table(
        all_models,
        X_train, y_train,
        k=5,
        verbose=True
    )
    
    print_results_table(results_cv5, "5-Fold Cross-Validation Results")
    
    # ============================================================
    # STEP 9: 10-Fold Cross-Validation
    # ============================================================
    print("\n[STEP 9] Evaluating Models (10-Fold Cross-Validation)...")
    
    results_cv10 = get_cv_table(
        all_models,
        X_train, y_train,
        k=10,
        verbose=False
    )
    
    print_results_table(results_cv10, "10-Fold Cross-Validation Results")



if __name__ == "__main__":
    main()