"""
Module for data preprocessing, feature engineering, and class balancing.

This module provides functionality for:
- Data cleaning (handling missing values, dropping columns)
- Feature engineering (age, tenure, total spending)
- Categorical grouping and encoding
- ColumnTransformer pipeline for scaling and encoding
- Class balancing techniques (SMOTE, Borderline-SMOTE, CTGAN)
"""

from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE, BorderlineSMOTE


def drop_unnecessary_columns(
    df: pd.DataFrame,
    columns_to_drop: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Drop unnecessary columns from the dataset.
    
    Args:
        df: Input DataFrame
        columns_to_drop: List of columns to drop. If None, uses default list.
                        Default: ['ID', 'AcceptedCmp1-5', 'Z_CostContact', 'Z_Revenue']
                        
    Returns:
        DataFrame with specified columns removed
        
    Example:
        >>> df = drop_unnecessary_columns(df)
        >>> df = drop_unnecessary_columns(df, columns_to_drop=['ID', 'Z_CostContact'])
    """
    if columns_to_drop is None:
        columns_to_drop = [
            'ID', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
            'AcceptedCmp4', 'AcceptedCmp5', 'Z_CostContact', 'Z_Revenue'
        ]
    
    # Only drop columns that exist in the DataFrame
    existing_cols_to_drop = [c for c in columns_to_drop if c in df.columns]
    
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
        print(f"✓ Dropped columns: {existing_cols_to_drop}")
    
    return df


def create_tenure_feature(
    df: pd.DataFrame,
    date_column: str = 'Dt_Customer',
    reference_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Create Tenure feature from customer joining date.
    
    Converts date column to datetime and calculates days since customer joined.
    Then drops the original date column.
    
    Args:
        df: Input DataFrame
        date_column: Name of the date column (default: 'Dt_Customer')
        reference_date: Reference date for tenure calculation (default: '2025-11-20')
                       Format: 'YYYY-MM-DD'
                       
    Returns:
        DataFrame with new 'Tenure_days' column and date column removed
        
    Example:
        >>> df = create_tenure_feature(df)
        >>> df = create_tenure_feature(df, reference_date='2025-12-31')
    """
    if date_column not in df.columns:
        print(f"⚠ Column '{date_column}' not found. Skipping tenure creation.")
        return df
    
    if reference_date is None:
        reference_date = '2025-11-20'
    
    try:
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column], dayfirst=True, errors='coerce')
        ref_date = pd.to_datetime(reference_date)
        df['Tenure_days'] = (ref_date - df[date_column]).dt.days
        df = df.drop(columns=[date_column])
        print(f"✓ Created 'Tenure_days' feature (reference: {reference_date})")
        return df
    except Exception as e:
        print(f"⚠ Error creating tenure feature: {str(e)}")
        return df


def create_age_feature(
    df: pd.DataFrame,
    birth_year_column: str = 'Year_Birth',
    current_year: int = 2025
) -> pd.DataFrame:
    """
    Create Age feature from birth year and drop the birth year column.
    
    Args:
        df: Input DataFrame
        birth_year_column: Name of birth year column (default: 'Year_Birth')
        current_year: Current year for age calculation (default: 2025)
        
    Returns:
        DataFrame with 'Age' column and birth year column removed
        
    Example:
        >>> df = create_age_feature(df)
        >>> df = create_age_feature(df, current_year=2026)
    """
    if birth_year_column not in df.columns:
        print(f"⚠ Column '{birth_year_column}' not found. Skipping age creation.")
        return df
    
    df = df.copy()
    df['Age'] = current_year - df[birth_year_column]
    df = df.drop(columns=[birth_year_column])
    print(f"✓ Created 'Age' feature")
    return df


def remove_age_outliers(
    df: pd.DataFrame,
    age_column: str = 'Age',
    max_age: int = 100
) -> pd.DataFrame:
    """
    Remove records with age outliers (e.g., age > max_age).
    
    Args:
        df: Input DataFrame
        age_column: Name of age column (default: 'Age')
        max_age: Maximum allowed age (default: 100)
        
    Returns:
        DataFrame with age outliers removed
        
    Example:
        >>> df = remove_age_outliers(df, max_age=100)
        >>> print(f"Removed {len(df_original) - len(df)} records")
    """
    if age_column not in df.columns:
        print(f"⚠ Column '{age_column}' not found. Skipping outlier removal.")
        return df
    
    original_len = len(df)
    df = df[df[age_column] <= max_age].copy()
    removed = original_len - len(df)
    
    if removed > 0:
        print(f"✓ Removed {removed} records with age > {max_age}")
    
    return df


def create_total_spend_feature(
    df: pd.DataFrame,
    prefix: str = 'Mnt'
) -> pd.DataFrame:
    """
    Create Total_Spend feature by summing all spending columns.
    
    Args:
        df: Input DataFrame
        prefix: Prefix of spending columns (default: 'Mnt')
        
    Returns:
        DataFrame with 'Total_Spend' feature added
        
    Example:
        >>> df = create_total_spend_feature(df)
    """
    money_cols = [c for c in df.columns if c.startswith(prefix)]
    
    if money_cols:
        df = df.copy()
        df['Total_Spend'] = df[money_cols].sum(axis=1)
        print(f"✓ Created 'Total_Spend' from {len(money_cols)} spending columns")
    else:
        print(f"⚠ No columns found with prefix '{prefix}'")
    
    return df


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'median_by_group'
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        strategy: Imputation strategy
                 'median_by_group': Fill by group median (default, for Income column)
                 'drop': Drop rows with missing values
                 'mean': Fill with column mean
                 
    Returns:
        DataFrame with missing values handled
        
    Example:
        >>> df = handle_missing_values(df, strategy='median_by_group')
    """
    df = df.copy()
    missing_counts = df.isnull().sum()
    
    if missing_counts.sum() == 0:
        print("✓ No missing values detected")
        return df
    
    print(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
    
    if strategy == 'median_by_group':
        # Fill Income by Education group median
        if 'Income' in df.columns and 'Education' in df.columns:
            df['Income'] = df.groupby('Education')['Income'].transform(
                lambda x: x.fillna(x.median())
            )
            print(f"✓ Filled missing Income values using Education group median")
        else:
            print("⚠ Income or Education column not found for median_by_group strategy")
    
    elif strategy == 'drop':
        df = df.dropna()
        print(f"✓ Dropped rows with missing values")
    
    elif strategy == 'mean':
        df = df.fillna(df.mean())
        print(f"✓ Filled missing values with column means")
    
    return df


def group_rare_categories(
    df: pd.DataFrame,
    column: str,
    min_frequency: float = 0.02,
    other_label: str = 'Other'
) -> pd.DataFrame:
    """
    Group rare categories into 'Other' category.
    
    Groups categories that appear less than min_frequency proportion
    of the time into a single 'Other' category.
    
    Args:
        df: Input DataFrame
        column: Column name to process
        min_frequency: Minimum frequency threshold (default: 0.02 = 2%)
        other_label: Label for grouped rare categories (default: 'Other')
        
    Returns:
        DataFrame with rare categories grouped
        
    Example:
        >>> df = group_rare_categories(df, 'Marital_Status', min_frequency=0.02)
    """
    if column not in df.columns:
        print(f"⚠ Column '{column}' not found")
        return df
    
    df = df.copy()
    value_counts = df[column].value_counts(normalize=True)
    rare_categories = value_counts[value_counts < min_frequency].index.tolist()
    
    if rare_categories:
        df[column] = df[column].replace(rare_categories, other_label)
        print(f"✓ Grouped rare categories in '{column}': {rare_categories} → {other_label}")
    
    return df


def create_preprocessor_pipeline(
    df_sample: pd.DataFrame,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> ColumnTransformer:
    """
    Create a ColumnTransformer pipeline for preprocessing.
    
    Applies StandardScaler to numeric features and OneHotEncoder to categorical features.
    
    Args:
        df_sample: Sample DataFrame to fit the preprocessor
        numeric_features: List of numeric column names. If None, auto-detects.
        categorical_features: List of categorical column names. If None, auto-detects.
        
    Returns:
        Fitted ColumnTransformer pipeline
        
    Example:
        >>> preprocessor = create_preprocessor_pipeline(X_train)
        >>> X_train_processed = preprocessor.transform(X_train)
    """
    if numeric_features is None:
        # Auto-detect: exclude Response and known categorical columns
        exclude_cols = {'Response', 'Education', 'Marital_Status'}
        numeric_features = [
            col for col in df_sample.columns 
            if col not in exclude_cols
        ]
    
    if categorical_features is None:
        categorical_features = ['Education', 'Marital_Status']
    
    # Filter to only existing columns
    numeric_features = [f for f in numeric_features if f in df_sample.columns]
    categorical_features = [f for f in categorical_features if f in df_sample.columns]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), 
             categorical_features)
        ]
    )
    
    preprocessor.fit(df_sample)
    print(f"✓ Created preprocessing pipeline:")
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    
    return preprocessor


def apply_class_balancing(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    method: str = 'smote',
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply class balancing technique to handle imbalanced data.
    
    Supports SMOTE, Borderline-SMOTE, and CTGAN (if available).
    Always returns clean numpy arrays to avoid multiprocessing issues.
    
    Args:
        X: Feature matrix (numpy array or DataFrame)
        y: Target labels (numpy array or Series)
        method: Balancing method - 'smote', 'borderline_smote', or 'ctgan'
               (default: 'smote')
        random_state: Random seed for reproducibility (default: 42)
        verbose: If True, print class distribution before/after
        
    Returns:
        Tuple of (X_balanced, y_balanced) as numpy arrays
        
    Raises:
        ValueError: If unsupported method is provided
        ImportError: If required library is not installed
        
    Example:
        >>> X_balanced, y_balanced = apply_class_balancing(X, y, method='smote')
        >>> print(Counter(y_balanced))
    """
    # Convert to numpy if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    # Print original distribution
    if verbose:
        print(f"\n{'='*60}")
        print(f"Class Balancing: {method.upper()}")
        print(f"{'='*60}")
        print(f"Original class distribution: {Counter(y)}")
    
    method = method.lower().strip()
    
    if method == 'smote':
        try:
            smote = SMOTE(
                sampling_strategy='minority',
                k_neighbors=5,
                random_state=random_state
            )
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            if verbose:
                print(f"Balanced class distribution: {Counter(y_balanced)}")
                print(f"✓ SMOTE applied successfully")
            
            # Ensure clean numpy arrays
            return np.asarray(X_balanced), np.asarray(y_balanced)
        
        except Exception as e:
            raise RuntimeError(f"Error applying SMOTE: {str(e)}")
    
    elif method == 'borderline_smote':
        try:
            bsmote = BorderlineSMOTE(
                random_state=random_state,
                kind='borderline-1'
            )
            X_balanced, y_balanced = bsmote.fit_resample(X, y)
            
            if verbose:
                print(f"Balanced class distribution: {Counter(y_balanced)}")
                print(f"✓ Borderline-SMOTE applied successfully")
            
            # Ensure clean numpy arrays
            return np.asarray(X_balanced), np.asarray(y_balanced)
        
        except Exception as e:
            raise RuntimeError(f"Error applying Borderline-SMOTE: {str(e)}")
    
    elif method == 'ctgan':
        try:
            from ctgan import CTGAN
        except ImportError:
            raise ImportError(
                "CTGAN not installed. Install it with: pip install ctgan"
            )
        
        # For CTGAN, we need the original DataFrame with proper discrete columns
        raise NotImplementedError(
            "CTGAN requires the original DataFrame with discrete column specifications. "
            "Please use apply_ctgan_balancing() instead."
        )
    
    else:
        raise ValueError(
            f"Unsupported balancing method: {method}. "
            f"Choose from: 'smote', 'borderline_smote', 'ctgan'"
        )


def apply_ctgan_balancing(
    df: pd.DataFrame,
    target_column: str = 'Response',
    discrete_columns: Optional[List[str]] = None,
    epochs: int = 300,
    batch_size: int = 500,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Apply CTGAN-based synthetic data generation for class balancing.
    
    Trains a CTGAN model on the minority class and generates synthetic samples
    to balance the dataset.
    
    Args:
        df: Input DataFrame with target column
        target_column: Name of target column (default: 'Response')
        discrete_columns: List of discrete/categorical columns for CTGAN
                         If None, auto-detects object-type columns
        epochs: Number of training epochs (default: 300)
        batch_size: Batch size for training (default: 500)
        random_state: Random seed (default: 42)
        verbose: If True, print progress information
        
    Returns:
        Tuple of (df_balanced, X_balanced, y_balanced)
        - df_balanced: Balanced DataFrame with original + synthetic data
        - X_balanced: Feature matrix (numpy array)
        - y_balanced: Target array (numpy array)
        
    Raises:
        ImportError: If ctgan is not installed
        
    Example:
        >>> df_balanced, X, y = apply_ctgan_balancing(
        ...     df, 
        ...     discrete_columns=['Education', 'Marital_Status']
        ... )
    """
    try:
        from ctgan import CTGAN
    except ImportError:
        raise ImportError(
            "CTGAN not installed. Install it with: pip install ctgan"
        )
    
    df = df.copy()
    
    # Auto-detect discrete columns if not provided
    if discrete_columns is None:
        discrete_columns = df.select_dtypes(include=['object']).columns.tolist()
        # Add known discrete numeric columns
        discrete_columns.extend([
            'Kidhome', 'Teenhome', 'NumDealsPurchases', 'NumWebPurchases',
            'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Complain'
        ])
        discrete_columns = [c for c in discrete_columns if c in df.columns and c != target_column]
    
    if verbose:
        print(f"\n{'='*60}")
        print("Class Balancing: CTGAN")
        print(f"{'='*60}")
        print(f"Original class distribution: {Counter(df[target_column])}")
    
    # Separate by class
    minority_data = df[df[target_column] == 1].copy()
    majority_data = df[df[target_column] == 0].copy()
    
    n_to_generate = len(majority_data) - len(minority_data)
    
    if n_to_generate <= 0:
        if verbose:
            print("✓ Data already balanced or minority is larger")
        return df, df.drop(target_column, axis=1).values, df[target_column].values
    
    if verbose:
        print(f"Training CTGAN to generate {n_to_generate} synthetic samples...")
    
    # Train CTGAN on minority class
    ctgan_model = CTGAN(
        epochs=epochs,
        batch_size=batch_size,
        generator_dim=(256, 256),
        embedding_dim=128,
        verbose=False
    )
    ctgan_model.fit(minority_data, discrete_columns)
    
    # Generate synthetic data
    synthetic_data = ctgan_model.sample(n_to_generate)
    
    # Post-processing: round integer columns
    int_columns = [
        'Kidhome', 'Teenhome', 'NumWebVisitsMonth', 'Age', 'Recency',
        'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
        'NumStorePurchases', 'Complain'
    ]
    for col in int_columns:
        if col in synthetic_data.columns:
            synthetic_data[col] = synthetic_data[col].round().astype(int)
    
    # Combine: majority + original minority + synthetic minority
    df_balanced = pd.concat(
        [majority_data, minority_data, synthetic_data],
        ignore_index=True
    )
    
    # Ensure Response is integer
    df_balanced[target_column] = df_balanced[target_column].round().astype(int)
    
    if verbose:
        print(f"Balanced class distribution: {Counter(df_balanced[target_column])}")
        print(f"✓ CTGAN balancing completed")
    
    # Return as numpy arrays to avoid multiprocessing issues
    X_balanced = df_balanced.drop(target_column, axis=1).values
    y_balanced = df_balanced[target_column].values
    
    return df_balanced, X_balanced, y_balanced


def split_and_encode(
    df: pd.DataFrame,
    target_column: str = 'Response',
    preprocessor: Optional[ColumnTransformer] = None,
    test_size: float = 0.3,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
    """
    Split data into train/test sets and apply preprocessing.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column (default: 'Response')
        preprocessor: Optional pre-fitted ColumnTransformer. If None, creates one.
        test_size: Proportion of test set (default: 0.3)
        random_state: Random seed (default: 42)
        stratify: If True, stratify split by target (default: True)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor)
        
    Example:
        >>> X_train, X_test, y_train, y_test, prep = split_and_encode(df)
    """
    from sklearn.model_selection import train_test_split
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    stratify_y = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y
    )
    
    if preprocessor is None:
        preprocessor = create_preprocessor_pipeline(X_train)
    
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"✓ Data split: Train {X_train_processed.shape}, Test {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed, y_train.values, y_test.values, preprocessor