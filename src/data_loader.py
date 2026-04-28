"""
Module for loading and validating data.

This module provides functionality to load the marketing campaign dataset
with proper path handling for different environments (notebook, script, production).
"""

import os
from pathlib import Path
from typing import Tuple
import pandas as pd


def resolve_data_path(file_name: str = "marketing_campaign.csv") -> Path:
    """
    Resolve the path to the data file regardless of where the script is called from.
    
    Handles multiple scenarios:
    - Called from notebooks/ directory
    - Called from src/ directory
    - Called from project root
    
    Args:
        file_name: Name of the data file (default: "marketing_campaign.csv")
        
    Returns:
        Path object pointing to the data file
        
    Raises:
        FileNotFoundError: If data file cannot be found in standard locations
    """
    # Get the project root by finding the data folder
    current_dir = Path.cwd()
    
    # Try different possible paths
    possible_paths = [
        current_dir / "data" / file_name,  # Called from project root
        current_dir.parent / "data" / file_name,  # Called from src/ or notebooks/
        current_dir.parent.parent / "data" / file_name,  # Called from subdirectories
        Path(__file__).parent.parent / "data" / file_name,  # Relative to this file
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # If not found, raise error with helpful message
    raise FileNotFoundError(
        f"Could not find '{file_name}' in any of the standard locations:\n"
        + "\n".join(str(p) for p in possible_paths)
    )


def load_data(
    file_path: str | Path | None = None,
    sep: str = '\t',
    **kwargs
) -> pd.DataFrame:
    """
    Load the marketing campaign dataset from CSV.
    
    Args:
        file_path: Optional path to the CSV file. If None, will auto-detect.
        sep: Column separator in the CSV (default: '\t' for tab-separated values)
        **kwargs: Additional arguments passed to pd.read_csv()
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If data file cannot be found
        pd.errors.ParserError: If CSV cannot be parsed
        
    Example:
        >>> df = load_data()  # Auto-detect path
        >>> df = load_data("../data/marketing_campaign.csv")  # Explicit path
    """
    if file_path is None:
        file_path = resolve_data_path()
    else:
        file_path = Path(file_path)
        if not file_path.exists():
            file_path = resolve_data_path(file_path.name)
    
    try:
        df = pd.read_csv(file_path, sep=sep, **kwargs)
        print(f"✓ Data loaded successfully from: {file_path}")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {file_path}: {str(e)}")


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate that the loaded data has expected structure.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    expected_columns = {
        'Response', 'Age', 'Income', 'Marital_Status', 
        'Education', 'Dt_Customer'
    }
    
    missing_cols = expected_columns - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Expected columns missing: {missing_cols}\n"
            f"Available columns: {set(df.columns)}"
        )
    
    if df.empty:
        raise ValueError("Dataset is empty")
    
    print("✓ Data validation passed")
    return True


def get_data_info(df: pd.DataFrame) -> None:
    """
    Print informative statistics about the dataset.
    
    Args:
        df: DataFrame to analyze
    """
    print("\n" + "="*60)
    print("Dataset Information")
    print("="*60)
    print(f"Shape: {df.shape}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nTarget Distribution:\n{df['Response'].value_counts()}")
    print("="*60 + "\n")