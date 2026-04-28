"""
Module for model initialization, training, and prediction.

This module provides functionality for:
- Initializing individual base learners (RF, XGBoost, LightGBM, AdaBoost)
- Building hybrid stacking ensemble (HSLR)
- Training and making predictions with models
- Hyperparameter management
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class ModelFactory:
    """Factory for creating machine learning models with predefined hyperparameters."""
    
    @staticmethod
    def create_random_forest(
        n_estimators: int = 200,
        criterion: str = 'entropy',
        max_depth: int = 20,
        random_state: int = 42,
        n_jobs: int = -1
    ) -> RandomForestClassifier:
        """
        Create a Random Forest classifier.
        
        Args:
            n_estimators: Number of trees (default: 200)
            criterion: Split criterion (default: 'entropy')
            max_depth: Maximum tree depth (default: 20)
            random_state: Random seed (default: 42)
            n_jobs: Number of parallel jobs (default: -1 = all)
            
        Returns:
            Configured RandomForestClassifier
        """
        return RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs
        )
    
    @staticmethod
    def create_xgboost(
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        random_state: int = 42,
        eval_metric: str = 'logloss',
        verbosity: int = 0
    ) -> XGBClassifier:
        """
        Create an XGBoost classifier.
        
        Args:
            n_estimators: Number of boosting rounds (default: 200)
            learning_rate: Learning rate / eta (default: 0.1)
            max_depth: Maximum tree depth (default: 6)
            random_state: Random seed (default: 42)
            eval_metric: Evaluation metric (default: 'logloss')
            verbosity: Verbosity level (default: 0 = silent)
            
        Returns:
            Configured XGBClassifier
        """
        return XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            eval_metric=eval_metric,
            random_state=random_state,
            verbosity=verbosity,
            use_label_encoder=False
        )
    
    @staticmethod
    def create_adaboost(
        n_estimators: int = 100,
        learning_rate: float = 0.8,
        random_state: int = 42
    ) -> AdaBoostClassifier:
        """
        Create an AdaBoost classifier.
        
        Args:
            n_estimators: Number of weak learners (default: 100)
            learning_rate: Learning rate (default: 0.8)
            random_state: Random seed (default: 42)
            
        Returns:
            Configured AdaBoostClassifier
        """
        return AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state
        )
    
    @staticmethod
    def create_lightgbm(
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        random_state: int = 42,
        verbosity: int = -1,
        n_jobs: int = -1
    ) -> LGBMClassifier:
        """
        Create a LightGBM classifier.
        
        Args:
            n_estimators: Number of boosting rounds (default: 200)
            learning_rate: Learning rate (default: 0.1)
            num_leaves: Number of leaves (default: 31)
            random_state: Random seed (default: 42)
            verbosity: Verbosity level (default: -1 = silent)
            n_jobs: Number of parallel jobs (default: -1 = all)
            
        Returns:
            Configured LGBMClassifier
        """
        return LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state,
            verbosity=verbosity,
            n_jobs=n_jobs
        )


def create_base_learners(
    random_state: int = 42,
    custom_params: Optional[Dict[str, Dict]] = None
) -> List[Tuple[str, Any]]:
    """
    Create a list of base learners for ensemble.
    
    Args:
        random_state: Random seed for reproducibility (default: 42)
        custom_params: Optional dictionary with custom hyperparameters
                      Keys should match model names ('RF', 'XGB', 'ADA', 'LGBM')
                      Example: {'RF': {'n_estimators': 300, 'max_depth': 25}}
                      
    Returns:
        List of (model_name, model_instance) tuples
        
    Example:
        >>> base_learners = create_base_learners()
        >>> custom = {'RF': {'n_estimators': 300}, 'XGB': {'learning_rate': 0.05}}
        >>> base_learners = create_base_learners(custom_params=custom)
    """
    factory = ModelFactory()
    custom_params = custom_params or {}
    
    base_learners = []
    
    # Random Forest
    rf_kwargs = custom_params.get('RF', {})
    rf_kwargs.setdefault('random_state', random_state)
    rf_model = factory.create_random_forest(**rf_kwargs)
    base_learners.append(('RF', rf_model))
    
    # XGBoost
    xgb_kwargs = custom_params.get('XGB', {})
    xgb_kwargs.setdefault('random_state', random_state)
    xgb_model = factory.create_xgboost(**xgb_kwargs)
    base_learners.append(('XGB', xgb_model))
    
    # AdaBoost
    ada_kwargs = custom_params.get('ADA', {})
    ada_kwargs.setdefault('random_state', random_state)
    ada_model = factory.create_adaboost(**ada_kwargs)
    base_learners.append(('ADA', ada_model))
    
    # LightGBM
    lgbm_kwargs = custom_params.get('LGBM', {})
    lgbm_kwargs.setdefault('random_state', random_state)
    lgbm_model = factory.create_lightgbm(**lgbm_kwargs)
    base_learners.append(('LGBM', lgbm_model))
    
    print(f"✓ Created {len(base_learners)} base learners:")
    for name, _ in base_learners:
        print(f"  - {name}")
    
    return base_learners


def create_stacking_ensemble(
    base_learners: List[Tuple[str, Any]],
    meta_learner: Optional[Any] = None,
    cv: int = 5,
    n_jobs: int = -1
) -> StackingClassifier:
    """
    Create a Stacking Ensemble model with base learners and meta-learner.
    
    The Hybrid Stacking Logistic Regression (HSLR) model combines:
    - Base learners: RF, XGBoost, AdaBoost, LightGBM
    - Meta-learner: Logistic Regression
    
    Args:
        base_learners: List of (name, model) tuples for base learners
        meta_learner: Meta-learner model. If None, uses Logistic Regression.
        cv: Number of cross-validation folds for training meta-learner (default: 5)
        n_jobs: Number of parallel jobs (default: -1 = all)
        
    Returns:
        Configured StackingClassifier
        
    Example:
        >>> base_learners = create_base_learners()
        >>> hslr = create_stacking_ensemble(base_learners)
    """
    if meta_learner is None:
        meta_learner = LogisticRegression(max_iter=1000, random_state=42)
    
    stacking_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=cv,
        n_jobs=n_jobs
    )
    
    print(f"✓ Created Stacking Ensemble with meta-learner: LogisticRegression")
    
    return stacking_model


def create_all_models(
    random_state: int = 42,
    base_custom_params: Optional[Dict[str, Dict]] = None,
    meta_learner: Optional[Any] = None
) -> List[Tuple[str, Any]]:
    """
    Create all models including base learners and stacking ensemble.
    
    Args:
        random_state: Random seed (default: 42)
        base_custom_params: Custom hyperparameters for base learners
        meta_learner: Custom meta-learner for stacking
        
    Returns:
        List of (model_name, model_instance) tuples including HSLR
        
    Example:
        >>> all_models = create_all_models()
    """
    # Create base learners
    base_learners = create_base_learners(
        random_state=random_state,
        custom_params=base_custom_params
    )
    
    # Create stacking ensemble
    stacking_model = create_stacking_ensemble(
        base_learners,
        meta_learner=meta_learner
    )
    
    # Combine all models
    all_models = base_learners + [('HSLR', stacking_model)]
    
    print(f"✓ Total models created: {len(all_models)}")
    
    return all_models


def train_model(
    model: Any,
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    verbose: bool = True
) -> Any:
    """
    Train a single model.
    
    Args:
        model: Scikit-learn model instance
        X_train: Training features
        y_train: Training labels
        verbose: If True, print training status
        
    Returns:
        Trained model
        
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> rf = RandomForestClassifier()
        >>> rf_trained = train_model(rf, X_train, y_train)
    """
    try:
        model.fit(X_train, y_train)
        if verbose:
            print(f"✓ Model trained successfully")
        return model
    except Exception as e:
        raise RuntimeError(f"Error training model: {str(e)}")


def predict(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    return_proba: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained scikit-learn model
        X: Feature matrix for prediction
        return_proba: If True, also return probability predictions
        
    Returns:
        Tuple of (y_pred, y_proba) if return_proba=True, else just y_pred
        
    Example:
        >>> y_pred = predict(model, X_test)
        >>> y_pred, y_proba = predict(model, X_test, return_proba=True)
    """
    try:
        y_pred = model.predict(X)
        
        if return_proba:
            try:
                y_proba = model.predict_proba(X)
                return y_pred, y_proba
            except AttributeError:
                print("⚠ Model does not support probability predictions")
                return y_pred, None
        
        return y_pred, None
    
    except Exception as e:
        raise RuntimeError(f"Error making predictions: {str(e)}")


def save_model(
    model: Any,
    file_path: str | Path,
    verbose: bool = True
) -> None:
    """
    Save a trained model to disk using pickle.
    
    Args:
        model: Trained model instance
        file_path: Path to save the model
        verbose: If True, print save status
        
    Example:
        >>> save_model(trained_rf, 'models/random_forest.pkl')
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        if verbose:
            print(f"✓ Model saved to: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error saving model to {file_path}: {str(e)}")


def load_model(
    file_path: str | Path,
    verbose: bool = True
) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        file_path: Path to the saved model
        verbose: If True, print load status
        
    Returns:
        Loaded model
        
    Example:
        >>> model = load_model('models/random_forest.pkl')
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        if verbose:
            print(f"✓ Model loaded from: {file_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model from {file_path}: {str(e)}")