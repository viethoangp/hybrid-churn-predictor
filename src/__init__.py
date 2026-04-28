"""
Customer Churn Prediction - Production-Ready ML Package

A modular, production-ready package for customer churn prediction using
hybrid ensemble models (HSLR) with multiple class balancing techniques
(SMOTE, Borderline-SMOTE, CTGAN).

Main Components:
- data_loader: Load and validate marketing campaign data
- preprocessing: Feature engineering, encoding, and class balancing
- model: Model creation (base learners and stacking ensemble)
- utils: Evaluation metrics, cross-validation, and visualization

Example Usage:
    >>> from src.data_loader import load_data
    >>> from src.preprocessing import (
    ...     drop_unnecessary_columns, create_age_feature,
    ...     create_tenure_feature, apply_class_balancing
    ... )
    >>> from src.model import create_all_models
    >>> from src.utils import get_cv_table, plot_model_comparison
    >>>
    >>> # Load and preprocess data
    >>> df = load_data()
    >>> df = drop_unnecessary_columns(df)
    >>> df = create_age_feature(df)
    >>> df = create_tenure_feature(df)
    >>> df = create_total_spend_feature(df)
    >>>
    >>> # Apply class balancing
    >>> X_balanced, y_balanced = apply_class_balancing(df.drop('Response', axis=1), df['Response'], method='smote')
    >>>
    >>> # Create and evaluate models
    >>> all_models = create_all_models()
    >>> results = get_cv_table(all_models, X_balanced, y_balanced, k=5)
    >>> print(results)

Version: 1.0.0
"""

__version__ = "1.0.0"
__all__ = [
    # Data Loading
    "load_data",
    "resolve_data_path",
    "validate_data",
    "get_data_info",
    
    # Preprocessing
    "drop_unnecessary_columns",
    "create_tenure_feature",
    "create_age_feature",
    "remove_age_outliers",
    "create_total_spend_feature",
    "handle_missing_values",
    "group_rare_categories",
    "create_preprocessor_pipeline",
    "apply_class_balancing",
    "apply_ctgan_balancing",
    "split_and_encode",
    
    # Models
    "ModelFactory",
    "create_base_learners",
    "create_stacking_ensemble",
    "create_all_models",
    "train_model",
    "predict",
    "save_model",
    "load_model",
    
    # Utils
    "get_independent_test_table",
    "get_cv_table",
    "plot_model_comparison",
    "plot_comparison_across_folds",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "print_results_table",
]


# Import from data_loader module
from .data_loader import (
    load_data,
    resolve_data_path,
    validate_data,
    get_data_info,
)

# Import from preprocessing module
from .preprocessing import (
    drop_unnecessary_columns,
    create_tenure_feature,
    create_age_feature,
    remove_age_outliers,
    create_total_spend_feature,
    handle_missing_values,
    group_rare_categories,
    create_preprocessor_pipeline,
    apply_class_balancing,
    apply_ctgan_balancing,
    split_and_encode,
)

# Import from model module
from .model import (
    ModelFactory,
    create_base_learners,
    create_stacking_ensemble,
    create_all_models,
    train_model,
    predict,
    save_model,
    load_model,
)

# Import from utils module
from .utils import (
    get_independent_test_table,
    get_cv_table,
    plot_model_comparison,
    plot_comparison_across_folds,
    plot_confusion_matrix,
    plot_roc_curve,
    print_results_table,
)


def print_package_info() -> None:
    """
    Print package information and available components.
    
    Example:
        >>> from src import print_package_info
        >>> print_package_info()
    """
    print(f"\n{'='*60}")
    print(f"Customer Churn Prediction Package v{__version__}")
    print(f"{'='*60}")
    print("\nAvailable Modules:")
    print("\n1. Data Loading (data_loader)")
    print("   - load_data(): Load marketing campaign dataset")
    print("   - resolve_data_path(): Smart path detection")
    print("   - validate_data(): Validate dataset structure")
    print("   - get_data_info(): Print dataset statistics")
    
    print("\n2. Preprocessing (preprocessing)")
    print("   - drop_unnecessary_columns(): Remove non-essential columns")
    print("   - create_age_feature(): Generate Age from birth year")
    print("   - create_tenure_feature(): Generate Tenure from joining date")
    print("   - remove_age_outliers(): Filter age outliers")
    print("   - create_total_spend_feature(): Sum spending columns")
    print("   - handle_missing_values(): Impute missing data")
    print("   - group_rare_categories(): Consolidate rare categories")
    print("   - create_preprocessor_pipeline(): Build ColumnTransformer")
    print("   - apply_class_balancing(): SMOTE/Borderline-SMOTE balancing")
    print("   - apply_ctgan_balancing(): CTGAN-based balancing")
    print("   - split_and_encode(): Train/test split with preprocessing")
    
    print("\n3. Model Building (model)")
    print("   - ModelFactory: Create individual models with defaults")
    print("   - create_base_learners(): Initialize RF, XGB, ADA, LGBM")
    print("   - create_stacking_ensemble(): Build HSLR meta-learner")
    print("   - create_all_models(): Create all models at once")
    print("   - train_model(): Train a single model")
    print("   - predict(): Make predictions")
    print("   - save_model(): Persist model to disk")
    print("   - load_model(): Load saved model")
    
    print("\n4. Evaluation & Visualization (utils)")
    print("   - get_independent_test_table(): Evaluate on test set")
    print("   - get_cv_table(): K-fold cross-validation results")
    print("   - plot_model_comparison(): Compare models across methods")
    print("   - plot_comparison_across_folds(): Compare evaluation strategies")
    print("   - plot_confusion_matrix(): Confusion matrix heatmap")
    print("   - plot_roc_curve(): ROC curve visualization")
    print("   - print_results_table(): Format results as table")
    
    print(f"\n{'='*60}\n")