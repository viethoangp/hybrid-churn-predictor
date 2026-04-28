"""
Module for evaluation metrics, cross-validation, and visualization utilities.

This module provides helper functions for:
- Model evaluation (independent test and cross-validation)
- Metrics computation (accuracy, precision, recall, F1, MCC, ROC-AUC)
- Result visualization and reporting
"""

from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, make_scorer, 
    confusion_matrix, roc_curve, auc
)


def get_independent_test_table(
    models: List[Tuple[str, Any]],
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluate models on independent test set and return metrics table.
    
    Trains each model on training data and evaluates on held-out test set.
    Computes: Accuracy, Precision, Recall, F1, MCC, and ROC-AUC.
    
    Args:
        models: List of tuples (model_name, model_instance)
        X_train: Training feature matrix (numpy array or DataFrame)
        y_train: Training labels (numpy array or Series)
        X_test: Test feature matrix (numpy array or DataFrame)
        y_test: Test labels (numpy array or Series)
        verbose: If True, print results during evaluation
        
    Returns:
        DataFrame with metrics for each model:
        Columns: [Classifier, Accuracy, Precision, Recall, MCC, F1, ROC]
        
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> rf = RandomForestClassifier(random_state=42)
        >>> ada = AdaBoostClassifier(random_state=42)
        >>> models = [('RF', rf), ('ADA', ada)]
        >>> results = get_independent_test_table(models, X_train, y_train, X_test, y_test)
    """
    results = []
    
    if verbose:
        print("\n" + "="*60)
        print("Independent Test Set Evaluation")
        print("="*60)
    
    for model_name, model in models:
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Compute ROC-AUC (if probability predictions available)
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                roc = roc_auc_score(y_test, y_prob)
            except (AttributeError, IndexError):
                roc = 0.0
            
            # Compute all metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            results.append({
                "Classifier": model_name,
                "Accuracy": acc * 100,
                "Precision": prec * 100,
                "Recall": rec * 100,
                "MCC": mcc * 100,
                "F1": f1 * 100,
                "ROC": roc
            })
            
            if verbose:
                print(f"\n{model_name}:")
                print(f"  Accuracy:  {acc*100:.2f}%")
                print(f"  Precision: {prec*100:.2f}%")
                print(f"  Recall:    {rec*100:.2f}%")
                print(f"  F1-Score:  {f1*100:.2f}%")
                print(f"  ROC-AUC:   {roc:.4f}")
        
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            results.append({
                "Classifier": model_name,
                "Accuracy": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "MCC": 0.0,
                "F1": 0.0,
                "ROC": 0.0
            })
    
    df_results = pd.DataFrame(results)
    return df_results[["Classifier", "Accuracy", "Precision", "Recall", "MCC", "F1", "ROC"]].round(3)


def get_cv_table(
    models: List[Tuple[str, Any]],
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    k: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluate models using K-Fold Cross-Validation.
    
    Performs stratified k-fold cross-validation for each model to get
    robust performance estimates. Computes mean scores across folds.
    
    Args:
        models: List of tuples (model_name, model_instance)
        X: Feature matrix (numpy array or DataFrame)
        y: Labels (numpy array or Series)
        k: Number of folds (default: 5)
        verbose: If True, print progress during evaluation
        
    Returns:
        DataFrame with mean cross-validation metrics for each model:
        Columns: [Classifier, Accuracy, Precision, Recall, MCC, F1, ROC]
        
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> rf = RandomForestClassifier(random_state=42)
        >>> models = [('RF', rf)]
        >>> cv_results = get_cv_table(models, X, y, k=5)
    """
    results = []
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    # Define scoring metrics for cross_validate
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'mcc': make_scorer(matthews_corrcoef),
        'roc_auc': 'roc_auc'
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"{k}-Fold Cross-Validation Evaluation")
        print(f"{'='*60}")
    
    for model_name, model in models:
        try:
            scores = cross_validate(
                model, X, y, 
                cv=cv, 
                scoring=scoring, 
                n_jobs=-1,
                return_train_score=False
            )
            
            results.append({
                "Classifier": model_name,
                "Accuracy": scores['test_accuracy'].mean() * 100,
                "Precision": scores['test_precision'].mean() * 100,
                "Recall": scores['test_recall'].mean() * 100,
                "MCC": scores['test_mcc'].mean() * 100,
                "F1": scores['test_f1'].mean() * 100,
                "ROC": scores['test_roc_auc'].mean()
            })
            
            if verbose:
                print(f"\n{model_name} ({k}-Fold):")
                print(f"  Accuracy:  {scores['test_accuracy'].mean()*100:.2f}% "
                      f"(± {scores['test_accuracy'].std()*100:.2f}%)")
                print(f"  Precision: {scores['test_precision'].mean()*100:.2f}% "
                      f"(± {scores['test_precision'].std()*100:.2f}%)")
                print(f"  F1-Score:  {scores['test_f1'].mean()*100:.2f}% "
                      f"(± {scores['test_f1'].std()*100:.2f}%)")
                print(f"  ROC-AUC:   {scores['test_roc_auc'].mean():.4f} "
                      f"(± {scores['test_roc_auc'].std():.4f})")
        
        except Exception as e:
            print(f"Error in {k}-fold CV for {model_name}: {str(e)}")
            results.append({
                "Classifier": model_name,
                "Accuracy": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "MCC": 0.0,
                "F1": 0.0,
                "ROC": 0.0
            })
    
    df_results = pd.DataFrame(results)
    return df_results[["Classifier", "Accuracy", "Precision", "Recall", "MCC", "F1", "ROC"]].round(3)


def plot_model_comparison(
    results_dict: Dict[str, pd.DataFrame],
    metric: str = "Accuracy",
    figsize: Tuple[int, int] = (14, 8),
    palette: str = "viridis"
) -> None:
    """
    Create a bar plot comparing metrics across different balancing methods.
    
    Args:
        results_dict: Dictionary with method names as keys and result DataFrames as values
                     e.g., {'SMOTE': df_smote, 'Borderline-SMOTE': df_bsmote}
        metric: Metric to plot (default: "Accuracy")
        figsize: Figure size as (width, height)
        palette: Seaborn color palette
        
    Example:
        >>> results = {'SMOTE': df_cv5_smote, 'Borderline-SMOTE': df_cv5_bsmote}
        >>> plot_model_comparison(results, metric="Accuracy", palette="viridis")
    """
    # Prepare data for melting
    combined_data = []
    for method, df in results_dict.items():
        df_copy = df.copy()
        df_copy['Method'] = method
        combined_data.append(df_copy)
    
    df_combined = pd.concat(combined_data, ignore_index=True)
    
    # Create plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df_combined,
        x="Classifier",
        y=metric,
        hue="Method",
        palette=palette
    )
    
    plt.title(f"Model Comparison: {metric}", fontsize=16, fontweight='bold')
    plt.ylabel(f"{metric} (%)", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.legend(title="Balancing Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
    
    plt.tight_layout()
    plt.show()


def plot_comparison_across_folds(
    df_split: pd.DataFrame,
    df_cv5: pd.DataFrame,
    df_cv10: pd.DataFrame,
    method_name: str = "SMOTE",
    figsize: Tuple[int, int] = (14, 8),
    palette: str = "viridis"
) -> None:
    """
    Compare model performance across different evaluation strategies
    (70/30 split, 5-fold CV, 10-fold CV).
    
    Args:
        df_split: Results from independent test split
        df_cv5: Results from 5-fold cross-validation
        df_cv10: Results from 10-fold cross-validation
        method_name: Name of the balancing method for title
        figsize: Figure size as (width, height)
        palette: Seaborn color palette
    """
    # Prepare summary table
    summary = pd.DataFrame({
        "70/30 Split": df_split.set_index("Classifier")["Accuracy"],
        "5-Fold CV": df_cv5.set_index("Classifier")["Accuracy"],
        "10-Fold CV": df_cv10.set_index("Classifier")["Accuracy"]
    }).reset_index()
    
    summary = summary.sort_values(by="10-Fold CV", ascending=False)
    
    # Melt for plotting
    summary_long = summary.melt(
        id_vars="Classifier",
        var_name="Evaluation Method",
        value_name="Accuracy"
    )
    
    # Create plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=summary_long,
        x="Classifier",
        y="Accuracy",
        hue="Evaluation Method",
        palette=palette
    )
    
    plt.title(
        f"Accuracy Comparison Across Evaluation Strategies ({method_name})",
        fontsize=16, fontweight='bold'
    )
    plt.ylim(80, 100)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.legend(title="Evaluation Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print(f"\n{'='*60}")
    print(f"Accuracy Summary: {method_name}")
    print(f"{'='*60}")
    print(summary.to_string(index=False))
    print(f"{'='*60}\n")


def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model (for title)
        figsize: Figure size as (width, height)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f"Confusion Matrix: {model_name}", fontsize=14, fontweight='bold')
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities for positive class
        model_name: Name of the model (for title)
        figsize: Figure size as (width, height)
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f"ROC Curve: {model_name}", fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_results_table(
    df: pd.DataFrame,
    title: str = "Evaluation Results"
) -> None:
    """
    Print a formatted results table.
    
    Args:
        df: Results DataFrame
        title: Title for the table
    """
    print(f"\n{'='*60}")
    print(title.center(60))
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"{'='*60}\n")