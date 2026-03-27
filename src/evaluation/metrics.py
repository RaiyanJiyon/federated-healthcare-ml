"""Evaluation metrics for machine learning models

Comprehensive metrics including accuracy, precision, recall, F1-score,
confusion matrix, and healthcare-specific metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from typing import Dict, Tuple


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     zero_division: int = 0) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        zero_division: Value to return when division by zero (default: 0)
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=zero_division)),
        'recall': float(recall_score(y_true, y_pred, zero_division=zero_division)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=zero_division)),
    }
    
    return metrics


def calculate_confusion_matrix(y_true: np.ndarray, 
                               y_pred: np.ndarray) -> Dict[str, int]:
    """
    Calculate confusion matrix components.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with TP, TN, FP, FN
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'TP': int(tp),     # True Positives
        'TN': int(tn),     # True Negatives
        'FP': int(fp),     # False Positives (Type I error)
        'FN': int(fn),     # False Negatives (Type II error)
        'confusion_matrix': cm.tolist()
    }


def calculate_healthcare_metrics(y_true: np.ndarray, 
                                y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate healthcare-specific metrics.
    
    Prioritizes recall (sensitivity) and specificity for clinical safety.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with healthcare metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Sensitivity (Recall): ability to identify positive cases
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Specificity: ability to identify negative cases
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Positive Predictive Value (Precision): quality of positive predictions
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Negative Predictive Value: quality of negative predictions
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Matthews Correlation Coefficient: balanced metric for binary classification
    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0
    
    return {
        'sensitivity': float(sensitivity),      # Also called recall/true positive rate
        'specificity': float(specificity),      # True negative rate
        'ppv': float(ppv),                      # Positive predictive value (precision)
        'npv': float(npv),                      # Negative predictive value
        'mcc': float(mcc),                      # Matthews correlation coefficient
    }


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Calculate all available metrics at once.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with standard metrics, confusion matrix, and healthcare metrics
    """
    metrics = calculate_metrics(y_true, y_pred)
    cm_dict = calculate_confusion_matrix(y_true, y_pred)
    healthcare_metrics = calculate_healthcare_metrics(y_true, y_pred)
    
    all_metrics = {
        **metrics,
        **cm_dict,
        **healthcare_metrics
    }
    
    return all_metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Generate and return classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Classification report as string
    """
    return classification_report(y_true, y_pred, 
                                 target_names=['Negative', 'Positive'])


def calculate_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate ROC-AUC score and curve.
    
    Args:
        y_true: True labels
        y_scores: Predicted probabilities/scores
        
    Returns:
        Tuple of (AUC score, FPR, TPR)
    """
    auc_score = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    return auc_score, fpr, tpr
