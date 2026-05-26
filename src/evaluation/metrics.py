"""Evaluation metrics for machine learning models

Comprehensive metrics including accuracy, precision, recall, F1-score,
confusion matrix, calibration metrics, and healthcare-specific metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    brier_score_loss
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


def calculate_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate Precision-Recall AUC (AUPRC) and curve.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities/scores
        
    Returns:
        Tuple of (AUPRC score, precision, recall)
    """
    from sklearn.metrics import precision_recall_curve, auc
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc_score = auc(recall, precision)
    
    return float(pr_auc_score), precision, recall


def calculate_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Calculate Brier Score: Mean Squared Error between predicted probabilities and true labels.
    
    Lower is better. Range: [0, 1]. 
    - 0: perfect calibration
    - 0.25: random guessing (binary classification)
    - 1: worst case
    
    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities for class 1
        
    Returns:
        Brier score (float)
    """
    return float(brier_score_loss(y_true, y_proba))


def calculate_expected_calibration_error(
    y_true: np.ndarray, 
    y_proba: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    Measures the difference between predicted probability and actual frequency 
    within probability bins. Lower is better (0 = perfect calibration).
    
    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities for class 1
        n_bins: Number of bins for calibration curve (default: 10)
        
    Returns:
        ECE score (float, range [0, 1])
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (y_proba >= bin_edges[i]) & (y_proba <= bin_edges[i + 1])
        
        if mask.sum() == 0:
            continue
        
        # Actual frequency (empirical probability)
        acc = y_true[mask].mean()
        
        # Predicted probability (average in bin)
        conf = y_proba[mask].mean()
        
        # Weight by number of samples in bin
        weight = mask.sum() / total_samples
        ece += weight * np.abs(acc - conf)
    
    return float(ece)


def calculate_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Calculate all calibration metrics at once.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities for class 1
        n_bins: Number of bins for ECE calculation (default: 10)
        
    Returns:
        Dictionary with calibration metrics:
        - brier_score: MSE between probabilities and labels
        - ece: Expected Calibration Error
    """
    return {
        'brier_score': calculate_brier_score(y_true, y_proba),
        'expected_calibration_error': calculate_expected_calibration_error(
            y_true, y_proba, n_bins=n_bins
        )
    }


def compute_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute calibration curve (reliability diagram) data.
    
    Returns pairs of (mean_predicted_proba, fraction_of_positives) for each bin.
    Perfect calibration follows the diagonal y=x.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities for class 1
        n_bins: Number of bins (default: 10)
        
    Returns:
        Tuple of (mean_predicted_proba, fraction_of_positives) for each bin
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    probabilities = []
    fractions = []
    
    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (y_proba >= bin_edges[i]) & (y_proba <= bin_edges[i + 1])
        
        if mask.sum() == 0:
            continue
        
        # Mean predicted probability in bin
        probabilities.append(y_proba[mask].mean())
        
        # Actual fraction of positives in bin
        fractions.append(y_true[mask].mean())
    
    return np.array(probabilities), np.array(fractions)
