"""Visualization utilities for model evaluation

Creates plots for comparing models, tracking FL rounds, and healthcare metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
import seaborn as sns


def plot_metric_over_rounds(metric_values: List[float], metric_name: str,
    save_path: Optional[str] = None) -> None:
    """
    Plot metric values over federated learning rounds.
    
    Args:
        metric_values: List of metric values per round
        metric_name: Name of the metric to plot
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 6))
    rounds = list(range(1, len(metric_values) + 1))
    plt.plot(rounds, metric_values, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'{metric_name} over Federated Learning Rounds', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_multiple_metrics(metrics_dict: Dict[str, List[float]],
    save_path: Optional[str] = None) -> None:
    """
    Plot multiple metrics on the same figure.
    
    Args:
        metrics_dict: Dictionary of metric names to values over rounds
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(12, 7))
    
    for metric_name, values in metrics_dict.items():
        rounds = list(range(1, len(values) + 1))
        plt.plot(rounds, values, marker='o', label=metric_name, linewidth=2, markersize=6)
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Metrics over Federated Learning Rounds', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, labels: List[str] = None,
    save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix (2D numpy array)
        labels: Class labels (default: ['Negative', 'Positive'])
        save_path: Path to save figure (optional)
    """
    if labels is None:
        labels = ['Negative', 'Positive']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_fl_vs_centralized(fl_metrics: Dict, centralized_metrics: Dict,
    metric_names: List[str] = None,
    save_path: Optional[str] = None) -> None:
    """
    Compare federated learning vs centralized training.
    
    Args:
        fl_metrics: Dictionary of FL metrics
        centralized_metrics: Dictionary of centralized training metrics
        metric_names: Specific metrics to compare (default: all common metrics)
        save_path: Path to save figure (optional)
    """
    if metric_names is None:
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Filter to only available metrics
    available_metrics = [m for m in metric_names 
                        if m in fl_metrics and m in centralized_metrics]
    
    fl_values = [fl_metrics[m] for m in available_metrics]
    centralized_values = [centralized_metrics[m] for m in available_metrics]
    
    x = np.arange(len(available_metrics))
    width = 0.35
    
    plt.figure(figsize=(12, 7))
    plt.bar(x - width/2, centralized_values, width, label='Centralized', alpha=0.8)
    plt.bar(x + width/2, fl_values, width, label='Federated Learning', alpha=0.8)
    
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Federated Learning vs Centralized Training', fontsize=14, fontweight='bold')
    plt.xticks(x, available_metrics, rotation=45)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim([0, 1.1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_healthcare_metrics(metrics_dict: Dict[str, float],
    save_path: Optional[str] = None) -> None:
    """
    Plot healthcare-specific metrics.
    
    Visualizes sensitivity, specificity, PPV, and NPV.
    
    Args:
        metrics_dict: Dictionary with healthcare metrics
        save_path: Path to save figure (optional)
    """
    healthcare_metrics = ['sensitivity', 'specificity', 'ppv', 'npv']
    values = [metrics_dict.get(m, 0) for m in healthcare_metrics]
    
    # Pretty labels
    labels = ['Sensitivity\n(Recall)', 'Specificity', 'PPV\n(Precision)', 'NPV']
    
    plt.figure(figsize=(10, 6))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = plt.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel('Score', fontsize=12)
    plt.title('Healthcare Evaluation Metrics', fontsize=14, fontweight='bold')
    plt.ylim([0, 1.1])
    plt.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float,
    save_path: Optional[str] = None) -> None:
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc: Area under curve score
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_client_convergence(client_accuracies: Dict[int, List[float]],
    save_path: Optional[str] = None) -> None:
    """
    Plot convergence of different clients over rounds.
    
    Args:
        client_accuracies: Dictionary mapping client ID to accuracy list
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(12, 7))
    
    for client_id, accuracies in client_accuracies.items():
        rounds = list(range(1, len(accuracies) + 1))
        plt.plot(rounds, accuracies, marker='o', label=f'Client {client_id}', linewidth=2)
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Client Convergence over Federated Learning Rounds', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
