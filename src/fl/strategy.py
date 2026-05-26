"""Federated Learning Aggregation Strategies

Implements FedAvg and other aggregation strategies for federated learning.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional


class FedAvgAggregator:
    """Federated Averaging (FedAvg) for aggregating model weights."""
    
    @staticmethod
    def aggregate(client_weights: List[Dict], client_sizes: List[int]) -> Dict:
        """
        Aggregate weights from multiple clients using FedAvg.
        
        FedAvg: weighted average of client weights, proportional to local dataset size.
        Formula: w_t+1 = sum(n_k / n * w_k^t) for all clients k
        where n_k = local dataset size, n = total dataset size
        
        Args:
            client_weights: List of weight dictionaries from each client
            Each dict contains 'coef', 'intercept', 'classes'
            client_sizes: List of local dataset sizes for each client
            
        Returns:
            Aggregated weights as dictionary with 'coef', 'intercept', 'classes'
        """
        if not client_weights or not client_sizes:
            raise ValueError("No client weights or sizes provided")
        
        if len(client_weights) != len(client_sizes):
            raise ValueError("Mismatch between number of clients and sizes")
        
        # Calculate total samples
        total_samples = sum(client_sizes)
        
        # Initialize aggregated weights
        aggregated = {
            'coef': np.zeros_like(client_weights[0]['coef'], dtype=np.float32),
            'intercept': np.zeros_like(client_weights[0]['intercept'], dtype=np.float32),
            'classes': client_weights[0]['classes'].copy()  # Classes are same for all clients
        }
        
        # Weighted average aggregation
        for client_idx, weights in enumerate(client_weights):
            weight = client_sizes[client_idx] / total_samples
            aggregated['coef'] += weight * weights['coef']
            aggregated['intercept'] += weight * weights['intercept']
        
        return aggregated
    
    @staticmethod
    def aggregate_metrics(client_metrics: List[Dict], 
        client_sizes: List[int]) -> Dict[str, float]:
        """
        Aggregate evaluation metrics from clients.
        
        Args:
            client_metrics: List of metrics dictionaries from each client
            client_sizes: List of local dataset sizes for each client
            
        Returns:
            Aggregated metrics dictionary
        """
        if not client_metrics or not client_sizes:
            return {}
        
        total_samples = sum(client_sizes)
        aggregated_metrics = {}
        
        # Get all metric names
        metric_names = set()
        for metrics in client_metrics:
            metric_names.update(metrics.keys())
        
        # Aggregate each metric
        for metric_name in metric_names:
            if metric_name == 'num_samples':
                aggregated_metrics[metric_name] = total_samples
            else:
                weighted_sum = 0.0
                for client_idx, metrics in enumerate(client_metrics):
                    if metric_name in metrics:
                        weight = client_sizes[client_idx] / total_samples
                        weighted_sum += weight * metrics[metric_name]
                
                aggregated_metrics[metric_name] = weighted_sum
        
        return aggregated_metrics


class FedProxAggregator:
    """Federated Proximal (FedProx) for handling statistical heterogeneity."""
    
    @staticmethod
    def aggregate(client_weights: List[Dict], client_sizes: List[int],
        global_weights: Optional[Dict] = None,
        mu: float = 0.01) -> Dict:
        """
        Aggregate weights using FedProx (adds proximal term).
        
        Helps with non-IID data by adding regularization.
        
        Args:
            client_weights: List of weight dictionaries from each client
            client_sizes: List of local dataset sizes for each client
            global_weights: Previous global weights (for proximal term)
            mu: Proximal term coefficient
            
        Returns:
            Aggregated weights as dictionary
        """
        # Use standard FedAvg aggregation
        aggregated = FedAvgAggregator.aggregate(client_weights, client_sizes)
        
        # Apply proximal regularization if global weights provided
        if global_weights is not None:
            # Proximal term: w = w - mu * (w - w_global)
            aggregated['coef'] = aggregated['coef'] - mu * (aggregated['coef'] - global_weights['coef'])
            aggregated['intercept'] = aggregated['intercept'] - mu * (aggregated['intercept'] - global_weights['intercept'])
        
        return aggregated


class ClinicalAwareAggregator:
    """Clinical-Sensitivity-Aware Federated Aggregation (FedF2).
    
    Blends standard sample-size weighting with local validation F2-score
    weighting.  The F2-score (beta=2) weights recall twice as heavily as
    precision, making it clinically appropriate for mortality prediction
    where missed diagnoses are far costlier than false alarms.
    
    Aggregation weight for client k at round t:
        alpha_k = (1 - gamma) * (n_k / n) + gamma * (F2_k / sum(F2_j))
    
    When gamma = 0 this reduces to standard FedAvg.
    When gamma > 0 clients with better clinical sensitivity receive higher
    aggregation weight, while degenerate all-positive classifiers are
    penalised because their precision (and therefore F2) is low.
    """
    
    @staticmethod
    def aggregate(
        client_weights: List[Dict],
        client_sizes: List[int],
        client_f2_scores: List[float],
        gamma: float = 0.3
    ) -> Dict:
        """Aggregate weights using a blend of sample-size and F2-score weighting.
        
        Args:
            client_weights: List of weight dicts (coef, intercept, classes)
            client_sizes:   List of local dataset sizes for each client
            client_f2_scores: List of local validation F2-scores for each client
            gamma: Blending factor in [0, 1). 0 = pure FedAvg.
            
        Returns:
            Aggregated weights as dictionary
        """
        if not client_weights or not client_sizes or not client_f2_scores:
            raise ValueError("Missing weights, sizes, or F2 scores for FedF2 aggregation")
        
        if len(client_weights) != len(client_sizes) or len(client_weights) != len(client_f2_scores):
            raise ValueError(
                f"Length mismatch: {len(client_weights)} weights, "
                f"{len(client_sizes)} sizes, {len(client_f2_scores)} F2 scores"
            )
        
        n_clients = len(client_weights)
        total_samples = sum(client_sizes)
        total_f2 = sum(client_f2_scores)
        
        # Fall back to equal F2 weighting if all scores are zero
        if total_f2 == 0:
            f2_weights = [1.0 / n_clients] * n_clients
        else:
            f2_weights = [f2 / total_f2 for f2 in client_f2_scores]
        
        # Combined weight: alpha_k = (1 - gamma) * (n_k / n) + gamma * (F2_k / sum(F2))
        alpha = []
        for idx in range(n_clients):
            sample_w = client_sizes[idx] / total_samples
            f2_w = f2_weights[idx]
            alpha.append((1 - gamma) * sample_w + gamma * f2_w)
        
        # Normalise to sum to 1.0 (guards against floating-point drift)
        sum_alpha = sum(alpha)
        alpha = [a / sum_alpha for a in alpha]
        
        # Weighted aggregation
        aggregated = {
            'coef': np.zeros_like(client_weights[0]['coef'], dtype=np.float64),
            'intercept': np.zeros_like(client_weights[0]['intercept'], dtype=np.float64)
                        if isinstance(client_weights[0]['intercept'], np.ndarray)
                        else 0.0,
            'classes': client_weights[0]['classes'].copy()
        }
        
        for idx, weights in enumerate(client_weights):
            aggregated['coef'] += alpha[idx] * weights['coef']
            if isinstance(aggregated['intercept'], np.ndarray):
                aggregated['intercept'] += alpha[idx] * weights['intercept']
            else:
                aggregated['intercept'] += alpha[idx] * float(weights['intercept'])
        
        return aggregated


def aggregate_weights(weights_list: List[Dict], 
    sample_counts: List[int],
    strategy: str = 'fedavg',
    **kwargs) -> Dict:
    """
    Aggregate model weights from multiple clients.
    
    Args:
        weights_list: List of weight dictionaries from each client
        sample_counts: List of local dataset sizes for each client
        strategy: Aggregation strategy ('fedavg', 'fedprox', or 'fedf2')
        **kwargs: Additional arguments for specific strategies.
                  For 'fedf2': client_f2_scores (List[float]), gamma (float)
        
    Returns:
        Aggregated weights as dictionary
    """
    if strategy == 'fedavg':
        return FedAvgAggregator.aggregate(weights_list, sample_counts)
    elif strategy == 'fedprox':
        return FedProxAggregator.aggregate(weights_list, sample_counts, **kwargs)
    elif strategy == 'fedf2':
        return ClinicalAwareAggregator.aggregate(weights_list, sample_counts, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def aggregate_metrics(metrics_list: List[Dict],
    sample_counts: List[int]) -> Dict[str, float]:
    """
    Aggregate evaluation metrics from multiple clients.
    
    Args:
        metrics_list: List of metrics dictionaries from each client
        sample_counts: List of local dataset sizes for each client
        
    Returns:
        Aggregated metrics dictionary
    """
    return FedAvgAggregator.aggregate_metrics(metrics_list, sample_counts)
