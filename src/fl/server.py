"""Federated Learning Server Implementation

Server-side logic for federated learning using Flower framework.
Aggregates model weights from multiple clients.
"""

import os
from typing import Tuple, Dict, Any, List
import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg
from src.models.model import LogisticRegressionModel


class FedAvgCustom(FedAvg):
    """Custom FedAvg strategy with healthcare-specific metrics tracking."""
    
    def __init__(self, *args, **kwargs):
        """Initialize custom FedAvg strategy."""
        super().__init__(*args, **kwargs)
        self.history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'loss': []
        }
    
    def aggregate_evaluate(self, rnd: int, results: List[Tuple[bool, Dict]],
        failures: List[BaseException]) -> Tuple[float, Dict]:
        """
        Aggregate evaluation metrics from clients.
        
        Args:
            rnd: Current round number
            results: List of (is_correct, metrics) tuples from clients
            failures: List of any failures
            
        Returns:
            Aggregated loss and metrics
        """
        # Aggregate metrics from successful evaluations
        if not results:
            return float('inf'), {}
        
        # Filter successful evaluations
        successful_results = [r for r in results if r[0]]
        
        if not successful_results:
            return float('inf'), {}
        
        # Aggregate loss
        total_loss = sum(r[1]['loss'] * r[1]['num_examples'] 
                        for r in successful_results)
        total_samples = sum(r[1]['num_examples'] for r in successful_results)
        aggregated_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        # Aggregate metrics
        metrics = {}
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
            if all(metric_name in r[1]['metrics'] for r in successful_results):
                weighted_sum = sum(
                    r[1]['metrics'][metric_name] * r[1]['num_examples']
                    for r in successful_results
                )
                weighted_metric = weighted_sum / total_samples
                metrics[metric_name] = weighted_metric
                
                # Track history
                if metric_name in self.history:
                    self.history[metric_name].append(weighted_metric)
        
        # Track loss
        self.history['loss'].append(aggregated_loss)
        
        return aggregated_loss, metrics


def start_fl_server(num_rounds: int = 10, num_clients: int = 5,
    initial_weights: Tuple = None) -> Dict:
    """
    Start Flower federated learning server.
    
    Args:
        num_rounds: Number of federated rounds
        num_clients: Expected number of clients
        initial_weights: Initial model weights (optional)
        
    Returns:
        History dictionary with metrics
    """
    # Define strategy
    strategy = FedAvgCustom(
        fraction_fit=1.0,  # All clients participate
        fraction_evaluate=1.0,  # Evaluate all clients
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_weights,
    )
    
    # Start server
    history = fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    return {
        'history': strategy.history,
        'final_metrics': history.metrics_centralized
    }


def create_server_config(num_rounds: int, epochs: int, 
                        batch_size: int) -> Dict[str, Any]:
    """
    Create server configuration for each round.
    
    Args:
        num_rounds: Total number of rounds
        epochs: Local epochs per client
        batch_size: Batch size for local training
        
    Returns:
        Configuration dictionary
    """
    return {
        'num_rounds': num_rounds,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': 0.01,
    }
