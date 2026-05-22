"""Federated training implementation using manual aggregation (no Flower required)."""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, List
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config.config import (
    RANDOM_SEED, MAX_ITER, DP_EPSILON, DP_DELTA, CLIPPING_THRESHOLD
)
from src.fl.privacy import DifferentialPrivacyMechanism
from src.data.split import distribute_by_care_unit, get_client_summary

logger = logging.getLogger(__name__)


class FederatedTrainer:
    """
    Orchestrates federated learning training with Flower framework.
    
    Supports multiple aggregation strategies, privacy mechanisms, and robustness features.
    
    Attributes:
        clients (Dict[str, Tuple[np.ndarray, np.ndarray]]): Client data by care unit
        val_data (Tuple[np.ndarray, np.ndarray]): Validation set (X_val, y_val)
        test_data (Tuple[np.ndarray, np.ndarray]): Test set (X_test, y_test)
        scaler (StandardScaler): Global feature scaler (fit on training data)
        privacy (GaussianDP): Privacy mechanism (ε=1.0, δ=1e-5)
    """
    
    def __init__(
        self,
        clients: Dict[str, Tuple[np.ndarray, np.ndarray]],
        val_data: Tuple[np.ndarray, np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray],
        num_rounds: int = 10,
        learning_rate: float = 0.01,
        use_dp: bool = True,
        random_seed: int = RANDOM_SEED
    ):
        """
        Initialize federated trainer.
        
        Args:
            clients: Client data dictionary from distribute_by_care_unit()
            val_data: (X_val, y_val) tuple
            test_data: (X_test, y_test) tuple
            num_rounds: Number of federated learning rounds
            learning_rate: Learning rate for client-side training
            use_dp: Whether to apply differential privacy
            random_seed: Random seed for reproducibility
        """
        self.clients = clients
        self.val_data = val_data
        self.test_data = test_data
        self.num_rounds = num_rounds
        self.learning_rate = learning_rate
        self.use_dp = use_dp
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        
        # Fit global scaler on training data
        all_X = np.vstack([X for X, y in clients.values()])
        self.scaler = StandardScaler()
        self.scaler.fit(all_X)
        
        # Initialize privacy mechanism
        self.privacy = DifferentialPrivacyMechanism(
            epsilon=DP_EPSILON,
            delta=DP_DELTA,
            clipping_norm=CLIPPING_THRESHOLD
        ) if use_dp else None
        
        # Log setup
        n_clients = len(clients)
        n_train = all_X.shape[0]
        n_val, n_test = val_data[0].shape[0], test_data[0].shape[0]
        
        logger.info(f"FederatedTrainer initialized:")
        logger.info(f"  Clients: {n_clients}")
        logger.info(f"  Training samples: {n_train}")
        logger.info(f"  Validation samples: {n_val}")
        logger.info(f"  Test samples: {n_test}")
        logger.info(f"  Features: {all_X.shape[1]}")
        logger.info(f"  Rounds: {num_rounds}")
        logger.info(f"  Privacy: {'Enabled (ε=1.0, δ=1e-5)' if use_dp else 'Disabled'}")
    
    def get_client_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for all federated clients.
        
        Returns:
            pd.DataFrame: Client statistics (unit name, n_patients, mortality, etc.)
        """
        return get_client_summary(self.clients)
    
    def train_client_local(
        self,
        unit_name: str,
        global_weights: Optional[Dict] = None,
        epochs: int = 1
    ) -> Dict:
        """
        Train a single client (care unit) locally.
        
        Args:
            unit_name (str): Care unit name
            global_weights (Dict): Global model weights (for FedAvg)
            epochs (int): Local training epochs
        
        Returns:
            Dict: {
                'unit': unit_name,
                'n_samples': int,
                'n_deaths': int,
                'weights': dict (scaled weights if DP enabled),
                'loss': float,
                'coef': np.ndarray (unscaled for aggregation)
            }
        """
        if unit_name not in self.clients:
            raise ValueError(f"Unknown care unit: {unit_name}")
        
        X_client, y_client = self.clients[unit_name]
        X_scaled = self.scaler.transform(X_client)
        
        # Initialize or load model
        model = LogisticRegression(
            max_iter=MAX_ITER,
            random_state=self.random_seed
        )
        
        # If global weights provided, set them (warm start for FedAvg)
        if global_weights is not None and 'coef' in global_weights:
            model.coef_ = global_weights['coef'].reshape(1, -1)
            model.intercept_ = global_weights['intercept']
            model.classes_ = global_weights['classes']
        
        # Train
        model.fit(X_scaled, y_client)
        
        # Compute loss
        loss = -model.score(X_scaled, y_client)  # Negative because sklearn uses accuracy
        
        # Extract weights for aggregation
        weights = {
            'coef': model.coef_[0],
            'intercept': model.intercept_[0],
            'classes': model.classes_,
            'n_samples': len(X_client)
        }
        
        # Apply DP if enabled (add noise BEFORE aggregation)
        if self.use_dp:
            weights['coef_dp'], _ = self.privacy.add_noise(weights['coef'])
            intercept_noisy, _ = self.privacy.add_noise(np.array([weights['intercept']]))
            weights['intercept_dp'] = intercept_noisy[0]
        
        result = {
            'unit': unit_name,
            'n_samples': len(X_client),
            'n_deaths': int(y_client.sum()),
            'weights': weights,
            'loss': loss
        }
        
        logger.info(
            f"  {unit_name}: {len(X_client)} samples, loss={loss:.4f}"
        )
        
        return result
    
    def aggregate_weights(self, client_results: List[Dict]) -> Dict:
        """
        Federated averaging (FedAvg) of client weights.
        
        Args:
            client_results: List of client training results from train_client_local()
        
        Returns:
            Dict: Aggregated global weights
        """
        total_samples = sum(r['n_samples'] for r in client_results)
        
        # Weighted average of coefficients
        avg_coef = np.zeros_like(client_results[0]['weights']['coef'])
        avg_intercept = 0.0
        
        for result in client_results:
            weights = result['weights']
            n_samples = result['n_samples']
            weight = n_samples / total_samples
            
            # Use DP weights if available, else use raw weights
            if 'coef_dp' in weights:
                avg_coef += weight * weights['coef_dp']
                avg_intercept += weight * weights['intercept_dp']
            else:
                avg_coef += weight * weights['coef']
                avg_intercept += weight * weights['intercept']
        
        return {
            'coef': avg_coef,
            'intercept': avg_intercept,
            'classes': client_results[0]['weights']['classes']
        }
    
    def federated_round(self, round_num: int) -> Tuple[Dict, List[Dict]]:
        """
        Execute one round of federated learning (client training + aggregation).
        
        Args:
            round_num (int): Round number (1-indexed)
        
        Returns:
            Tuple[Dict, List[Dict]]: (aggregated_weights, client_results)
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Round {round_num}/{self.num_rounds}")
        logger.info(f"{'='*70}")
        
        # Train all clients
        client_results = []
        for unit_name in sorted(self.clients.keys()):
            result = self.train_client_local(unit_name)
            client_results.append(result)
        
        # Aggregate
        global_weights = self.aggregate_weights(client_results)
        
        logger.info(f"Round {round_num} complete: {len(client_results)} clients aggregated")
        
        return global_weights, client_results
    
    def train(self) -> Dict:
        """
        Execute full federated training loop (num_rounds rounds).
        
        Returns:
            Dict: {
                'final_weights': dict,
                'history': list of round results,
                'val_auroc': float,
                'test_auroc': float
            }
        """
        logger.info(f"\n{'#'*70}")
        logger.info(f"# FEDERATED LEARNING TRAINING ({self.num_rounds} rounds)")
        logger.info(f"{'#'*70}\n")
        
        global_weights = None
        history = []
        
        # Federated learning rounds
        for round_num in range(1, self.num_rounds + 1):
            global_weights, client_results = self.federated_round(round_num)
            
            # Evaluate on validation set
            val_auroc = self.evaluate(global_weights, self.val_data)
            
            history.append({
                'round': round_num,
                'n_clients': len(client_results),
                'val_auroc': val_auroc,
                'client_results': client_results
            })
            
            logger.info(f"Round {round_num} Validation AUROC: {val_auroc:.4f}")
        
        # Final evaluation
        test_auroc = self.evaluate(global_weights, self.test_data)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Final Test AUROC: {test_auroc:.4f}")
        logger.info(f"{'='*70}\n")
        
        return {
            'final_weights': global_weights,
            'history': history,
            'val_auroc': history[-1]['val_auroc'] if history else None,
            'test_auroc': test_auroc
        }
    
    def evaluate(self, weights: Dict, data: Tuple[np.ndarray, np.ndarray]) -> float:
        """
        Evaluate global model on a dataset.
        
        Args:
            weights (Dict): Model weights
            data (Tuple[np.ndarray, np.ndarray]): (X, y) dataset
        
        Returns:
            float: AUROC score
        """
        from sklearn.metrics import roc_auc_score
        
        X, y = data
        X_scaled = self.scaler.transform(X)
        
        model = LogisticRegression(max_iter=1000)
        model.coef_ = weights['coef'].reshape(1, -1)
        model.intercept_ = np.array([weights['intercept']]) if isinstance(weights['intercept'], (int, float)) else weights['intercept']
        model.classes_ = weights['classes']
        
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        auroc = roc_auc_score(y, y_pred_proba)
        
        return auroc

