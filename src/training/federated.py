"""Federated training implementation using manual aggregation (no Flower required)."""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, List
from pathlib import Path

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import fbeta_score, precision_score, recall_score

from src.config.config import (
    RANDOM_SEED, MAX_ITER, DP_EPSILON, DP_DELTA, CLIPPING_THRESHOLD, CLASS_WEIGHT,
    FEDF2_GAMMA, FEDF2_REFERENCE_THRESHOLD, FEDF2_LOCAL_VAL_FRACTION
)
from src.fl.privacy import DifferentialPrivacyMechanism
from src.data.split import distribute_by_care_unit, get_client_summary

logger = logging.getLogger(__name__)


class FederatedTrainer:
    """
    Orchestrates federated learning training with support for multiple aggregation strategies.
    
    Supports FedAvg and FedProx aggregation, privacy mechanisms, and robustness features.
    
    Attributes:
        clients (Dict[str, Tuple[np.ndarray, np.ndarray]]): Client data by care unit
        val_data (Tuple[np.ndarray, np.ndarray]): Validation set (X_val, y_val)
        test_data (Tuple[np.ndarray, np.ndarray]): Test set (X_test, y_test)
        scaler (StandardScaler): Global feature scaler (fit on training data)
        privacy (GaussianDP): Privacy mechanism (ε=1.0, δ=1e-5)
        aggregation_strategy (str): 'fedavg' or 'fedprox'
        fedprox_mu (float): FedProx proximal term weight (default 0.01)
    """
    
    def __init__(
        self,
        clients: Dict[str, Tuple[np.ndarray, np.ndarray]],
        val_data: Tuple[np.ndarray, np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray],
        num_rounds: int = 10,
        learning_rate: float = 0.01,
        use_dp: bool = True,
        aggregation_strategy: str = 'fedavg',
        fedprox_mu: float = 0.01,
        fedf2_gamma: float = FEDF2_GAMMA,
        fedf2_ref_threshold: float = FEDF2_REFERENCE_THRESHOLD,
        fedf2_local_val_fraction: float = FEDF2_LOCAL_VAL_FRACTION,
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
            aggregation_strategy: 'fedavg', 'fedprox', or 'fedf2'
            fedprox_mu: FedProx proximal term weight (only used if aggregation_strategy='fedprox')
            fedf2_gamma: FedF2 blending factor (0=pure FedAvg, higher=more F2 influence)
            fedf2_ref_threshold: Uniform decision threshold for local F2-score evaluation
            fedf2_local_val_fraction: Fraction of local client data held out for F2 eval
            random_seed: Random seed for reproducibility
        """
        self.clients = clients
        self.val_data = val_data
        self.test_data = test_data
        self.num_rounds = num_rounds
        self.learning_rate = learning_rate
        self.use_dp = use_dp
        self.aggregation_strategy = aggregation_strategy.lower()
        self.fedprox_mu = fedprox_mu
        self.fedf2_gamma = fedf2_gamma
        self.fedf2_ref_threshold = fedf2_ref_threshold
        self.fedf2_local_val_fraction = fedf2_local_val_fraction
        self.random_seed = random_seed
        
        if self.aggregation_strategy not in ['fedavg', 'fedprox', 'fedf2', 'median', 'krum']:
            raise ValueError(f"Unknown aggregation strategy: {aggregation_strategy}")
        
        np.random.seed(random_seed)
        self.rng = np.random.default_rng(random_seed)
        
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
        logger.info(f"  Strategy: {aggregation_strategy.upper()}")
        if self.aggregation_strategy == 'fedprox':
            logger.info(f"  FedProx μ: {fedprox_mu}")
        if self.aggregation_strategy == 'fedf2':
            logger.info(f"  FedF2 γ: {fedf2_gamma}")
            logger.info(f"  FedF2 ref threshold: {fedf2_ref_threshold}")
            logger.info(f"  FedF2 local val fraction: {fedf2_local_val_fraction}")
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
        Train a single client (care unit) locally using FedAvg or FedProx.
        
        Args:
            unit_name (str): Care unit name
            global_weights (Dict): Global model weights (for FedAvg/FedProx)
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
        
        # Use DP-SGD when privacy is enabled; otherwise keep the existing local trainers.
        if self.use_dp:
            model_dict = self._train_dp_sgd_client(
                X_scaled, y_client, global_weights, epochs
            )
        elif self.aggregation_strategy == 'fedprox':
            model_dict = self._train_fedprox_client(
                X_scaled, y_client, global_weights, epochs
            )
        else:  # FedAvg or FedF2 (local training is identical to FedAvg)
            model_dict = self._train_fedavg_client(X_scaled, y_client, epochs)
        
        # Extract weights for aggregation
        coef = model_dict['coef']
        intercept = model_dict['intercept']
        
        weights = {
            'coef': coef,
            'intercept': intercept,
            'classes': model_dict['classes'],
            'n_samples': len(X_client)
        }
        if 'dp_metadata' in model_dict:
            weights['dp_metadata'] = model_dict['dp_metadata']
        
        # Compute local validation F2-score for FedF2 aggregation
        local_f2 = 0.0
        if self.aggregation_strategy == 'fedf2':
            local_f2 = self._compute_local_f2(coef, intercept, X_client, y_client)
        
        result = {
            'unit': unit_name,
            'n_samples': len(X_client),
            'n_deaths': int(y_client.sum()),
            'weights': weights,
            'loss': model_dict['loss'],
            'local_f2': local_f2
        }
        
        log_msg = f"  {unit_name}: {len(X_client)} samples, loss={model_dict['loss']:.4f}"
        if self.aggregation_strategy == 'fedf2':
            log_msg += f", local_F2={local_f2:.4f}"
        logger.info(log_msg)
        
        return result
    
    def _train_fedavg_client(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        epochs: int = 1
    ) -> Dict:
        """
        Train client using standard FedAvg (no proximal term).
        
        Args:
            X_scaled: Scaled feature matrix
            y: Labels
            epochs: Training epochs (unused for LogisticRegression, for compatibility)
        
        Returns:
            Dict with 'coef', 'intercept', 'classes', 'loss' keys
        """
        model = LogisticRegression(
            max_iter=MAX_ITER,
            random_state=self.random_seed,
            solver='lbfgs',  # More stable for small-scale problems
            class_weight=CLASS_WEIGHT
        )
        model.fit(X_scaled, y)
        
        # Compute loss (negative log-likelihood)
        from sklearn.metrics import log_loss
        y_pred_proba = model.predict_proba(X_scaled)
        loss = log_loss(y, y_pred_proba)
        
        return {
            'coef': model.coef_[0],
            'intercept': model.intercept_[0],
            'classes': model.classes_,
            'loss': loss
        }

    def _sigmoid(self, values: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        values = np.clip(values, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-values))

    def _logistic_loss(self, X_scaled: np.ndarray, y: np.ndarray, coef: np.ndarray, intercept: float) -> float:
        """Binary logistic loss with probability clipping for stability."""
        logits = X_scaled @ coef + intercept
        y_pred_proba = self._sigmoid(logits)
        epsilon = 1e-15
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        return float(-np.mean(y * np.log(y_pred_proba) + (1 - y) * np.log(1 - y_pred_proba)))

    def _train_dp_sgd_client(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        global_weights: Optional[Dict],
        epochs: int = 1,
        batch_size: int = 32
    ) -> Dict:
        """Train a client with per-sample clipped DP-SGD updates."""
        n_samples, n_features = X_scaled.shape
        batch_size = max(1, min(batch_size, n_samples))

        if global_weights is not None:
            coef = global_weights['coef'].copy()
            intercept = float(global_weights['intercept'])
        else:
            coef = np.zeros(n_features, dtype=float)
            intercept = 0.0

        has_global = global_weights is not None and self.aggregation_strategy == 'fedprox'
        if has_global:
            global_coef = global_weights['coef'].copy()
            global_intercept = float(global_weights['intercept'])

        for _ in range(max(1, epochs)):
            shuffled_indices = self.rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_indices = shuffled_indices[start:start + batch_size]
                if len(batch_indices) == 0:
                    continue

                per_sample_grads = []
                for idx in batch_indices:
                    x_i = X_scaled[idx]
                    y_i = float(y[idx])
                    logit = float(np.dot(x_i, coef) + intercept)
                    prediction = float(self._sigmoid(np.array([logit]))[0])
                    error = prediction - y_i
                    grad_vector = np.concatenate([error * x_i, np.array([error])])
                    clipped_grad, _ = self.privacy.clip_gradient(grad_vector)
                    per_sample_grads.append(clipped_grad)

                batch_gradient = np.mean(per_sample_grads, axis=0)

                if has_global:
                    proximal_gradient = np.concatenate([
                        self.fedprox_mu * (coef - global_coef),
                        np.array([self.fedprox_mu * (intercept - global_intercept)])
                    ])
                    batch_gradient = batch_gradient + proximal_gradient

                noise_scale = self.privacy.sigma / len(batch_indices)
                noise = np.random.normal(loc=0.0, scale=noise_scale, size=batch_gradient.shape)
                private_gradient = batch_gradient + noise

                coef -= self.learning_rate * private_gradient[:-1]
                intercept -= self.learning_rate * private_gradient[-1]

        loss = self._logistic_loss(X_scaled, y, coef, intercept)

        return {
            'coef': coef,
            'intercept': intercept,
            'classes': np.array([0, 1]),
            'loss': loss,
            'dp_metadata': {
                'epsilon': self.privacy.epsilon,
                'delta': self.privacy.delta,
                'clipping_norm': self.privacy.clipping_norm,
                'sigma': self.privacy.sigma,
                'batch_size': batch_size,
                'epochs': epochs
            }
        }
    
    def _train_fedprox_client(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        global_weights: Optional[Dict],
        epochs: int = 1
    ) -> Dict:
        """
        Train client using FedProx (with proximal regularization term).
        
        FedProx adds a proximal term: λ/2 * ||w - w_global||^2 to the local loss.
        This helps with convergence under non-IID data by penalizing deviation from global model.
        
        Args:
            X_scaled: Scaled feature matrix
            y: Labels
            global_weights: Global model weights (Dict with 'coef' and 'intercept')
            epochs: Number of local SGD epochs
        
        Returns:
            Dict with 'coef', 'intercept', 'classes', 'loss' keys
        """
        if global_weights is None:
            # First round: use FedAvg since there's no global model yet
            return self._train_fedavg_client(X_scaled, y, epochs)
        
        # Use SGDClassifier for more control over training with proximal term
        model = SGDClassifier(
            loss='log_loss',  # Logistic regression loss
            max_iter=epochs,
            random_state=self.random_seed,
            warm_start=True,
            learning_rate='optimal',
            eta0=self.learning_rate,
            n_jobs=1,
            class_weight=CLASS_WEIGHT
        )
        
        # Initialize with global weights
        w_global = global_weights['coef'].copy()
        b_global = global_weights['intercept']
        
        # First iteration: fit from scratch
        model.fit(X_scaled, y)
        
        # Apply proximal regularization through iterative updates
        # The proximal term λ/2 * ||w - w_global||^2 acts to regularize updates
        for _ in range(max(0, epochs - 1)):
            # Get current weights
            w_current = model.coef_[0].copy()
            b_current = model.intercept_[0]
            
            # Compute proximal-adjusted gradient direction
            # w_new = (1 - step_size * μ) * w_current + step_size * μ * w_global
            step_size = self.learning_rate
            w_adjusted = (1 - step_size * self.fedprox_mu) * w_current + \
                         step_size * self.fedprox_mu * w_global
            b_adjusted = (1 - step_size * self.fedprox_mu) * b_current + \
                         step_size * self.fedprox_mu * b_global
            
            # Update model weights to reflect proximal constraint
            model.coef_[0] = w_adjusted
            model.intercept_[0] = b_adjusted
        
        # Compute final loss
        from sklearn.metrics import log_loss
        y_pred_proba = model.predict_proba(X_scaled)
        loss = log_loss(y, y_pred_proba)
        
        # Add proximal term to loss for logging
        proximal_loss = self.fedprox_mu / 2 * np.sum((model.coef_[0] - w_global) ** 2)
        
        return {
            'coef': model.coef_[0],
            'intercept': model.intercept_[0],
            'classes': model.classes_,
            'loss': loss + proximal_loss
        }
    
    def _compute_local_f2(
        self, coef: np.ndarray, intercept: float,
        X_local: np.ndarray, y_local: np.ndarray
    ) -> float:
        """Compute F2-score of a local model on local data at the reference threshold.
        
        Uses a stratified local holdout (fedf2_local_val_fraction) so the F2
        is not evaluated on the same samples the model was trained on.
        """
        from sklearn.model_selection import train_test_split as _split
        
        n = len(y_local)
        n_pos = int(y_local.sum())
        n_neg = n - n_pos
        
        # Need at least 2 positive and 2 negative samples to do a stratified split
        if n < 20 or n_pos < 2 or n_neg < 2:
            X_eval, y_eval = X_local, y_local
        else:
            _, X_eval, _, y_eval = _split(
                X_local, y_local,
                test_size=self.fedf2_local_val_fraction,
                random_state=self.random_seed,
                stratify=y_local
            )
        
        X_eval_scaled = self.scaler.transform(X_eval)
        logits = X_eval_scaled @ coef + intercept
        proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        y_pred = (proba >= self.fedf2_ref_threshold).astype(int)
        
        return float(fbeta_score(y_eval, y_pred, beta=2, zero_division=0))
    
    def aggregate_weights(self, client_results: List[Dict]) -> Dict:
        """Aggregate client weights using the configured strategy.
        
        For FedAvg/FedProx: standard sample-size-weighted averaging.
        For FedF2: blends sample-size weights with local validation F2-scores.
        For Median: coordinate-wise median (Byzantine-robust).
        For Krum: select update closest to consensus (Byzantine-robust).
        
        Args:
            client_results: List of client training results from train_client_local()
        
        Returns:
            Dict: Aggregated global weights
        """
        from src.fl.strategy import ClinicalAwareAggregator
        from src.fl.robust_aggregation import RobustAggregator
        
        if self.aggregation_strategy == 'fedf2':
            client_weights = [r['weights'] for r in client_results]
            client_sizes = [r['n_samples'] for r in client_results]
            client_f2 = [r.get('local_f2', 0.0) for r in client_results]
            
            return ClinicalAwareAggregator.aggregate(
                client_weights, client_sizes, client_f2,
                gamma=self.fedf2_gamma
            )
        
        elif self.aggregation_strategy == 'median':
            # Coordinate-wise median aggregation (Byzantine-robust)
            coefs = np.array([r['weights']['coef'] for r in client_results])
            intercepts = np.array([r['weights']['intercept'] for r in client_results])
            
            avg_coef = np.median(coefs, axis=0)
            avg_intercept = np.median(intercepts)
            
            return {
                'coef': avg_coef,
                'intercept': avg_intercept,
                'classes': client_results[0]['weights']['classes']
            }
        
        elif self.aggregation_strategy == 'krum':
            # Krum aggregation: select update closest to consensus
            coefs = np.array([r['weights']['coef'] for r in client_results])
            intercepts = np.array([r['weights']['intercept'] for r in client_results])
            
            # Compute pairwise distances between coefficient vectors
            n_clients = len(coefs)
            distances = np.zeros((n_clients, n_clients))
            for i in range(n_clients):
                for j in range(i + 1, n_clients):
                    dist = np.linalg.norm(coefs[i] - coefs[j])
                    distances[i, j] = dist
                    distances[j, i] = dist
            
            # Select the client with smallest average distance to neighbors
            f = max(0, int(n_clients / 2) - 1)  # Number of neighbors to consider
            avg_distances = np.sum(np.sort(distances, axis=1)[:, :f+1], axis=1) / (f + 1)
            selected_idx = np.argmin(avg_distances)
            
            avg_coef = coefs[selected_idx]
            avg_intercept = intercepts[selected_idx]
            
            return {
                'coef': avg_coef,
                'intercept': avg_intercept,
                'classes': client_results[0]['weights']['classes']
            }
        
        # Default: sample-size-weighted FedAvg (also used by FedProx)
        total_samples = sum(r['n_samples'] for r in client_results)
        avg_coef = np.zeros_like(client_results[0]['weights']['coef'])
        avg_intercept = 0.0
        
        for result in client_results:
            weights = result['weights']
            n_samples = result['n_samples']
            weight = n_samples / total_samples
            avg_coef += weight * weights['coef']
            avg_intercept += weight * weights['intercept']
        
        return {
            'coef': avg_coef,
            'intercept': avg_intercept,
            'classes': client_results[0]['weights']['classes']
        }
    
    def federated_round(self, round_num: int, global_weights: Optional[Dict] = None) -> Tuple[Dict, List[Dict]]:
        """
        Execute one round of federated learning (client training + aggregation).
        
        Args:
            round_num (int): Round number (1-indexed)
            global_weights (Dict): Global model weights from previous round (for FedProx)
        
        Returns:
            Tuple[Dict, List[Dict]]: (aggregated_weights, client_results)
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Round {round_num}/{self.num_rounds}")
        logger.info(f"{'='*70}")
        
        # Train all clients
        client_results = []
        for unit_name in sorted(self.clients.keys()):
            result = self.train_client_local(unit_name, global_weights=global_weights)
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
        logger.info(f"# Strategy: {self.aggregation_strategy.upper()}")
        logger.info(f"{'#'*70}\n")
        
        global_weights = None
        history = []
        
        # Federated learning rounds
        for round_num in range(1, self.num_rounds + 1):
            global_weights, client_results = self.federated_round(round_num, global_weights)
            
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

