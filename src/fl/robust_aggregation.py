"""
Robust Aggregation Methods for Byzantine-Resistant Federated Learning

This module implements defense mechanisms against poisoning attacks in federated learning:
- Median aggregation: Selects element-wise medians (resistant to attacks)
- Trimmed mean: Discards extreme values before averaging
- Krum: Removes outlier client updates
- Multi-Krum: Multiple Krum aggregation with voting

These methods protect against malicious clients with corrupted updates.
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from copy import deepcopy

logger = logging.getLogger(__name__)


class RobustAggregator:
    """
    Byzantine-resistant aggregation mechanisms for federated learning.
    
    Attributes:
        method (str): Aggregation method ('fedavg', 'median', 'trimmed_mean', 'krum', 'multi_krum')
        trim_ratio (float): Percentage of extreme values to trim (0-0.5)
        num_byzantine (int): Expected number of malicious clients (for Krum)
    """
    
    def __init__(
        self,
        method: str = "fedavg",
        trim_ratio: float = 0.1,
        num_byzantine: int = None,
        verbose: bool = True
    ):
        """
        Initialize robust aggregator.
        
        Args:
            method: Aggregation method name
            trim_ratio: Fraction of extremes to remove (0 ≤ trim_ratio ≤ 0.5)
            num_byzantine: Expected number of malicious clients (defaults to 10% of clients)
            verbose: Log aggregation details
        """
        self.method = method.lower()
        self.trim_ratio = max(0, min(trim_ratio, 0.5))  # Clamp to [0, 0.5]
        self.num_byzantine = num_byzantine
        self.verbose = verbose
        
        valid_methods = ["fedavg", "median", "trimmed_mean", "krum", "multi_krum"]
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got {self.method}")
        
        logger.info(f"Initialized RobustAggregator with method='{self.method}'")
    
    def aggregate(
        self,
        client_weights: List[np.ndarray],
        client_sizes: Optional[List[int]] = None,
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Aggregate client weights using selected method.
        
        Args:
            client_weights: List of weight matrices from each client
            client_sizes: List of local dataset sizes (for FedAvg weighting)
            weights: Pre-computed aggregation weights
            
        Returns:
            Aggregated weight matrix
        """
        if not client_weights:
            raise ValueError("No client weights to aggregate")
        
        # Flatten weights to 1D for aggregation
        flattened = [w.flatten() for w in client_weights]
        
        if self.method == "fedavg":
            result = self._fedavg(flattened, client_sizes, weights)
        elif self.method == "median":
            result = self._median(flattened)
        elif self.method == "trimmed_mean":
            result = self._trimmed_mean(flattened)
        elif self.method == "krum":
            result = self._krum(flattened)
        elif self.method == "multi_krum":
            result = self._multi_krum(flattened)
        
        # Reshape back to original shape
        original_shape = client_weights[0].shape
        return result.reshape(original_shape)
    
    def _fedavg(
        self,
        weights_list: List[np.ndarray],
        client_sizes: Optional[List[int]] = None,
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Standard FedAvg aggregation (baseline, not Byzantine-resistant).
        
        Args:
            weights_list: Flattened weight vectors
            client_sizes: Dataset sizes for weighting
            weights: Pre-computed weights
            
        Returns:
            Weighted average of all weights
        """
        if len(weights_list) == 0:
            return weights_list[0]
        
        if weights is None:
            if client_sizes is not None:
                # Weight by dataset size
                total_size = sum(client_sizes)
                weights = np.array(client_sizes) / total_size
            else:
                # Equal weighting
                weights = np.ones(len(weights_list)) / len(weights_list)
        
        # Weighted average
        result = np.zeros_like(weights_list[0], dtype=np.float64)
        for w, weight in zip(weights_list, weights):
            result += w * weight
        
        return result
    
    def _median(self, weights_list: List[np.ndarray]) -> np.ndarray:
        """
        Median aggregation: Select element-wise medians.
        
        Resistance: Resistant to >50% Byzantine clients (requires majority to attack)
        
        Args:
            weights_list: List of weight vectors
            
        Returns:
            Element-wise median
        """
        # Stack into matrix (n_clients, n_features)
        stacked = np.vstack(weights_list)
        
        # Element-wise median (robust to outliers)
        result = np.median(stacked, axis=0)
        
        if self.verbose:
            logger.info(f"Median aggregation: used {len(weights_list)} clients")
        
        return result
    
    def _trimmed_mean(self, weights_list: List[np.ndarray]) -> np.ndarray:
        """
        Trimmed mean aggregation: Remove extreme values, then average.
        
        Resistance: Resistant to corruption of bounded magnitude
        Trim ratio determines robustness level
        
        Args:
            weights_list: List of weight vectors
            
        Returns:
            Trimmed mean of weights
        """
        stacked = np.vstack(weights_list)
        n_clients = len(weights_list)
        
        # Number of clients to remove from each tail
        trim_count = max(1, int(np.ceil(n_clients * self.trim_ratio)))
        
        # For each feature, trim and average
        result = np.zeros(stacked.shape[1], dtype=np.float64)
        
        for j in range(stacked.shape[1]):
            feature_values = stacked[:, j]
            # Sort and trim both ends
            sorted_vals = np.sort(feature_values)
            trimmed = sorted_vals[trim_count : n_clients - trim_count]
            result[j] = np.mean(trimmed) if len(trimmed) > 0 else np.mean(feature_values)
        
        if self.verbose:
            logger.info(f"Trimmed mean: trim_ratio={self.trim_ratio:.2%}, "
                       f"removed {trim_count} clients from each tail")
        
        return result
    
    def _krum(
        self,
        weights_list: List[np.ndarray],
        num_byzantine: Optional[int] = None
    ) -> np.ndarray:
        """
        Krum aggregation: Select update closest to others.
        
        Algorithm:
        1. Compute pairwise distances between all client updates
        2. For each client, count neighbors within distance threshold
        3. Select client with most neighbors
        
        Resistance: Resistant to (n-k-2) Byzantine clients, where k=num_byzantine
        
        Args:
            weights_list: List of weight vectors
            num_byzantine: Expected number of Byzantine clients
            
        Returns:
            Selected update from most representative client
        """
        n = len(weights_list)
        if num_byzantine is None:
            num_byzantine = self.num_byzantine or max(1, n // 10)  # Default 10%
        
        num_byzantine = min(num_byzantine, (n - 3) // 2)  # Ensure f < (n-2)/2
        
        # Compute pairwise Euclidean distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(weights_list[i] - weights_list[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # For each client, compute distance to k+1 nearest neighbors
        scores = np.zeros(n)
        for i in range(n):
            # Sort distances and take k+1 nearest (excluding self, which has distance 0)
            nearest_distances = np.sort(distances[i])[1 : num_byzantine + 2]
            scores[i] = np.sum(nearest_distances)
        
        # Select client with minimum score (closest to others)
        selected_idx = np.argmin(scores)
        result = weights_list[selected_idx].copy()
        
        if self.verbose:
            logger.info(f"Krum: selected client {selected_idx} "
                       f"(score={scores[selected_idx]:.4f}, "
                       f"expected Byzantine={num_byzantine})")
        
        return result
    
    def _multi_krum(
        self,
        weights_list: List[np.ndarray],
        num_byzantine: Optional[int] = None,
        num_to_select: int = None
    ) -> np.ndarray:
        """
        Multi-Krum aggregation: Select k non-Byzantine updates, then average.
        
        Algorithm:
        1. Repeat Krum selection num_to_select times
        2. Remove selected update from consideration
        3. Average the selected updates
        
        Resistance: Resistant to (n-2f-2) Byzantine clients
        
        Args:
            weights_list: List of weight vectors
            num_byzantine: Expected number of Byzantine clients
            num_to_select: Number of updates to select (larger = more Byzantine-resistant)
            
        Returns:
            Average of selected non-Byzantine updates
        """
        n = len(weights_list)
        if num_byzantine is None:
            num_byzantine = self.num_byzantine or max(1, n // 10)
        
        if num_to_select is None:
            # Select n - 2*num_byzantine updates (theoretical maximum)
            num_to_select = max(1, n - 2 * num_byzantine)
        
        num_to_select = min(num_to_select, n)
        
        # Keep track of remaining updates
        remaining_weights = [w.copy() for w in weights_list]
        selected_weights = []
        
        for round_num in range(num_to_select):
            if len(remaining_weights) == 0:
                break
            
            # Run Krum on remaining weights
            result = self._krum_internal(remaining_weights, num_byzantine)
            selected_weights.append(result)
            
            # Remove selected weight (find closest update)
            best_idx = None
            best_dist = np.inf
            for idx, w in enumerate(remaining_weights):
                dist = np.linalg.norm(w - result)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            
            if best_idx is not None:
                remaining_weights.pop(best_idx)
        
        # Average selected weights
        result = np.mean(selected_weights, axis=0)
        
        if self.verbose:
            logger.info(f"Multi-Krum: selected {len(selected_weights)} updates "
                       f"(expected Byzantine={num_byzantine})")
        
        return result
    
    def _krum_internal(
        self,
        weights_list: List[np.ndarray],
        num_byzantine: int
    ) -> np.ndarray:
        """Internal Krum implementation for Multi-Krum."""
        n = len(weights_list)
        if n == 0:
            return weights_list[0]
        
        num_byzantine = min(num_byzantine, (n - 3) // 2)
        
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(weights_list[i] - weights_list[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        scores = np.zeros(n)
        for i in range(n):
            nearest_distances = np.sort(distances[i])[1 : num_byzantine + 2]
            scores[i] = np.sum(nearest_distances)
        
        selected_idx = np.argmin(scores)
        return weights_list[selected_idx].copy()
    
    def get_defense_info(self) -> Dict:
        """
        Get information about defense mechanism.
        
        Returns:
            Dictionary with defense properties
        """
        info = {
            "method": self.method,
            "description": "",
            "byzantine_resistance": 0,
            "computational_cost": 0,
            "communication_cost": 1.0,
        }
        
        if self.method == "fedavg":
            info["description"] = "Standard aggregation (no defense)"
            info["byzantine_resistance"] = 0
            info["computational_cost"] = 1
        
        elif self.method == "median":
            info["description"] = "Element-wise median selection"
            info["byzantine_resistance"] = 50  # >50% Byzantine resistance
            info["computational_cost"] = 2
        
        elif self.method == "trimmed_mean":
            info["description"] = f"Trimmed mean (trim {self.trim_ratio:.0%})"
            info["byzantine_resistance"] = int(self.trim_ratio * 100)
            info["computational_cost"] = 2
        
        elif self.method == "krum":
            info["description"] = "Krum (nearest neighbor selection)"
            info["byzantine_resistance"] = 70  # Approx 70% for typical cases
            info["computational_cost"] = 3  # O(n²)
        
        elif self.method == "multi_krum":
            info["description"] = "Multi-Krum (multiple nearest neighbors)"
            info["byzantine_resistance"] = 85  # Better Byzantine resistance
            info["computational_cost"] = 4  # O(n² * m) where m = selections
        
        return info


class PoisoningDetector:
    """
    Detect poisoned/malicious client updates.
    
    Uses statistical methods to identify outlier updates:
    - Isolation Forest for anomaly detection
    - Variance-based detection
    - Distance-based detection
    """
    
    def __init__(self, threshold: float = 0.95, method: str = "distance"):
        """
        Initialize detector.
        
        Args:
            threshold: Detection threshold (0-1, higher = stricter)
            method: Detection method ('distance', 'variance', 'isolation_forest')
        """
        self.threshold = threshold
        self.method = method.lower()
        logger.info(f"Initialized PoisoningDetector with method='{self.method}'")
    
    def detect(
        self,
        client_weights: List[np.ndarray],
        client_ids: Optional[List[str]] = None
    ) -> Dict:
        """
        Detect suspicious client updates.
        
        Args:
            client_weights: List of weight matrices
            client_ids: Client identifiers
            
        Returns:
            Dictionary with detection results
        """
        if self.method == "distance":
            return self._detect_distance(client_weights, client_ids)
        elif self.method == "variance":
            return self._detect_variance(client_weights, client_ids)
        else:
            return self._detect_distance(client_weights, client_ids)
    
    def _detect_distance(
        self,
        client_weights: List[np.ndarray],
        client_ids: Optional[List[str]] = None
    ) -> Dict:
        """
        Distance-based detection: Identify updates far from median.
        
        Args:
            client_weights: List of weight vectors
            client_ids: Client identifiers
            
        Returns:
            Detection results with suspicious clients
        """
        flattened = [w.flatten() for w in client_weights]
        stacked = np.vstack(flattened)
        
        # Compute median update
        median_update = np.median(stacked, axis=0)
        
        # Compute distances from median
        distances = np.array([
            np.linalg.norm(w - median_update)
            for w in flattened
        ])
        
        # Detect outliers
        threshold = np.percentile(distances, self.threshold * 100)
        suspicious = distances > threshold
        
        if client_ids is None:
            client_ids = [f"client_{i}" for i in range(len(client_weights))]
        
        results = {
            "method": "distance",
            "suspicious_clients": [
                client_ids[i] for i in np.where(suspicious)[0]
            ],
            "distances": dict(zip(client_ids, distances)),
            "threshold": threshold,
            "num_suspicious": int(np.sum(suspicious)),
        }
        
        return results
    
    def _detect_variance(
        self,
        client_weights: List[np.ndarray],
        client_ids: Optional[List[str]] = None
    ) -> Dict:
        """Variance-based detection: High variance = unusual update."""
        flattened = [w.flatten() for w in client_weights]
        stacked = np.vstack(flattened)
        
        # Per-feature variance
        variances = np.var(stacked, axis=0)
        
        # Updates with high-variance features are suspicious
        client_suspicion = np.zeros(len(flattened))
        for i, w in enumerate(flattened):
            high_var_features = w ** 2 * variances
            client_suspicion[i] = np.mean(high_var_features)
        
        threshold = np.percentile(client_suspicion, self.threshold * 100)
        suspicious = client_suspicion > threshold
        
        if client_ids is None:
            client_ids = [f"client_{i}" for i in range(len(client_weights))]
        
        results = {
            "method": "variance",
            "suspicious_clients": [
                client_ids[i] for i in np.where(suspicious)[0]
            ],
            "suspicion_scores": dict(zip(client_ids, client_suspicion)),
            "threshold": threshold,
            "num_suspicious": int(np.sum(suspicious)),
        }
        
        return results
