"""
Adversarial Robustness Simulation for Federated Learning

This module simulates Byzantine attacks and evaluates defense mechanisms:
- Malicious client simulation (corrupts local updates)
- Model poisoning attacks (gradient-based and label-flipping)
- Attack detection mechanisms
- Robustness metrics

Attacks simulated:
1. Constant attack: Send fixed malicious update
2. Scaling attack: Multiply update by poison_factor
3. Sign-flip attack: Reverse gradient directions
4. Random attack: Send random values
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Callable
from copy import deepcopy
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PoisoningConfig:
    """Configuration for poisoning attacks."""
    strategy: str = "scaling"  # scaling, constant, sign_flip, label_flip, random
    poison_factor: float = -4.0  # Scale factor (negative reverses)
    magnitude: float = 1.0  # Attack magnitude
    seed: int = 42


class MaliciousClient:
    """
    Simulates a malicious client that sends poisoned updates.
    
    Attributes:
        client_id (str): Unique identifier
        poison_config (PoisoningConfig): Attack configuration
    """
    
    def __init__(
        self,
        client_id: str,
        poison_config: PoisoningConfig,
        local_model: Optional[np.ndarray] = None
    ):
        """
        Initialize malicious client.
        
        Args:
            client_id: Client identifier
            poison_config: Poisoning configuration
            local_model: Local model weights (for gradient computation)
        """
        self.client_id = client_id
        self.poison_config = poison_config
        self.local_model = local_model
        self.attack_history = []
        
        logger.debug(f"Initialized malicious client {client_id} "
                    f"(strategy={poison_config.strategy})")
    
    def poison_update(
        self,
        update: np.ndarray,
        global_model: np.ndarray,
        round_num: int = 0
    ) -> np.ndarray:
        """
        Apply poisoning attack to local update.
        
        Args:
            update: Local model update (local_model - global_model)
            global_model: Current global model
            round_num: Current FL round
            
        Returns:
            Poisoned update
        """
        config = self.poison_config
        poisoned = update.copy()
        
        if config.strategy == "scaling":
            poisoned = poisoned * config.poison_factor
        
        elif config.strategy == "sign_flip":
            poisoned = -poisoned * config.magnitude
        
        elif config.strategy == "constant":
            poisoned = np.full_like(update, config.magnitude)
        
        elif config.strategy == "random":
            np.random.seed(config.seed + round_num)
            poisoned = np.random.randn(*update.shape) * config.magnitude
        
        elif config.strategy == "label_flip":
            # Flip labels: invert loss gradients
            poisoned = -update * config.poison_factor
        
        self.attack_history.append({
            "round": round_num,
            "original_norm": np.linalg.norm(update),
            "poisoned_norm": np.linalg.norm(poisoned),
            "strategy": config.strategy,
        })
        
        return poisoned
    
    def get_attack_metrics(self) -> Dict:
        """Get attack effectiveness metrics."""
        if not self.attack_history:
            return {}
        
        history = np.array([
            [h["original_norm"], h["poisoned_norm"]]
            for h in self.attack_history
        ])
        
        return {
            "client_id": self.client_id,
            "num_attacks": len(self.attack_history),
            "avg_amplification": np.mean(history[:, 1] / (history[:, 0] + 1e-8)),
            "max_amplification": np.max(history[:, 1] / (history[:, 0] + 1e-8)),
        }


class AdversarialSimulator:
    """
    Simulate Byzantine attacks in federated learning.
    
    Supports multiple attack strategies:
    - Gradient scaling attacks
    - Sign-flip attacks
    - Label-flipping attacks
    - Random poisoning
    """
    
    def __init__(
        self,
        num_clients: int = 10,
        num_byzantine: int = 2,
        poison_config: Optional[PoisoningConfig] = None,
        seed: int = 42
    ):
        """
        Initialize adversarial simulator.
        
        Args:
            num_clients: Total number of clients
            num_byzantine: Number of malicious clients
            poison_config: Poisoning configuration
            seed: Random seed
        """
        self.num_clients = num_clients
        self.num_byzantine = min(num_byzantine, num_clients - 1)
        self.poison_config = poison_config or PoisoningConfig()
        self.seed = seed
        np.random.seed(seed)
        
        # Select which clients are malicious
        self.byzantine_indices = np.random.choice(
            num_clients, self.num_byzantine, replace=False
        )
        self.byzantine_clients = {}
        
        logger.info(f"Initialized AdversarialSimulator with "
                   f"{num_clients} clients, {self.num_byzantine} malicious "
                   f"(strategy={self.poison_config.strategy})")
    
    def create_byzantine_clients(
        self,
        client_ids: Optional[List[str]] = None
    ) -> Dict[str, MaliciousClient]:
        """
        Create malicious client instances.
        
        Args:
            client_ids: List of client identifiers
            
        Returns:
            Dictionary mapping client_id -> MaliciousClient
        """
        if client_ids is None:
            client_ids = [f"client_{i}" for i in range(self.num_clients)]
        
        self.byzantine_clients = {}
        for idx in self.byzantine_indices:
            client_id = client_ids[idx]
            self.byzantine_clients[client_id] = MaliciousClient(
                client_id, self.poison_config
            )
        
        return self.byzantine_clients
    
    def poison_round(
        self,
        client_updates: List[np.ndarray],
        global_model: np.ndarray,
        client_ids: Optional[List[str]] = None,
        round_num: int = 0
    ) -> List[np.ndarray]:
        """
        Apply poisoning to updates from Byzantine clients.
        
        Args:
            client_updates: List of client updates
            global_model: Current global model
            client_ids: Client identifiers
            round_num: Current FL round
            
        Returns:
            Modified updates (with poisoning applied to Byzantine clients)
        """
        if client_ids is None:
            client_ids = [f"client_{i}" for i in range(len(client_updates))]
        
        poisoned_updates = client_updates.copy()
        
        for client_id, malicious_client in self.byzantine_clients.items():
            idx = client_ids.index(client_id)
            poisoned_updates[idx] = malicious_client.poison_update(
                client_updates[idx], global_model, round_num
            )
        
        return poisoned_updates
    
    def get_byzantine_info(self) -> Dict:
        """Get information about Byzantine clients."""
        return {
            "Byzantine indices": list(self.byzantine_indices),
            "Byzantine count": self.num_byzantine,
            "Total clients": self.num_clients,
            "Byzantine percentage": f"{100 * self.num_byzantine / self.num_clients:.1f}%",
            "Attack strategy": self.poison_config.strategy,
            "Poison factor": self.poison_config.poison_factor,
        }


class RobustnessEvaluator:
    """
    Evaluate robustness of aggregation methods against Byzantine attacks.
    
    Metrics:
    - Attack success rate: Percentage of rounds where attack affects accuracy
    - Defense success rate: Percentage of rounds where attack is mitigated
    - Accuracy degradation: Drop in accuracy under attack
    - Detection rate: Percentage of attackers detected
    """
    
    def __init__(self):
        """Initialize robustness evaluator."""
        self.results = []
    
    def evaluate(
        self,
        clean_accuracies: np.ndarray,
        poisoned_accuracies: np.ndarray,
        aggregation_method: str,
        num_byzantine: int,
        attack_strategy: str
    ) -> Dict:
        """
        Evaluate robustness against attack.
        
        Args:
            clean_accuracies: Accuracies without attack
            poisoned_accuracies: Accuracies under attack
            aggregation_method: Name of aggregation method
            num_byzantine: Number of Byzantine clients
            attack_strategy: Attack strategy name
            
        Returns:
            Robustness assessment
        """
        # Calculate metrics
        accuracy_drop = np.mean(clean_accuracies - poisoned_accuracies)
        
        # Attack success: rounds where accuracy drops significantly
        significant_drop = accuracy_drop > 0.05  # 5% threshold
        attack_success_rate = np.mean(poisoned_accuracies < 0.80)  # Below safety threshold
        
        # Defense effectiveness: aggregate still accurate
        defense_success_rate = np.mean(poisoned_accuracies > 0.80)
        
        result = {
            "aggregation_method": aggregation_method,
            "attack_strategy": attack_strategy,
            "num_byzantine": num_byzantine,
            "clean_accuracy_mean": float(np.mean(clean_accuracies)),
            "clean_accuracy_std": float(np.std(clean_accuracies)),
            "poisoned_accuracy_mean": float(np.mean(poisoned_accuracies)),
            "poisoned_accuracy_std": float(np.std(poisoned_accuracies)),
            "accuracy_drop": float(accuracy_drop),
            "attack_success_rate": float(attack_success_rate),
            "defense_success_rate": float(defense_success_rate),
            "robust": defense_success_rate > 0.7,  # Robust if >70% success
        }
        
        self.results.append(result)
        return result
    
    def get_defense_ranking(self) -> List[Dict]:
        """
        Rank aggregation methods by robustness.
        
        Returns:
            Sorted list of aggregation methods by defense success
        """
        if not self.results:
            return []
        
        # Group by aggregation method
        method_scores = {}
        for result in self.results:
            method = result["aggregation_method"]
            if method not in method_scores:
                method_scores[method] = []
            method_scores[method].append(result["defense_success_rate"])
        
        # Average scores and rank
        ranking = []
        for method, scores in method_scores.items():
            ranking.append({
                "method": method,
                "avg_defense_success": np.mean(scores),
                "num_tests": len(scores),
            })
        
        return sorted(ranking, key=lambda x: x["avg_defense_success"], reverse=True)


class CollaborativeAttack:
    """
    Coordinated attacks by multiple Byzantine clients.
    
    Enables empirical evaluation of Byzantine-resistant aggregation.
    
    Attack types:
    1. Sybil attack: Many controlled clients send same malicious update
    2. Distributed attack: Coordinated updates with combined effect
    3. Evasion attack: Subtle attacks to avoid detection
    """
    
    def __init__(self, byzantine_clients: Dict[str, MaliciousClient]):
        """
        Initialize collaborative attack.
        
        Args:
            byzantine_clients: Dictionary of malicious clients
        """
        self.byzantine_clients = byzantine_clients
        self.attack_log = []
    
    def coordinate_scaling_attack(
        self,
        target_direction: np.ndarray,
        poison_factor: float = -4.0
    ) -> Dict[str, np.ndarray]:
        """
        Coordinate gradient scaling attack.
        
        All Byzantine clients send scaled versions of target direction.
        
        Args:
            target_direction: Gradient direction to amplify
            poison_factor: Scaling factor
            
        Returns:
            Dictionary mapping client_id -> poisoned_update
        """
        attacks = {}
        for client_id, client in self.byzantine_clients.items():
            # All send scaled target direction
            attacks[client_id] = target_direction * poison_factor
        
        self.attack_log.append({
            "type": "scaling",
            "target_norm": np.linalg.norm(target_direction),
            "poison_factor": poison_factor,
            "num_attackers": len(attacks),
        })
        
        return attacks
    
    def coordinate_flip_attack(
        self,
        target_direction: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Coordinate gradient flip attack.
        
        All Byzantine clients flip signs to reverse learning.
        
        Args:
            target_direction: Gradient direction to flip
            
        Returns:
            Dictionary mapping client_id -> poisoned_update
        """
        attacks = {}
        for client_id, client in self.byzantine_clients.items():
            attacks[client_id] = -target_direction
        
        self.attack_log.append({
            "type": "flip",
            "target_norm": np.linalg.norm(target_direction),
            "num_attackers": len(attacks),
        })
        
        return attacks
    
    def coordinate_mimic_attack(
        self,
        benign_updates: List[np.ndarray],
        deviation_factor: float = 2.0
    ) -> Dict[str, np.ndarray]:
        """
        Coordinate mimic attack: Follow benign updates then deviate.
        
        Byzantine clients start as benign (avoid detection), then attack.
        
        Args:
            benign_updates: Recent benign updates to mimic
            deviation_factor: How much to deviate at end
            
        Returns:
            Dictionary mapping client_id -> poisoned_update
        """
        if not benign_updates:
            raise ValueError("Need benign updates to mimic")
        
        # Average benign direction
        avg_benign = np.mean(benign_updates, axis=0)
        
        attacks = {}
        for client_id, client in self.byzantine_clients.items():
            # Mimic then amplify
            attacks[client_id] = avg_benign * deviation_factor
        
        self.attack_log.append({
            "type": "mimic",
            "deviation_factor": deviation_factor,
            "num_attackers": len(attacks),
        })
        
        return attacks
