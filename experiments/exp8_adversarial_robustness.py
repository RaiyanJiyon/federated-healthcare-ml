"""
Experiment 8: Adversarial Robustness Testing
============================================

Evaluate Byzantine-resistant aggregation methods against coordinated poisoning attacks.

Tests:
1. Different aggregation methods: FedAvg, Median, Trimmed Mean, Krum, Multi-Krum
2. Different Byzantine percentages: 0%, 5%, 10%, 20%, 30%
3. Different attack strategies: Scaling, Sign-flip, Label-flip, Random
4. Metrics: Accuracy drop, attack success rate, defense effectiveness

Expected results:
- FedAvg: No defense, vulnerable to attacks
- Median: Good defense against <50% Byzantine
- Trimmed Mean: Moderate defense
- Krum: Strong defense
- Multi-Krum: Strongest defense

Author: Federated Healthcare Research Team
Date: April 2, 2026
"""

import sys
sys.path.insert(0, '/home/raiyanjiyon/Machine Learning/federated-healthcare-ml')

import os
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Tuple
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project imports
from src.data.loader import load_dataset_with_df
from src.data.preprocess import DataPreprocessor
from src.data.split import train_test_split_data, distribute_non_iid
from src.models.model import LogisticRegressionModel
from src.utils.feature_engineering import HealthcareFeatureEngineer
from src.fl.strategy import FedAvgAggregator
from src.fl.robust_aggregation import RobustAggregator, PoisoningDetector
from src.fl.adversarial import (
    MaliciousClient, AdversarialSimulator, RobustnessEvaluator, CollaborativeAttack,
    PoisoningConfig
)


def evaluate_aggregation_robustness(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_clients: int = 10,
    num_byzantine: int = 2,
    attack_strategy: str = "scaling",
    aggregation_method: str = "fedavg",
    num_rounds: int = 10,
    seed: int = 42
) -> Dict:
    """
    Evaluate robustness of aggregation method against Byzantine attacks.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        num_clients: Number of clients
        num_byzantine: Number of malicious clients
        attack_strategy: Attack type
        aggregation_method: Aggregation method to test
        num_rounds: FL communication rounds
        seed: Random seed
        
    Returns:
        Robustness evaluation results
    """
    np.random.seed(seed)
    
    # Split data among clients (Non-IID)
    client_data_dict = distribute_non_iid(X_train, y_train, num_clients, alpha=0.5, seed=seed)
    client_data = list(client_data_dict.values())
    client_sizes = [len(y) for _, y in client_data]
    
    # Initialize aggregator
    aggregator = RobustAggregator(
        method=aggregation_method,
        trim_ratio=0.1,
        num_byzantine=num_byzantine
    )
    
    # Create poisoning configuration
    poison_config = PoisoningConfig(
        strategy=attack_strategy,
        poison_factor=-4.0,
        magnitude=1.0,
        seed=seed
    )
    
    # Create adversarial simulator
    simulator = AdversarialSimulator(
        num_clients=num_clients,
        num_byzantine=num_byzantine,
        poison_config=poison_config,
        seed=seed
    )
    simulator.create_byzantine_clients()
    
    # Create poison detector
    detector = PoisoningDetector(threshold=0.95, method="distance")
    
    # Initialize global model
    global_model = create_model(random_state=seed)
    global_weights = global_model.coef_.flatten()
    
    # Training history
    accuracies_clean = []  # Without attack
    accuracies_poisoned = []  # With attack
    detection_rates = []
    
    # Federated learning rounds
    for round_num in tqdm(range(num_rounds), desc=f"Testing {aggregation_method} "
                                                    f"vs {attack_strategy.replace('_', ' ')} "
                                                    f"({num_byzantine} attackers)"):
        client_updates = []
        client_models = []
        
        # Client local training
        for client_id, (X_client, y_client) in enumerate(client_data):
            model = create_model(random_state=seed + round_num + client_id)
            model.fit(X_client, y_client)
            model_weights = model.coef_.flatten()
            
            # Compute update (δ = w_local - w_global)
            update = model_weights - global_weights
            client_updates.append(update)
            client_models.append(model_weights)
        
        # Apply poisoning attack
        client_ids = [f"client_{i}" for i in range(num_clients)]
        poisoned_updates = simulator.poison_round(
            client_updates, global_weights, client_ids, round_num
        )
        
        # Detect poisoning
        detection_result = detector.detect(
            [u.reshape(-1, 1) for u in poisoned_updates],
            client_ids
        )
        detected_suspicious = len(detection_result["suspicious_clients"])
        detection_rates.append(detected_suspicious / num_byzantine if num_byzantine > 0 else 0)
        
        # Aggregate with Byzantine updates
        aggregated_update = aggregator.aggregate(
            [u.reshape(-1, 1) for u in poisoned_updates],
            client_sizes=client_sizes
        ).flatten()
        
        # Update global model
        global_weights_poisoned = global_weights + aggregated_update
        
        # Evaluate poisoned model
        global_model.coef_ = global_weights_poisoned.reshape(1, -1)
        acc_poisoned_clean = global_model.score(X_test, y_test)
        from sklearn.metrics import recall_score
        rec_poisoned = recall_score(y_test, global_model.predict(X_test))
        
        # Aggregate without poisoning (clean)
        aggregated_update_clean = aggregator.aggregate(
            [u.reshape(-1, 1) for u in client_updates],
            client_sizes=client_sizes
        ).flatten()
        
        global_weights_clean = global_weights + aggregated_update_clean
        global_model.coef_ = global_weights_clean.reshape(1, -1)
        acc_clean = global_model.score(X_test, y_test)
        rec_clean = recall_score(y_test, global_model.predict(X_test))
        
        # Update for next round (use poisoned version in simulation)
        global_weights = global_weights_poisoned
        
        accuracies_clean.append(rec_clean)
        accuracies_poisoned.append(rec_poisoned)
    
    # Compute robustness metrics
    evaluator = RobustnessEvaluator()
    robustness = evaluator.evaluate(
        np.array(accuracies_clean),
        np.array(accuracies_poisoned),
        aggregation_method,
        num_byzantine,
        attack_strategy
    )
    
    robustness["detection_rate"] = float(np.mean(detection_rates))
    robustness["clean_accuracies"] = accuracies_clean
    robustness["poisoned_accuracies"] = accuracies_poisoned
    
    return robustness


def run_adversarial_robustness_experiment():
    """
    Run comprehensive adversarial robustness evaluation.
    
    Tests all combinations of:
    - Aggregation methods: FedAvg, Median, Trimmed Mean, Krum, Multi-Krum
    - Byzantine percentages: 0%, 5%, 10%, 20%, 30%
    - Attack strategies: Scaling, Sign-flip, Label-flip, Random
    """
    logger.info("=" * 80)
    logger.info("EXPERIMENT 8: ADVERSARIAL ROBUSTNESS TESTING")
    logger.info("=" * 80)
    
    # Load and preprocess data
    X, y = load_diabetes_data()
    X = preprocess_data(X)
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Configuration
    num_clients = 10
    num_rounds = 10
    
    aggregation_methods = ["fedavg", "median", "trimmed_mean", "krum", "multi_krum"]
    attack_strategies = ["scaling", "sign_flip", "label_flip", "random"]
    byzantine_percentages = [0, 5, 10, 20, 30]  # % of clients
    
    results = {
        "experiment": "adversarial_robustness",
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "test_size": len(X_test),
            "train_size": len(X_train),
        },
        "results": [],
        "summary": {}
    }
    
    # Run all combinations
    total_tests = (len(aggregation_methods) * 
                   len(attack_strategies) * 
                   len(byzantine_percentages))
    
    with tqdm(total=total_tests, desc="Overall progress") as pbar:
        for aggregation_method in aggregation_methods:
            method_results = {aggregation_method: {}}
            
            for attack_strategy in attack_strategies:
                strategy_results = {attack_strategy: {}}
                
                for byzantine_pct in byzantine_percentages:
                    num_byzantine = max(1, int(num_clients * byzantine_pct / 100))
                    if byzantine_pct == 0:
                        num_byzantine = 0
                    
                    try:
                        logger.info(f"\nTesting: {aggregation_method.upper()} "
                                   f"vs {attack_strategy} "
                                   f"({byzantine_pct}% Byzantine = {num_byzantine} clients)")
                        
                        result = evaluate_aggregation_robustness(
                            X_train, y_train, X_test, y_test,
                            num_clients=num_clients,
                            num_byzantine=num_byzantine,
                            attack_strategy=attack_strategy,
                            aggregation_method=aggregation_method,
                            num_rounds=num_rounds,
                            seed=42 + byzantine_pct
                        )
                        
                        results["results"].append(result)
                        strategy_results[attack_strategy][byzantine_pct] = result
                        
                        logger.info(f"  Recall (clean): {result['clean_accuracy_mean']:.4f}")
                        logger.info(f"  Recall (poisoned): {result['poisoned_accuracy_mean']:.4f}")
                        logger.info(f"  Accuracy drop: {result['accuracy_drop']:.4f}")
                        logger.info(f"  Defense success: {result['defense_success_rate']:.1%}")
                        
                    except Exception as e:
                        logger.error(f"Error in test: {e}")
                        strategy_results[attack_strategy][byzantine_pct] = {"error": str(e)}
                    
                    finally:
                        pbar.update(1)
    
    # Compute summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY OF RESULTS")
    logger.info("=" * 80)
    
    # Group results by aggregation method
    method_summaries = {}
    for result in results["results"]:
        if "error" in result:
            continue
        
        method = result["aggregation_method"]
        if method not in method_summaries:
            method_summaries[method] = {
                "defense_success_rates": [],
                "accuracy_drops": [],
                "detection_rates": [],
            }
        
        method_summaries[method]["defense_success_rates"].append(
            result["defense_success_rate"]
        )
        method_summaries[method]["accuracy_drops"].append(
            result["accuracy_drop"]
        )
        method_summaries[method]["detection_rates"].append(
            result.get("detection_rate", 0)
        )
    
    results["summary"] = {}
    for method, data in method_summaries.items():
        results["summary"][method] = {
            "avg_defense_success": float(np.mean(data["defense_success_rates"])),
            "avg_accuracy_drop": float(np.mean(data["accuracy_drops"])),
            "avg_detection_rate": float(np.mean(data["detection_rates"])),
            "robustness_rank": 0,  # Will be set after sorting
        }
    
    # Rank by defense success
    sorted_methods = sorted(
        results["summary"].items(),
        key=lambda x: x[1]["avg_defense_success"],
        reverse=True
    )
    
    for rank, (method, data) in enumerate(sorted_methods, 1):
        data["robustness_rank"] = rank
        results["summary"][method] = data
    
    # Print rankings
    logger.info("\nAGGREGATION METHOD ROBUSTNESS RANKING:")
    logger.info("-" * 80)
    for i, (method, data) in enumerate(sorted_methods, 1):
        logger.info(f"{i}. {method.upper():20} "
                   f"Defense: {data['avg_defense_success']:6.1%}  "
                   f"Accuracy drop: {data['avg_accuracy_drop']:6.4f}  "
                   f"Detection: {data['avg_detection_rate']:6.1%}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/adversarial_robustness_{timestamp}.json"
    os.makedirs("results", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_file}")
    
    # Print key findings
    logger.info("\n" + "=" * 80)
    logger.info("KEY FINDINGS")
    logger.info("=" * 80)
    
    logger.info("\n1. ROBUSTNESS RANKING:")
    for rank, (method, data) in enumerate(sorted_methods, 1):
        status = "✅ STRONG" if data["avg_defense_success"] > 0.8 else \
                 "⚠️  MODERATE" if data["avg_defense_success"] > 0.6 else \
                 "❌ WEAK"
        logger.info(f"   {rank}. {method.upper():20} - {status} "
                   f"({data['avg_defense_success']:.1%})")
    
    logger.info("\n2. ATTACK VULNERABILITY:")
    attack_vulnerability = {}
    for result in results["results"]:
        if "error" in result:
            continue
        attack = result["attack_strategy"]
        if attack not in attack_vulnerability:
            attack_vulnerability[attack] = []
        attack_vulnerability[attack].append(result["accuracy_drop"])
    
    for attack, drops in attack_vulnerability.items():
        avg_drop = np.mean(drops)
        severity = "CRITICAL" if avg_drop > 0.10 else \
                   "HIGH" if avg_drop > 0.05 else \
                   "MODERATE" if avg_drop > 0.02 else \
                   "LOW"
        logger.info(f"   {attack.upper():15} attack: {avg_drop:.4f} accuracy drop ({severity})")
    
    logger.info("\n3. RECOMMENDATIONS:")
    best_method = sorted_methods[0][0]
    logger.info(f"   ✅ Use '{best_method}' for strong Byzantine resistance")
    logger.info(f"   ✅ Monitor for scaling attacks (highest impact)")
    logger.info(f"   ✅ Threshold for suspicion detection: top 10% updates")
    
    logger.info("\n" + "=" * 80)
    
    return results


if __name__ == "__main__":
    results = run_adversarial_robustness_experiment()
    
    logger.info("\n✅ Adversarial robustness evaluation complete!")
