#!/usr/bin/env python
"""
Experiment 8: Adversarial Robustness (Phase 3)

Tests federated learning robustness against Byzantine attacks.
Compares standard FedAvg against Byzantine client attacks.

Key Questions:
1. How robust is federated learning to malicious clients?
2. What happens if some clients flip labels or return corrupted weights?
3. Can robust aggregation (e.g., Krum) mitigate Byzantine attacks?
4. What fraction of Byzantine clients can be tolerated?

Attack Scenarios:
- Label Flipping: Malicious clients flip binary labels (0 → 1, 1 → 0)
- Impact: Gradients point in opposite direction
- Tested with 1/7 and 2/7 malicious clients (14% and 29%)
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_dataset_with_df
from src.data.split import distribute_by_care_unit
from src.training.federated import FederatedTrainer
from src.evaluation.metrics import calculate_brier_score, calculate_expected_calibration_error
from src.config.config import RANDOM_SEED
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KrumAggregator:
    """
    Krum aggregation: selects the weight vector closest to all others.
    More robust to Byzantine attacks than simple averaging.
    """
    
    @staticmethod
    def aggregate(weight_vectors: list, sample_sizes: list = None) -> Dict:
        """
        Krum aggregation: select the model closest to all neighbors.
        
        For each model, compute sum of distances to all other models.
        Select the model with minimum sum of distances.
        
        Args:
            weight_vectors: List of weight dicts from clients
            sample_sizes: Optional sample counts (unused in Krum)
        
        Returns:
            Selected weight dict (most representative)
        """
        if len(weight_vectors) <= 1:
            return weight_vectors[0]
        
        # Convert weights to vectors for distance computation
        weight_arrays = []
        for w in weight_vectors:
            coef = w['coef'].flatten()
            intercept = np.array([w['intercept']]) if np.isscalar(w['intercept']) else w['intercept']
            weight_arrays.append(np.concatenate([coef, intercept]))
        
        # Compute pairwise distances
        n = len(weight_arrays)
        distances = np.zeros(n)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(weight_arrays[i] - weight_arrays[j]) ** 2
                    distances[i] += dist
        
        # Select model with minimum distance sum (closest to others)
        selected_idx = np.argmin(distances)
        return weight_vectors[selected_idx]


def run_robustness_variant(X_train, y_train, X_val, y_val, X_test, y_test, 
                            care_units_train, malicious_fraction=0.0, seed=42):
    """
    Run federated experiment with optional Byzantine clients.
    
    Args:
        malicious_fraction: Fraction of clients to make malicious (0.0 = clean)
    """
    
    # Distribute to care units
    clients = distribute_by_care_unit(X_train, y_train, care_units_train, min_patients_per_unit=100)
    client_names = sorted(clients.keys())
    n_clients = len(client_names)
    n_malicious = max(1, int(np.ceil(n_clients * malicious_fraction)))
    
    # Mark malicious clients (first n_malicious)
    malicious_clients = set(client_names[:n_malicious]) if malicious_fraction > 0 else set()
    
    if malicious_clients:
        logger.info(f"  Malicious clients ({malicious_fraction:.0%}): {malicious_clients}")
        
        # Apply label flipping to malicious clients
        for client_name in malicious_clients:
            X_c, y_c = clients[client_name]
            y_c_flipped = 1 - y_c  # Flip binary labels
            clients[client_name] = (X_c, y_c_flipped)
    
    # Prepare validation and test data
    val_data = (X_val, y_val)
    test_data = (X_test, y_test)
    
    # Initialize trainer
    trainer = FederatedTrainer(
        clients=clients,
        val_data=val_data,
        test_data=test_data,
        num_rounds=20,
        learning_rate=0.01,
        use_dp=False,
        aggregation_strategy='fedavg',
        fedprox_mu=0.01,
        random_seed=seed
    )
    
    # Train and get final weights
    train_result = trainer.train()
    final_weights = train_result['final_weights']
    test_auroc = train_result['test_auroc']
    
    # Get predictions for additional metrics
    X_test_scaled = trainer.scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.coef_ = final_weights['coef'].reshape(1, -1)
    model.intercept_ = np.array([final_weights['intercept']]) if isinstance(final_weights['intercept'], (int, float)) else final_weights['intercept']
    model.classes_ = final_weights['classes']
    
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Evaluate
    brier = calculate_brier_score(y_test, y_pred_proba)
    ece = calculate_expected_calibration_error(y_test, y_pred_proba)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    return {
        'auroc': test_auroc,
        'brier': brier,
        'ece': ece,
        'recall': recall,
        'precision': precision,
        'malicious_fraction': malicious_fraction,
        'n_malicious': n_malicious,
        'n_total_clients': n_clients,
        'seed': seed
    }


def run_exp8():
    """
    Run adversarial robustness analysis with Byzantine client scenarios.
    """
    logger.info("="*70)
    logger.info("EXPERIMENT 8: ADVERSARIAL ROBUSTNESS (Byzantine Resilience)")
    logger.info("="*70)
    
    # Load data
    logger.info("\n[1/3] Loading MIMIC-IV cohort...")
    df, X, y = load_dataset_with_df(use_cache=True)
    logger.info(f"Cohort loaded: {X.shape[0]} patients, {X.shape[1]} features")
    
    # Split
    logger.info("\n[2/3] Splitting data...")
    n_train = int(0.70 * len(X))
    n_val = int(0.15 * len(X))
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    
    # Get care units
    if 'first_careunit' in df.columns:
        care_units_train = df.iloc[:n_train]['first_careunit']
    else:
        care_units_train = pd.Series(['Unit_' + str(i % 7) for i in range(n_train)])
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Test scenarios
    logger.info("\n[3/3] Running Byzantine robustness experiments...")
    
    all_results = []
    
    # Scenario 1: Clean (no Byzantine clients)
    logger.info("\n--- Scenario 1: Clean Federation (No Byzantine Clients) ---")
    try:
        res = run_robustness_variant(
            X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test,
            care_units_train, malicious_fraction=0.0, seed=42
        )
        all_results.append(res)
        logger.info(f"✓ Clean: AUROC={res['auroc']:.4f}, Recall={res['recall']:.1%}")
    except Exception as e:
        logger.error(f"✗ Clean failed: {e}")
    
    # Scenario 2: 1/7 Byzantine clients (~14%)
    logger.info("\n--- Scenario 2: Byzantine Attack (1/7 clients, ~14%) ---")
    try:
        res = run_robustness_variant(
            X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test,
            care_units_train, malicious_fraction=1/7, seed=42
        )
        all_results.append(res)
        logger.info(f"✓ 1/7 Byzantine: AUROC={res['auroc']:.4f}, Recall={res['recall']:.1%}")
    except Exception as e:
        logger.error(f"✗ 1/7 Byzantine failed: {e}")
    
    # Scenario 3: 2/7 Byzantine clients (~29%)
    logger.info("\n--- Scenario 3: Byzantine Attack (2/7 clients, ~29%) ---")
    try:
        res = run_robustness_variant(
            X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test,
            care_units_train, malicious_fraction=2/7, seed=42
        )
        all_results.append(res)
        logger.info(f"✓ 2/7 Byzantine: AUROC={res['auroc']:.4f}, Recall={res['recall']:.1%}")
    except Exception as e:
        logger.error(f"✗ 2/7 Byzantine failed: {e}")
    
    if not all_results:
        logger.error("No successful runs!")
        return None
    
    results_df = pd.DataFrame(all_results)
    
    # Analysis
    logger.info("\n" + "="*70)
    logger.info("ADVERSARIAL ROBUSTNESS ANALYSIS SUMMARY")
    logger.info("="*70)
    
    if len(all_results) > 0:
        clean_result = all_results[0]
        baseline_auroc = clean_result['auroc']
        baseline_recall = clean_result['recall']
        
        logger.info(f"\nClean Baseline:")
        logger.info(f"  AUROC:     {baseline_auroc:.4f}")
        logger.info(f"  Recall:    {baseline_recall:.1%}")
        logger.info(f"  Precision: {clean_result['precision']:.1%}")
        
        if len(all_results) > 1:
            logger.info(f"\nByzantine Resilience:")
            for i, res in enumerate(all_results[1:], 1):
                auroc_loss_pct = (1 - res['auroc'] / baseline_auroc) * 100
                recall_loss_pct = (1 - res['recall'] / baseline_recall) * 100
                logger.info(f"\n  Scenario {i}: {res['n_malicious']}/{res['n_total_clients']} Byzantine clients")
                logger.info(f"    AUROC:  {res['auroc']:.4f} (loss: {auroc_loss_pct:.1f}%)")
                logger.info(f"    Recall: {res['recall']:.1%} (loss: {recall_loss_pct:.1f}%)")
                
                # Resilience assessment
                if auroc_loss_pct < 5:
                    logger.info(f"    Status: ✓ RESILIENT (loss < 5%)")
                elif auroc_loss_pct < 10:
                    logger.info(f"    Status: ⚠ DEGRADED (loss 5-10%)")
                else:
                    logger.info(f"    Status: ✗ VULNERABLE (loss > 10%)")
    
    # Save results
    output_dir = Path('results/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'exp8_adversarial_robustness.csv'
    results_df.to_csv(results_file, index=False)
    logger.info(f"\n✓ Results saved to {results_file}")
    
    # Create summary table
    summary_rows = []
    for res in all_results:
        scenario = f"Clean" if res['malicious_fraction'] == 0 else f"{res['n_malicious']}/{res['n_total_clients']} Byzantine"
        summary_rows.append({
            'scenario': scenario,
            'n_malicious': res['n_malicious'],
            'auroc': f"{res['auroc']:.4f}",
            'recall': f"{res['recall']:.1%}",
            'precision': f"{res['precision']:.1%}",
            'brier': f"{res['brier']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_dir / 'exp8_adversarial_robustness_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"✓ Summary saved to {summary_file}")
    
    # Recommendation
    logger.info("\n" + "="*70)
    logger.info("RECOMMENDATIONS")
    logger.info("="*70)
    if len(all_results) > 1:
        worst_loss = max([(1 - r['auroc'] / baseline_auroc) * 100 for r in all_results[1:]])
        if worst_loss < 5:
            logger.info("\n✓ FedAvg is RESILIENT to Byzantine attacks in this setting")
            logger.info("  Recommendation: Current approach acceptable for federated deployment")
        else:
            logger.info("\n⚠ FedAvg shows vulnerability to Byzantine attacks")
            logger.info("  Recommendation: Consider robust aggregation (Krum, Median) for production")
    
    return results_df


if __name__ == '__main__':
    results_df = run_exp8()
    logger.info("\n✅ EXPERIMENT 8 COMPLETE")
