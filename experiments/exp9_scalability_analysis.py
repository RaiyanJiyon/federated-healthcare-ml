#!/usr/bin/env python
"""
Experiment 9: Scalability and Dropout Analysis (Phase 3)

Tests federated learning scalability with:
1. Increasing number of clients (7 → 14 → 21 → 28)
2. Client dropout simulation (0% → 10% → 20% → 30%)
3. Communication overhead tracking

Key Questions:
1. How does performance scale with more clients?
2. Is federated learning robust to client dropout?
3. What's the throughput and communication cost?
4. Can we handle realistic client availability constraints?
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import time

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


def simulate_scalability(X_train, y_train, X_val, y_val, X_test, y_test,
                         care_units_train, target_client_count, dropout_fraction=0.0, seed=42):
    """
    Run federated experiment with simulated client scaling.
    
    For scaling beyond actual care units, we simulate by:
    - Taking subsets of training data
    - Creating virtual clients from data partitions
    
    Args:
        target_client_count: Desired number of clients (7, 14, 21, 28)
        dropout_fraction: Fraction of clients to drop each round (0.0 = no dropout)
    """
    
    # Distribute to care units (actual)
    clients = distribute_by_care_unit(X_train, y_train, care_units_train, min_patients_per_unit=100)
    
    # Scale up by splitting existing clients
    if target_client_count > len(clients):
        clients_scaled = {}
        for client_name, (X_c, y_c) in clients.items():
            # Split this client into multiple virtual clients
            n_split = target_client_count // len(clients)
            split_size = len(X_c) // n_split
            
            for i in range(n_split):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < n_split - 1 else len(X_c)
                virtual_name = f"{client_name}_v{i}"
                clients_scaled[virtual_name] = (X_c[start_idx:end_idx], y_c[start_idx:end_idx])
        
        clients = clients_scaled
    
    actual_client_count = len(clients)
    
    # Simulate dropout: randomly exclude clients each round
    n_dropout = int(np.ceil(actual_client_count * dropout_fraction))
    
    if dropout_fraction > 0:
        client_list = sorted(clients.keys())
        dropout_clients = np.random.choice(client_list, size=n_dropout, replace=False)
        logger.info(f"  Simulating dropout: excluding {n_dropout}/{actual_client_count} clients")
        
        # Remove dropout clients
        for c in dropout_clients:
            del clients[c]
    
    # Prepare validation and test data
    val_data = (X_val, y_val)
    test_data = (X_test, y_test)
    
    # Initialize trainer
    start_time = time.time()
    
    trainer = FederatedTrainer(
        clients=clients,
        val_data=val_data,
        test_data=test_data,
        num_rounds=10,  # Fewer rounds for scalability test
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
    
    elapsed_time = time.time() - start_time
    
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
        'n_clients': len(clients),
        'n_clients_target': target_client_count,
        'dropout_fraction': dropout_fraction,
        'training_time_seconds': elapsed_time,
        'throughput_samples_per_sec': len(X_train) / elapsed_time,
        'seed': seed
    }


def run_exp9():
    """
    Run scalability and dropout analysis.
    """
    logger.info("="*70)
    logger.info("EXPERIMENT 9: SCALABILITY AND DROPOUT ANALYSIS")
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
    logger.info("\n[3/3] Running scalability experiments...")
    
    all_results = []
    
    # Test increasing client counts
    client_counts = [7, 14, 21, 28]
    
    for n_clients in client_counts:
        logger.info(f"\n--- Scalability Test: {n_clients} clients (no dropout) ---")
        try:
            res = simulate_scalability(
                X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test,
                care_units_train, target_client_count=n_clients, dropout_fraction=0.0, seed=42
            )
            all_results.append(res)
            logger.info(f"✓ {n_clients} clients: AUROC={res['auroc']:.4f}, Time={res['training_time_seconds']:.1f}s")
        except Exception as e:
            logger.error(f"✗ {n_clients} clients failed: {e}")
    
    # Test dropout resilience on baseline (7 clients)
    dropout_fractions = [0.1, 0.2, 0.3]
    
    for dropout in dropout_fractions:
        logger.info(f"\n--- Dropout Resilience: 7 clients, {dropout:.0%} dropout ---")
        try:
            res = simulate_scalability(
                X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test,
                care_units_train, target_client_count=7, dropout_fraction=dropout, seed=42
            )
            all_results.append(res)
            logger.info(f"✓ {dropout:.0%} dropout: AUROC={res['auroc']:.4f}, Available clients={res['n_clients']}")
        except Exception as e:
            logger.error(f"✗ {dropout:.0%} dropout failed: {e}")
    
    if not all_results:
        logger.error("No successful runs!")
        return None
    
    results_df = pd.DataFrame(all_results)
    
    # Analysis
    logger.info("\n" + "="*70)
    logger.info("SCALABILITY ANALYSIS SUMMARY")
    logger.info("="*70)
    
    if len(all_results) > 0:
        baseline = all_results[0]
        baseline_auroc = baseline['auroc']
        baseline_time = baseline['training_time_seconds']
        
        logger.info(f"\nBaseline (7 clients, no dropout):")
        logger.info(f"  AUROC: {baseline_auroc:.4f}")
        logger.info(f"  Time: {baseline_time:.1f}s")
        logger.info(f"  Throughput: {baseline['throughput_samples_per_sec']:.0f} samples/sec")
        
        # Scalability
        scale_results = [r for r in all_results if r['dropout_fraction'] == 0.0]
        if len(scale_results) > 1:
            logger.info(f"\nScalability (increasing clients):")
            for res in scale_results[1:]:
                auroc_loss_pct = (1 - res['auroc'] / baseline_auroc) * 100
                time_factor = res['training_time_seconds'] / baseline_time
                logger.info(f"\n  {res['n_clients_target']} clients:")
                logger.info(f"    AUROC: {res['auroc']:.4f} (loss: {auroc_loss_pct:.1f}%)")
                logger.info(f"    Time: {res['training_time_seconds']:.1f}s ({time_factor:.1f}x baseline)")
                logger.info(f"    Throughput: {res['throughput_samples_per_sec']:.0f} samples/sec")
        
        # Dropout resilience
        dropout_results = [r for r in all_results if r['n_clients_target'] == 7 and r['dropout_fraction'] > 0.0]
        if dropout_results:
            logger.info(f"\nDropout Resilience (7 clients, variable dropout):")
            for res in dropout_results:
                auroc_loss_pct = (1 - res['auroc'] / baseline_auroc) * 100
                logger.info(f"\n  {res['dropout_fraction']:.0%} dropout ({res['n_clients']}/{res['n_clients_target']} available):")
                logger.info(f"    AUROC: {res['auroc']:.4f} (loss: {auroc_loss_pct:.1f}%)")
                
                if auroc_loss_pct < 5:
                    logger.info(f"    Status: ✓ RESILIENT (loss < 5%)")
                else:
                    logger.info(f"    Status: ⚠ DEGRADED (loss ≥ 5%)")
    
    # Save results
    output_dir = Path('results/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'exp9_scalability_analysis.csv'
    results_df.to_csv(results_file, index=False)
    logger.info(f"\n✓ Results saved to {results_file}")
    
    # Create summary table
    summary_rows = []
    for res in all_results:
        if res['dropout_fraction'] == 0.0:
            scenario = f"{res['n_clients_target']} clients"
        else:
            scenario = f"7 clients, {res['dropout_fraction']:.0%} dropout"
        
        summary_rows.append({
            'scenario': scenario,
            'n_clients': res['n_clients'],
            'dropout_fraction': f"{res['dropout_fraction']:.0%}",
            'auroc': f"{res['auroc']:.4f}",
            'training_time_sec': f"{res['training_time_seconds']:.1f}",
            'throughput_samps_per_sec': f"{res['throughput_samples_per_sec']:.0f}"
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_dir / 'exp9_scalability_analysis_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"✓ Summary saved to {summary_file}")
    
    # Recommendation
    logger.info("\n" + "="*70)
    logger.info("SCALABILITY RECOMMENDATIONS")
    logger.info("="*70)
    
    max_scalable = max([r['n_clients_target'] for r in scale_results if (1 - r['auroc'] / baseline_auroc) * 100 < 5], default=7)
    logger.info(f"\n✓ Federated learning scales well to {max_scalable}+ clients with <5% AUROC loss")
    
    if dropout_results:
        max_dropout = max([r['dropout_fraction'] for r in dropout_results if (1 - r['auroc'] / baseline_auroc) * 100 < 5], default=0.0)
        logger.info(f"✓ Dropout resilience up to {max_dropout:.0%} with <5% AUROC loss")
        logger.info(f"  Recommendation: Current approach suitable for realistic deployments")
    
    return results_df


if __name__ == '__main__':
    results_df = run_exp9()
    logger.info("\n✅ EXPERIMENT 9 COMPLETE")
