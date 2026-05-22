#!/usr/bin/env python
"""
Experiment 4: Federated Aggregation Comparison (FedAvg vs FedProx)

Compares FedAvg and FedProx performance on care-unit non-IID MIMIC-IV data.

FedProx adds a proximal term to regularize local updates toward the global model,
which should improve convergence under data heterogeneity across different ICU units.

Results are saved to results/plots/exp4_aggregation_comparison.csv
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_dataset_with_df
from src.data.split import distribute_by_care_unit
from src.training.federated import FederatedTrainer
from src.config.config import RANDOM_SEED

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_exp4():
    """
    Compare FedAvg and FedProx aggregation strategies.
    """
    logger.info("="*70)
    logger.info("EXPERIMENT 4: Federated Aggregation Comparison (FedAvg vs FedProx)")
    logger.info("="*70)
    
    # Load data
    logger.info("\n[1/4] Loading MIMIC-IV cohort...")
    df, X, y = load_dataset_with_df(use_cache=True)
    logger.info(f"Cohort loaded: {X.shape[0]} patients, {X.shape[1]} features")
    logger.info(f"Mortality rate: {y.mean():.1%}")
    
    # Split into train/val/test
    logger.info("\n[2/4] Splitting data (70/15/15)...")
    n_train = int(0.70 * len(X))
    n_val = int(0.15 * len(X))
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    
    logger.info(f"Train: {X_train.shape[0]} samples ({y_train.mean():.1%} mortality)")
    logger.info(f"Validation: {X_val.shape[0]} samples ({y_val.mean():.1%} mortality)")
    logger.info(f"Test: {X_test.shape[0]} samples ({y_test.mean():.1%} mortality)")
    
    # Extract care units for federated distribution
    logger.info("\n[3/4] Distributing training data to care units...")
    if 'first_careunit' in df.columns:
        care_units_train = df.iloc[:n_train]['first_careunit']
    else:
        logger.warning("'first_careunit' column not found. Using dummy care units.")
        care_units_train = pd.Series(['Unit_' + str(i % 7) for i in range(n_train)])
    
    clients = distribute_by_care_unit(X_train, y_train, care_units_train, min_patients_per_unit=100)
    logger.info(f"Created {len(clients)} federated clients")
    
    for unit_name in sorted(clients.keys()):
        X_c, y_c = clients[unit_name]
        logger.info(f"  {unit_name}: {len(X_c)} samples ({y_c.mean():.1%} mortality)")
    
    # Run FedAvg (baseline)
    logger.info("\n" + "="*70)
    logger.info("FedAvg (Baseline Aggregation)")
    logger.info("="*70)
    
    trainer_fedavg = FederatedTrainer(
        clients=clients,
        val_data=(X_val, y_val),
        test_data=(X_test, y_test),
        num_rounds=20,
        learning_rate=0.01,
        use_dp=False,
        aggregation_strategy='fedavg',
        random_seed=RANDOM_SEED
    )
    
    result_fedavg = trainer_fedavg.train()
    fedavg_auroc = result_fedavg['test_auroc']
    logger.info(f"FedAvg Final Test AUROC: {fedavg_auroc:.4f}")
    
    # Run FedProx with different μ values
    fedprox_mu_values = [0.001, 0.01, 0.1]
    fedprox_results = {}
    
    for mu in fedprox_mu_values:
        logger.info("\n" + "="*70)
        logger.info(f"FedProx with μ={mu}")
        logger.info("="*70)
        
        trainer_fedprox = FederatedTrainer(
            clients=clients,
            val_data=(X_val, y_val),
            test_data=(X_test, y_test),
            num_rounds=20,
            learning_rate=0.01,
            use_dp=False,
            aggregation_strategy='fedprox',
            fedprox_mu=mu,
            random_seed=RANDOM_SEED
        )
        
        result_fedprox = trainer_fedprox.train()
        fedprox_results[mu] = result_fedprox['test_auroc']
        logger.info(f"FedProx (μ={mu}) Final Test AUROC: {result_fedprox['test_auroc']:.4f}")
    
    # Compile results
    logger.info("\n[4/4] Compiling results...")
    
    results_data = {
        'aggregation': ['FedAvg'] + [f'FedProx (μ={mu})' for mu in fedprox_mu_values],
        'test_auroc': [fedavg_auroc] + list(fedprox_results.values())
    }
    
    results_df = pd.DataFrame(results_data)
    
    logger.info("\n" + "="*70)
    logger.info("FINAL RESULTS - AGGREGATION COMPARISON")
    logger.info("="*70)
    logger.info(f"\n{results_df.to_string(index=False)}")
    
    # Save results
    output_dir = Path('results/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'exp4_aggregation_comparison.csv'
    results_df.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to {output_file}")
    
    # Analysis
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS")
    logger.info("="*70)
    
    best_idx = results_df['test_auroc'].idxmax()
    best_method = results_df.loc[best_idx, 'aggregation']
    best_auroc = results_df.loc[best_idx, 'test_auroc']
    
    logger.info(f"✓ Best aggregation: {best_method} (AUROC: {best_auroc:.4f})")
    
    # Improvement over FedAvg
    improvements = {}
    for mu in fedprox_mu_values:
        fedprox_auroc = fedprox_results[mu]
        improvement = fedprox_auroc - fedavg_auroc
        improvements[mu] = improvement
        direction = "↑" if improvement > 0 else "↓"
        logger.info(f"  FedProx (μ={mu}): {fedprox_auroc:.4f} {direction} {improvement:+.4f} vs FedAvg")
    
    # Statistical insight
    logger.info("\nKey Findings:")
    max_improvement = max(improvements.values())
    if max_improvement > 0.002:
        best_mu = max(improvements.items(), key=lambda x: x[1])[0]
        logger.info(f"  • FedProx with μ={best_mu} shows improvement (+{improvements[best_mu]:.4f})")
        logger.info(f"  • Proximal regularization helps with care-unit heterogeneity")
    else:
        logger.info(f"  • All methods converge to similar performance (variance < 0.002)")
        logger.info(f"  • Care-unit partitioning may have limited heterogeneity effect")
    
    return results_df


if __name__ == '__main__':
    results = run_exp4()
    logger.info("\n✅ EXPERIMENT 4 COMPLETE")
