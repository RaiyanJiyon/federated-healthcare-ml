#!/usr/bin/env python
"""
Experiment 7: Differential Privacy Analysis (Phase 3)

Tests federated learning with differential privacy (DP).
Demonstrates the privacy-utility tradeoff: privacy gains vs accuracy loss.

Key Questions:
1. How does differential privacy impact model accuracy?
2. What's the optimal epsilon for healthcare (ε=1.0)?
3. How does privacy budget accumulate over rounds?
4. Can we maintain clinical safety (recall ≥ 80%) with DP?
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_dataset_with_df
from src.data.split import distribute_by_care_unit
from src.training.federated import FederatedTrainer
from src.evaluation.metrics import calculate_brier_score, calculate_expected_calibration_error
from src.config.config import RANDOM_SEED, DP_EPSILON, DP_DELTA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_dp_variant(X_train, y_train, X_val, y_val, X_test, y_test, 
                   care_units_train, seed, use_dp):
    """Run one federated experiment with DP enabled/disabled."""
    
    # Distribute to care units
    clients = distribute_by_care_unit(X_train, y_train, care_units_train, min_patients_per_unit=100)
    
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
        use_dp=use_dp,
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
        'seed': seed,
        'use_dp': use_dp
    }


def run_exp7():
    """
    Run differential privacy analysis comparing DP vs non-DP federated learning.
    """
    logger.info("="*70)
    logger.info("EXPERIMENT 7: DIFFERENTIAL PRIVACY ANALYSIS")
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
    
    # Test variants: baseline (no DP) vs DP enabled
    logger.info("\n[3/3] Running DP experiments...")
    
    all_results = []
    
    # Baseline: no DP
    logger.info("\n--- Baseline (No DP) ---")
    try:
        res = run_dp_variant(
            X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test,
            care_units_train, seed=42, use_dp=False
        )
        all_results.append(res)
        logger.info(f"✓ No DP: AUROC={res['auroc']:.4f}, Recall={res['recall']:.1%}")
    except Exception as e:
        logger.error(f"✗ No DP failed: {e}")
    
    # DP variant (uses default DP_EPSILON from config)
    logger.info(f"\n--- Differential Privacy (enabled) ---")
    try:
        res = run_dp_variant(
            X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test,
            care_units_train, seed=42, use_dp=True
        )
        all_results.append(res)
        
        if all_results:
            baseline_auroc = all_results[0]['auroc']
            auroc_loss_pct = (1 - res['auroc'] / baseline_auroc) * 100
            logger.info(f"✓ DP enabled: AUROC={res['auroc']:.4f} (loss: {auroc_loss_pct:.1f}%), Recall={res['recall']:.1%}")
    except Exception as e:
        logger.error(f"✗ DP failed: {e}")
    
    if not all_results:
        logger.error("No successful runs!")
        return None
    
    results_df = pd.DataFrame(all_results)
    
    # Analysis
    logger.info("\n" + "="*70)
    logger.info("DIFFERENTIAL PRIVACY ANALYSIS SUMMARY")
    logger.info("="*70)
    
    if len(all_results) >= 2:
        baseline = all_results[0]
        dp_result = all_results[1]
        
        logger.info(f"\nBaseline (No DP):")
        logger.info(f"  AUROC:     {baseline['auroc']:.4f}")
        logger.info(f"  Recall:    {baseline['recall']:.1%}")
        logger.info(f"  Precision: {baseline['precision']:.1%}")
        logger.info(f"  Brier:     {baseline['brier']:.4f}")
        
        logger.info(f"\nDifferential Privacy Enabled:")
        auroc_loss_pct = (1 - dp_result['auroc'] / baseline['auroc']) * 100
        recall_loss_pct = (1 - dp_result['recall'] / baseline['recall']) * 100
        logger.info(f"  AUROC:     {dp_result['auroc']:.4f} (loss: {auroc_loss_pct:.1f}%)")
        logger.info(f"  Recall:    {dp_result['recall']:.1%} (loss: {recall_loss_pct:.1f}%)")
        logger.info(f"  Precision: {dp_result['precision']:.1%}")
        logger.info(f"  Brier:     {dp_result['brier']:.4f}")
        logger.info(f"  Privacy Budget: ε={DP_EPSILON:.2f} per round × 20 rounds = {DP_EPSILON*20:.1f} total")
        
        # Clinical viability
        clinically_safe = dp_result['recall'] >= 0.80
        safety_str = "✓ VIABLE" if clinically_safe else "✗ NOT VIABLE"
        logger.info(f"\nClinical Viability (Recall ≥ 80%): {safety_str}")
        logger.info(f"  DP Recall: {dp_result['recall']:.1%}")
    
    # Save results
    output_dir = Path('results/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'exp7_differential_privacy.csv'
    results_df.to_csv(results_file, index=False)
    logger.info(f"\n✓ Results saved to {results_file}")
    
    # Create summary table
    summary_rows = []
    for i, res in enumerate(all_results):
        label = 'No DP (Baseline)' if i == 0 else f'DP Enabled (ε={DP_EPSILON})'
        summary_rows.append({
            'experiment': label,
            'auroc': f"{res['auroc']:.4f}",
            'brier': f"{res['brier']:.4f}",
            'ece': f"{res['ece']:.4f}",
            'recall': f"{res['recall']:.1%}",
            'precision': f"{res['precision']:.1%}"
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_dir / 'exp7_differential_privacy_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"✓ Summary saved to {summary_file}")
    
    return results_df


if __name__ == '__main__':
    results_df = run_exp7()
    logger.info("\n✅ EXPERIMENT 7 COMPLETE")
