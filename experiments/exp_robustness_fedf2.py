#!/usr/bin/env python
"""
Comprehensive Robustness Evaluation of FedF2 under Byzantine Attacks

Tests FedF2 against multiple attack scenarios:
- 1, 2, 3 malicious clients
- Label-flip attacks
- Sign-flip attacks  
- Adaptive attacks (label-flip + scaling)

Compares aggregation methods:
- FedAvg (baseline)
- FedProx (proximal regularization)
- Median (coordinate-wise robust)
- Krum (distance-based selection)
- FedF2 (F2-score weighted)

Results saved to: results/plots/exp_robustness_fedf2_comprehensive.csv
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_dataset_with_df
from src.data.split import distribute_by_care_unit
from src.training.federated import FederatedTrainer
from src.fl.adversarial import PoisoningConfig, AdversarialSimulator
from src.config.config import RANDOM_SEED
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, fbeta_score
)
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_ece(y_true, y_probs, n_bins=10):
    """Compute Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    ece = 0.0
    bin_sums = np.zeros(n_bins)
    bin_true = np.zeros(n_bins)
    bin_total = np.zeros(n_bins)
    
    for i in range(len(y_true)):
        bin_idx = min(int(y_probs[i] * n_bins), n_bins - 1)
        bin_sums[bin_idx] += y_probs[i]
        bin_true[bin_idx] += y_true[i]
        bin_total[bin_idx] += 1
    
    bin_accs = np.divide(bin_true, bin_total, where=bin_total > 0, out=np.zeros(n_bins))
    bin_confs = np.divide(bin_sums, bin_total, where=bin_total > 0, out=np.zeros(n_bins))
    
    for i in range(n_bins):
        if bin_total[i] > 0:
            ece += (bin_total[i] / len(y_true)) * np.abs(bin_accs[i] - bin_confs[i])
    
    return ece


def evaluate_at_threshold(y_true, y_scores, threshold):
    """Evaluate classification metrics at a fixed decision threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    return {
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f2': fbeta_score(y_true, y_pred, beta=2, zero_division=0),
    }


def select_recall_threshold(y_true, y_scores, target_recall=0.85):
    """Pick the threshold closest to a target recall."""
    best = None
    best_threshold = 0.5
    for threshold in [i / 100 for i in range(1, 100)]:
        metrics = evaluate_at_threshold(y_true, y_scores, threshold)
        candidate = (
            abs(metrics['recall'] - target_recall),
            -metrics['precision'],
            -metrics['f2'],
            threshold,
            metrics['recall'],
            metrics['precision'],
            metrics['f2'],
        )
        if best is None or candidate < best:
            best = candidate
            best_threshold = threshold
    return best_threshold, {
        'recall': best[4], 'precision': best[5], 'f2': best[6]
    }


def scores_from_weights(weights, scaler, X):
    """Compute predicted probabilities from raw model weights."""
    X_scaled = scaler.transform(X)
    logits = X_scaled @ weights['coef'] + weights['intercept']
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def create_poisoned_clients(
    clean_clients: Dict,
    num_malicious: int,
    attack_type: str,
    seed: int = 42
) -> Tuple[Dict, List[str]]:
    """
    Create poisoned clients by corrupting local data/labels.
    
    Args:
        clean_clients: Original client dictionary
        num_malicious: Number of clients to poison
        attack_type: 'label_flip', 'sign_flip', or 'adaptive'
        seed: Random seed
        
    Returns:
        (poisoned_clients_dict, list_of_poisoned_client_names)
    """
    poisoned_clients = dict(clean_clients)
    client_names = sorted(poisoned_clients.keys())
    
    # Select which clients to poison
    np.random.seed(seed)
    poisoned_indices = np.random.choice(
        len(client_names), min(num_malicious, len(client_names)), replace=False
    )
    poisoned_names = [client_names[i] for i in sorted(poisoned_indices)]
    
    for client_name in poisoned_names:
        X_client, y_client = poisoned_clients[client_name]
        y_poisoned = y_client.copy()
        
        if attack_type == 'label_flip':
            # Flip all labels: 0→1, 1→0
            y_poisoned = 1 - y_poisoned
            
        elif attack_type == 'sign_flip':
            # Sign-flip: swap positive/negative class by oversampling positive
            # Make ~99% of samples positive
            n_flip = max(1, int(0.99 * len(y_poisoned)))
            flip_indices = np.random.choice(len(y_poisoned), n_flip, replace=False)
            y_poisoned[flip_indices] = 1
            
        elif attack_type == 'adaptive':
            # Combined: mostly label-flip + some extreme oversampling
            y_poisoned = 1 - y_poisoned  # Flip all labels
            n_flip = max(1, int(0.5 * len(y_poisoned)))
            flip_indices = np.random.choice(len(y_poisoned), n_flip, replace=False)
            y_poisoned[flip_indices] = 1  # Push to all positive
        
        # Ensure at least 2 classes to avoid sklearn errors
        if len(np.unique(y_poisoned)) < 2:
            y_poisoned[0] = 1 - y_poisoned[0]
        
        poisoned_clients[client_name] = (X_client, y_poisoned)
    
    return poisoned_clients, poisoned_names


def run_strategy(
    clients,
    X_val, y_val, X_test, y_test,
    strategy, seed, num_rounds=5, **kwargs
):
    """Run federated training and evaluate performance."""
    try:
        trainer = FederatedTrainer(
            clients=clients,
            val_data=(X_val, y_val),
            test_data=(X_test, y_test),
            num_rounds=num_rounds,
            learning_rate=0.01,
            use_dp=False,
            aggregation_strategy=strategy,
            random_seed=seed,
            **kwargs
        )
        result = trainer.train()
        w = result['final_weights']
        
        # Evaluate
        val_scores = scores_from_weights(w, trainer.scaler, X_val)
        test_scores = scores_from_weights(w, trainer.scaler, X_test)
        
        try:
            test_auroc = roc_auc_score(y_test, test_scores)
        except:
            test_auroc = 0.5
        
        threshold, _ = select_recall_threshold(y_val, val_scores)
        test_metrics = evaluate_at_threshold(y_test, test_scores, threshold)
        
        ece = compute_ece(y_test, test_scores)
        
        return {
            'test_auroc': test_auroc,
            'threshold': threshold,
            'test_recall': test_metrics['recall'],
            'test_precision': test_metrics['precision'],
            'test_f2': test_metrics['f2'],
            'test_ece': ece,
            'success': True
        }
    except Exception as e:
        logger.warning(f"Strategy {strategy} failed: {e}")
        return {
            'test_auroc': np.nan,
            'threshold': 0.5,
            'test_recall': np.nan,
            'test_precision': np.nan,
            'test_f2': np.nan,
            'test_ece': np.nan,
            'success': False
        }


def run_robustness_exp():
    """
    Comprehensive robustness evaluation of FedF2 under Byzantine attacks.
    """
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE ROBUSTNESS EVALUATION: FedF2 vs BYZANTINE ATTACKS")
    logger.info("=" * 80)
    
    seed = int(os.getenv('RANDOM_SEED', str(RANDOM_SEED)))
    
    # ── 1. Load and split data ────────────────────────────────────────
    logger.info("\n[1/4] Loading MIMIC-IV data...")
    df, X, y = load_dataset_with_df(use_cache=True)
    logger.info(f"  Cohort: {X.shape[0]} patients, {X.shape[1]} features, "
                f"mortality {y.mean():.1%}")
    
    logger.info("\n[2/4] Train/val/test split (70/15/15)...")
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=seed, stratify=y)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=seed, stratify=y[temp_idx])
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    care_units_train = df.iloc[train_idx]['first_careunit']
    
    logger.info("\n[3/4] Creating federated clients...")
    clean_clients = distribute_by_care_unit(
        X_train, y_train, care_units_train, min_patients_per_unit=100)
    logger.info(f"  Created {len(clean_clients)} clients")
    
    n_clients = len(clean_clients)
    
    # Define aggregation strategies
    strategies = [
        ('FedAvg', 'fedavg', {}),
        ('FedProx (μ=0.01)', 'fedprox', {'fedprox_mu': 0.01}),
        ('Median', 'median', {}),
        ('Krum', 'krum', {}),
        ('FedF2 (γ=0.5)', 'fedf2', {'fedf2_gamma': 0.5}),
    ]
    
    # Define attack scenarios
    attack_scenarios = [
        ('clean', 0, None),
        ('label-flip-1', 1, 'label_flip'),
        ('label-flip-2', 2, 'label_flip'),
        ('label-flip-3', 3, 'label_flip'),
        ('sign-flip-1', 1, 'sign_flip'),
        ('sign-flip-2', 2, 'sign_flip'),
        ('adaptive-1', 1, 'adaptive'),
        ('adaptive-2', 2, 'adaptive'),
    ]
    
    # ── 4. Run all experiments ────────────────────────────────────────
    logger.info("\n[4/4] Running comprehensive robustness evaluation...")
    
    rows = []
    total_configs = len(attack_scenarios) * len(strategies)
    current = 0
    
    for scenario_name, num_malicious, attack_type in attack_scenarios:
        logger.info(f"\n{'='*80}")
        logger.info(f"SCENARIO: {scenario_name} ({num_malicious} malicious clients)")
        logger.info(f"{'='*80}")
        
        # Create poisoned clients for this scenario
        if scenario_name == 'clean':
            clients_to_use = clean_clients
            poisoned_names = []
        else:
            clients_to_use, poisoned_names = create_poisoned_clients(
                clean_clients, num_malicious, attack_type, seed=seed
            )
            logger.info(f"  Poisoned clients: {poisoned_names}")
        
        for strategy_label, strategy_name, extra_kwargs in strategies:
            current += 1
            logger.info(f"\n  [{current}/{total_configs}] {strategy_label:20s} ...")
            
            m = run_strategy(
                clients_to_use, X_val, y_val, X_test, y_test,
                strategy_name, seed, num_rounds=5, **extra_kwargs
            )
            
            m['scenario'] = scenario_name
            m['num_malicious'] = num_malicious
            m['attack_type'] = attack_type if attack_type else 'none'
            m['strategy'] = strategy_label
            
            rows.append(m)
            
            if m['success']:
                logger.info(f"    ✓ AUROC={m['test_auroc']:.4f}, F2={m['test_f2']:.4f}, ECE={m['test_ece']:.4f}")
            else:
                logger.info(f"    ✗ FAILED")
    
    # ── Save results ──────────────────────────────────────────────────
    results_df = pd.DataFrame(rows)
    col_order = [
        'scenario', 'num_malicious', 'attack_type', 'strategy',
        'test_auroc', 'test_recall', 'test_precision', 'test_f2', 'test_ece',
        'threshold', 'success'
    ]
    results_df = results_df[col_order]
    
    output_dir = Path('results/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'exp_robustness_fedf2_comprehensive.csv'
    results_df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Results saved to {output_file}")
    
    # ── Generate summary tables ───────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("ROBUSTNESS SUMMARY: AUROC BY SCENARIO AND STRATEGY")
    logger.info("=" * 80)
    
    pivot_auroc = results_df.pivot_table(
        values='test_auroc',
        index='scenario',
        columns='strategy',
        aggfunc='mean'
    )
    logger.info(f"\n{pivot_auroc.to_string()}")
    
    logger.info("\n" + "=" * 80)
    logger.info("EXPECTED CALIBRATION ERROR (ECE) BY SCENARIO AND STRATEGY")
    logger.info("=" * 80)
    
    pivot_ece = results_df.pivot_table(
        values='test_ece',
        index='scenario',
        columns='strategy',
        aggfunc='mean'
    )
    logger.info(f"\n{pivot_ece.to_string()}")
    
    logger.info("\n" + "=" * 80)
    logger.info("F2-SCORE (CLINICAL UTILITY) BY SCENARIO AND STRATEGY")
    logger.info("=" * 80)
    
    pivot_f2 = results_df.pivot_table(
        values='test_f2',
        index='scenario',
        columns='strategy',
        aggfunc='mean'
    )
    logger.info(f"\n{pivot_f2.to_string()}")
    
    # Identify best performer per attack scenario
    logger.info("\n" + "=" * 80)
    logger.info("BEST PERFORMER PER ATTACK SCENARIO (by AUROC)")
    logger.info("=" * 80)
    
    for scenario_name in results_df['scenario'].unique():
        scenario_df = results_df[results_df['scenario'] == scenario_name]
        best_idx = scenario_df['test_auroc'].idxmax()
        best = scenario_df.loc[best_idx]
        logger.info(f"{scenario_name:20s}: {best['strategy']:20s} "
                   f"(AUROC={best['test_auroc']:.4f}, F2={best['test_f2']:.4f})")
    
    return results_df


if __name__ == '__main__':
    results = run_robustness_exp()
    logger.info("\n" + "=" * 80)
    logger.info("✅ ROBUSTNESS EVALUATION COMPLETE")
    logger.info("=" * 80)
