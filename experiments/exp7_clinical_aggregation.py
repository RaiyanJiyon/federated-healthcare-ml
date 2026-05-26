#!/usr/bin/env python
"""
Experiment 7: Clinical-Sensitivity-Aware Federated Aggregation (FedF2)

Compares FedF2 against FedAvg and FedProx on care-unit non-IID MIMIC-IV data.
Tests under both clean conditions and with a simulated degenerate client
(always-positive predictor) to validate robustness against trivial classifiers.

FedF2 blends sample-size weighting with local validation F2-scores:
    alpha_k = (1 - gamma) * (n_k / n) + gamma * (F2_k / sum(F2))

Results are saved to results/plots/exp7_clinical_aggregation.csv
"""

import sys
import os
import argparse
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, fbeta_score
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def run_strategy(clients, X_val, y_val, X_test, y_test,
                 strategy, seed, num_rounds=5, **kwargs):
    """Run a federated training loop and return evaluation metrics."""
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
    test_auroc = roc_auc_score(y_test, test_scores)

    threshold, _ = select_recall_threshold(y_val, val_scores)
    test_metrics = evaluate_at_threshold(y_test, test_scores, threshold)

    return {
        'test_auroc': test_auroc,
        'threshold': threshold,
        'test_recall': test_metrics['recall'],
        'test_precision': test_metrics['precision'],
        'test_f2': test_metrics['f2'],
    }


def run_exp7():
    """Compare FedF2 against FedAvg and FedProx."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 7: Clinical-Sensitivity-Aware Aggregation (FedF2)")
    logger.info("=" * 70)

    # CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--rounds', type=int, default=5)
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else int(
        os.getenv('RANDOM_SEED', str(RANDOM_SEED)))

    # ── 1. Load data ──────────────────────────────────────────────────────
    logger.info("\n[1/5] Loading MIMIC-IV cohort...")
    df, X, y = load_dataset_with_df(use_cache=True)
    logger.info(f"Cohort: {X.shape[0]} patients, {X.shape[1]} features, "
                f"mortality {y.mean():.1%}")

    # ── 2. Split ──────────────────────────────────────────────────────────
    logger.info("\n[2/5] Train / val / test split (70/15/15)...")
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=seed, stratify=y)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=seed, stratify=y[temp_idx])

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    care_units_train = df.iloc[train_idx]['first_careunit']

    # ── 3. Create federated clients ───────────────────────────────────────
    logger.info("\n[3/5] Distributing to care-unit clients...")
    clients = distribute_by_care_unit(
        X_train, y_train, care_units_train, min_patients_per_unit=100)
    logger.info(f"Created {len(clients)} clients")
    for name in sorted(clients):
        Xc, yc = clients[name]
        logger.info(f"  {name}: {len(Xc)} samples, "
                     f"mortality {yc.mean():.1%}")

    # ── 4. Run clean comparison ───────────────────────────────────────────
    logger.info("\n[4/5] Running aggregation comparison (CLEAN)...")

    configs = [
        ('FedAvg', 'fedavg', {}),
        ('FedProx (μ=0.01)', 'fedprox', {'fedprox_mu': 0.01}),
        ('FedF2 (γ=0.1)', 'fedf2', {'fedf2_gamma': 0.1}),
        ('FedF2 (γ=0.3)', 'fedf2', {'fedf2_gamma': 0.3}),
        ('FedF2 (γ=0.5)', 'fedf2', {'fedf2_gamma': 0.5}),
    ]

    rows = []
    for label, strategy, extra in configs:
        logger.info(f"\n{'─'*60}")
        logger.info(f"Strategy: {label}")
        logger.info(f"{'─'*60}")
        m = run_strategy(clients, X_val, y_val, X_test, y_test,
                         strategy, seed, num_rounds=args.rounds, **extra)
        m['strategy'] = label
        m['scenario'] = 'clean'
        rows.append(m)
        logger.info(f"  AUROC={m['test_auroc']:.4f}  "
                     f"Recall={m['test_recall']:.2%}  "
                     f"Precision={m['test_precision']:.2%}  "
                     f"F2={m['test_f2']:.4f}")

    # ── 5. Poisoned scenario ──────────────────────────────────────────────
    logger.info("\n[5/5] Running POISONED scenario (1 degenerate client)...")

    # Create a poisoned client set: replace the first client with a
    # degenerate client whose labels are ALL positive (simulates a model
    # that always predicts death → 100% recall, ~11% precision).
    poisoned_clients = dict(clients)  # shallow copy
    first_unit = sorted(poisoned_clients.keys())[0]
    X_poison, _ = poisoned_clients[first_unit]
    y_poison = np.ones(len(X_poison), dtype=int)  # all-positive labels
    y_poison[0] = 0  # ensure at least two classes to avoid sklearn crash
    poisoned_clients[first_unit] = (X_poison, y_poison)
    logger.info(f"  Poisoned client: {first_unit} "
                f"({len(X_poison)} samples → 99.9% positive labels)")

    for label, strategy, extra in configs:
        logger.info(f"\n{'─'*60}")
        logger.info(f"Strategy (poisoned): {label}")
        logger.info(f"{'─'*60}")
        m = run_strategy(poisoned_clients, X_val, y_val, X_test, y_test,
                         strategy, seed, num_rounds=args.rounds, **extra)
        m['strategy'] = label
        m['scenario'] = 'poisoned'
        rows.append(m)
        logger.info(f"  AUROC={m['test_auroc']:.4f}  "
                     f"Recall={m['test_recall']:.2%}  "
                     f"Precision={m['test_precision']:.2%}  "
                     f"F2={m['test_f2']:.4f}")

    # ── Save results ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(rows)
    col_order = ['scenario', 'strategy', 'test_auroc', 'test_recall',
                 'test_precision', 'test_f2', 'threshold']
    results_df = results_df[col_order]

    output_dir = Path('results/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'exp7_clinical_aggregation.csv'
    results_df.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to {output_file}")

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS — CLINICAL AGGREGATION COMPARISON")
    logger.info("=" * 70)
    logger.info(f"\n{results_df.to_string(index=False)}")

    # Highlight key finding
    clean = results_df[results_df['scenario'] == 'clean']
    poisoned = results_df[results_df['scenario'] == 'poisoned']

    best_clean = clean.loc[clean['test_f2'].idxmax()]
    best_poisoned = poisoned.loc[poisoned['test_f2'].idxmax()]

    logger.info(f"\n✓ Best clean:    {best_clean['strategy']} "
                f"(F2={best_clean['test_f2']:.4f})")
    logger.info(f"✓ Best poisoned: {best_poisoned['strategy']} "
                f"(F2={best_poisoned['test_f2']:.4f})")

    return results_df


if __name__ == '__main__':
    results = run_exp7()
    logger.info("\n✅ EXPERIMENT 7 COMPLETE")
