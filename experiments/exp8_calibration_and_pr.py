#!/usr/bin/env python
"""
Experiment 8: Advanced Professional Evaluation (Ablation, ECE, AUPRC, and Reliability Plotting)

Trains FedAvg, FedProx, and FedF2 under clean and poisoned conditions.
Measures AUROC, AUPRC (PR-AUC), ECE (Expected Calibration Error), Brier Score,
Recall, Precision, and F2. Generates reliability diagrams (calibration curves)
and PR curves saved to results/plots/.
"""

import sys
import os
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_dataset_with_df
from src.data.split import distribute_by_care_unit
from src.training.federated import FederatedTrainer
from src.config.config import RANDOM_SEED
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as _LR
from src.evaluation.metrics import (
    calculate_pr_auc, calculate_expected_calibration_error,
    compute_calibration_curve, calculate_roc_auc
)
from sklearn.metrics import recall_score, precision_score, fbeta_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def scores_from_weights(weights, scaler, X):
    """Compute predicted probabilities from raw model weights."""
    X_scaled = scaler.transform(X)
    logits = X_scaled @ weights['coef'] + weights['intercept']
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def evaluate_threshold_metrics(y_true, y_scores, threshold):
    y_pred = (y_scores >= threshold).astype(int)
    return {
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f2': fbeta_score(y_true, y_pred, beta=2, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    seed = args.seed

    logger.info("=" * 70)
    logger.info("EXPERIMENT 8: Advanced Professional Ablation & Calibration")
    logger.info("=" * 70)

    # Configure publication quality plots
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

    # Curated publication color scheme
    color_map = {
        'FedAvg (Raw)': '#718096',             # Muted grey
        'FedAvg (Calibrated)': '#1A365D',      # Deep Navy
        'FedProx (Calibrated)': '#2C7A7B',     # Teal/Sage
        'FedF2 (Calibrated, γ=0.5)': '#D69E2E' # Muted Gold
    }

    # ── 1. Load data ──────────────────────────────────────────────────────
    logger.info("\n[1/5] Loading cohort...")
    df, X, y = load_dataset_with_df(use_cache=True)

    # ── 2. Split ──────────────────────────────────────────────────────────
    logger.info("[2/5] Creating reproducible splits...")
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=seed, stratify=y)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=seed, stratify=y[temp_idx])

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    care_units_train = df.iloc[train_idx]['first_careunit']
    clients = distribute_by_care_unit(
        X_train, y_train, care_units_train, min_patients_per_unit=100)

    # Poisoned set: 1 degenerate care unit
    poisoned_clients = dict(clients)
    first_unit = sorted(poisoned_clients.keys())[0]
    X_poison, _ = poisoned_clients[first_unit]
    y_poison = np.ones(len(X_poison), dtype=int)
    y_poison[0] = 0  # ensure 2 classes
    poisoned_clients[first_unit] = (X_poison, y_poison)

    # Define experimental runs
    runs = [
        # Scenario, Strategy name, clients set, use_dp, platt_calibrate, label, kwargs
        ('clean', 'fedavg', clients, False, False, 'FedAvg (Raw)', {}),
        ('clean', 'fedavg', clients, False, True, 'FedAvg (Calibrated)', {}),
        ('clean', 'fedprox', clients, False, True, 'FedProx (Calibrated)', {'fedprox_mu': 0.01}),
        ('clean', 'fedf2', clients, False, True, 'FedF2 (Calibrated, γ=0.5)', {'fedf2_gamma': 0.5}),
        ('poisoned', 'fedavg', poisoned_clients, False, False, 'Poisoned FedAvg (Raw)', {}),
        ('poisoned', 'fedavg', poisoned_clients, False, True, 'Poisoned FedAvg (Calibrated)', {}),
        ('poisoned', 'fedf2', poisoned_clients, False, True, 'Poisoned FedF2 (Calibrated, γ=0.5)', {'fedf2_gamma': 0.5}),
    ]

    results = []
    plot_data = {}

    # Initialize calibration plot
    fig_cal, ax_cal = plt.subplots(figsize=(6, 5))
    ax_cal.plot([0, 1], [0, 1], color='#4A5568', linestyle='--', linewidth=1.2, label='Perfect Calibration')

    logger.info("\n[3/5] Executing ablation models...")
    for scenario, strategy, cls_set, use_dp, calibrate, label, extra in runs:
        logger.info(f"Training: {label}...")
        trainer = FederatedTrainer(
            clients=cls_set,
            val_data=(X_val, y_val),
            test_data=(X_test, y_test),
            num_rounds=args.rounds,
            learning_rate=0.01,
            use_dp=use_dp,
            aggregation_strategy=strategy,
            random_seed=seed,
            **extra
        )
        res = trainer.train()
        w = res['final_weights']

        # Get scores
        val_proba = scores_from_weights(w, trainer.scaler, X_val)
        test_proba = scores_from_weights(w, trainer.scaler, X_test)

        if calibrate:
            # Fit Platt scaling
            val_scaled = trainer.scaler.transform(X_val)
            test_scaled = trainer.scaler.transform(X_test)
            logits_val = val_scaled @ w['coef'] + w['intercept']
            logits_test = test_scaled @ w['coef'] + w['intercept']

            platt = _LR(max_iter=1000)
            platt.fit(logits_val.reshape(-1, 1), y_val)
            final_val_proba = platt.predict_proba(logits_val.reshape(-1, 1))[:, 1]
            final_test_proba = platt.predict_proba(logits_test.reshape(-1, 1))[:, 1]
            threshold = 0.39  # Standardized calibrated threshold
        else:
            final_val_proba = val_proba
            final_test_proba = test_proba
            threshold = 0.05  # Raw threshold

        # Metrics
        auroc, _, _ = calculate_roc_auc(y_test, final_test_proba)
        auprc, prec_c, rec_c = calculate_pr_auc(y_test, final_test_proba)
        ece = calculate_expected_calibration_error(y_test, final_test_proba)
        thresh_metrics = evaluate_threshold_metrics(y_test, final_test_proba, threshold)

        results.append({
            'scenario': scenario,
            'label': label,
            'auroc': auroc,
            'auprc': auprc,
            'ece': ece,
            'recall': thresh_metrics['recall'],
            'precision': thresh_metrics['precision'],
            'f2': thresh_metrics['f2'],
            'threshold': threshold
        })

        # Save plotting data for clean scenarios to avoid clutter
        if scenario == 'clean':
            # Calibration reliability curve
            mpv, fop = compute_calibration_curve(y_test, final_test_proba, n_bins=10)
            color = color_map.get(label, '#2D3748')
            ax_cal.plot(mpv, fop, marker='o', markersize=4, color=color, linewidth=1.2, label=f"{label} (ECE={ece:.3f})")
            plot_data[label] = (prec_c, rec_c, auprc)

    # Save Reliability Diagram (Calibration Curves)
    ax_cal.set_xlabel('Mean Predicted Probability')
    ax_cal.set_ylabel('Fraction of Positives')
    ax_cal.set_title('Reliability Diagram (Calibration Curves)')
    ax_cal.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')
    
    # Beautify reliability axes
    ax_cal.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.5, alpha=0.7)
    ax_cal.spines['top'].set_visible(False)
    ax_cal.spines['right'].set_visible(False)
    ax_cal.spines['left'].set_linewidth(0.8)
    ax_cal.spines['bottom'].set_linewidth(0.8)
    ax_cal.spines['left'].set_color('#4A5568')
    ax_cal.spines['bottom'].set_color('#4A5568')
    
    plot_dir = Path('results/plots')
    paper_dir = Path('paper/figures')
    plot_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    
    fig_cal.savefig(plot_dir / 'reliability_curves.png', dpi=300, bbox_inches='tight')
    fig_cal.savefig(plot_dir / 'reliability_curves.pdf', bbox_inches='tight')
    fig_cal.savefig(paper_dir / 'reliability_curves.png', dpi=300, bbox_inches='tight')
    fig_cal.savefig(paper_dir / 'reliability_curves.pdf', bbox_inches='tight')
    plt.close(fig_cal)
    logger.info("✓ Reliability curves saved (PNG/PDF).")

    # Generate PR Curves Plot
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    for name, (prec, rec, auprc_val) in plot_data.items():
        color = color_map.get(name, '#2D3748')
        ax_pr.plot(rec, prec, color=color, linewidth=1.2, label=f"{name} (AUPRC={auprc_val:.3f})")
        
    ax_pr.set_xlabel('Recall (Sensitivity)')
    ax_pr.set_ylabel('Precision (PPV)')
    ax_pr.set_title('Precision-Recall Curves')
    ax_pr.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none')
    
    # Beautify PR axes
    ax_pr.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.5, alpha=0.7)
    ax_pr.spines['top'].set_visible(False)
    ax_pr.spines['right'].set_visible(False)
    ax_pr.spines['left'].set_linewidth(0.8)
    ax_pr.spines['bottom'].set_linewidth(0.8)
    ax_pr.spines['left'].set_color('#4A5568')
    ax_pr.spines['bottom'].set_color('#4A5568')
    
    fig_pr.savefig(plot_dir / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
    fig_pr.savefig(plot_dir / 'precision_recall_curves.pdf', bbox_inches='tight')
    fig_pr.savefig(paper_dir / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
    fig_pr.savefig(paper_dir / 'precision_recall_curves.pdf', bbox_inches='tight')
    plt.close(fig_pr)
    logger.info("✓ Precision-Recall curves saved (PNG/PDF).")

    # Save ablation stats to CSV
    results_df = pd.DataFrame(results)
    out_dir = Path('results/summary')
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / 'ablation_and_calibration_metrics.csv'
    results_df.to_csv(summary_file, index=False)
    logger.info(f"✓ Saved ablation summary CSV to {summary_file}")

    # Display beautifully formatted markdown output
    logger.info("\n" + "=" * 80)
    logger.info("FINAL COMPREHENSIVE EXPERIMENTAL PROFESSIONALISM SUMMARY")
    logger.info("=" * 80)
    print(results_df.to_string(index=False))
    logger.info("\n✅ EXPERIMENT 8 RUN COMPLETE")


if __name__ == '__main__':
    main()
