"""Experiment 1: Federated Learning Baseline

Trains federated learning model on 7 ICU care units.
Each unit acts as an independent federated client.
Compares federated performance against centralized baseline.

This is the primary Phase 1 validation: does federated LR achieve
≥0.85 AUROC when trained on 7 care-unit clients?

Requirements:
- Phase 0 cohort validation completed (baseline AUROC 0.8887)
- Cached cohort at data/cache/mimic_iv_cohort.csv

Metrics:
- Federated Test AUROC vs Centralized Test AUROC
- Train-test divergence
- Per-client performance statistics
"""

import sys
import os
import argparse
import logging
from pathlib import Path

import numpy as np

# Add parent directory to path to import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_dataset_with_df
from src.data.split import distribute_by_care_unit
from sklearn.metrics import roc_auc_score, recall_score, precision_score, fbeta_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_centralized_baseline_simple(X_train, y_train, X_val, y_val, X_test, y_test, seed=42):
    """Simple centralized LR baseline (matches federated feature preprocessing)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=seed, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    val_proba = model.predict_proba(X_val_scaled)[:, 1]
    test_proba = model.predict_proba(X_test_scaled)[:, 1]
    threshold, threshold_metrics = select_recall_calibrated_threshold(y_val, val_proba)

    val_metrics = evaluate_threshold_metrics(y_val, val_proba, threshold)
    test_metrics = evaluate_threshold_metrics(y_test, test_proba, threshold)
    
    metrics = {
        'train_auroc': roc_auc_score(y_train, model.predict_proba(X_train_scaled)[:, 1]),
        'val_auroc': roc_auc_score(y_val, model.predict_proba(X_val_scaled)[:, 1]),
        'test_auroc': roc_auc_score(y_test, test_proba),
        'decision_threshold': threshold,
        'val_recall': val_metrics['recall'],
        'val_precision': val_metrics['precision'],
        'val_f2': val_metrics['f2'],
        'test_recall': test_metrics['recall'],
        'test_precision': test_metrics['precision'],
        'test_f2': test_metrics['f2']
    }
    
    return model, scaler, metrics


def evaluate_threshold_metrics(y_true, y_scores, threshold):
    """Evaluate recall-oriented classification metrics at a fixed threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    return {
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f2': fbeta_score(y_true, y_pred, beta=2, zero_division=0),
    }


def select_recall_calibrated_threshold(y_true, y_scores, target_recall=0.85):
    """Pick the threshold closest to a target recall, preferring precision and F2 on ties."""
    best = None
    best_threshold = 0.5
    for threshold in [i / 100 for i in range(1, 100)]:
        metrics = evaluate_threshold_metrics(y_true, y_scores, threshold)
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
        'recall': best[4],
        'precision': best[5],
        'f2': best[6],
    }


def main():
    """Run federated learning baseline experiment."""
    logger.info("\n" + "=" * 80)
    logger.info("# EXPERIMENT 1: FEDERATED LEARNING BASELINE")
    logger.info("=" * 80)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-dp', action='store_true', help='Enable Gaussian DP on client coefficients')
    parser.add_argument('--epsilon', type=float, default=None, help='Override DP epsilon (overrides src.config.config.DP_EPSILON)')
    parser.add_argument('--clipping', type=float, default=None, help='Override clipping threshold (overrides src.config.config.CLIPPING_THRESHOLD)')
    parser.add_argument('--rounds', type=int, default=5, help='Number of federated rounds to run')
    parser.add_argument('--seed', type=int, default=None, help='Random seed override')
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int(os.getenv('RANDOM_SEED', '42'))

    try:
        # ===== PHASE 1.1: LOAD PHASE 0 DATA =====
        logger.info("\nPhase 1.1: Loading Phase 0 cohort and splits...")
        df_full, X, y = load_dataset_with_df(use_cache=True)
        
        logger.info(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target distribution: {y.sum()} deaths ({100*y.mean():.1f}%)")
        
        # ===== PHASE 1.2: DISTRIBUTE DATA BY CARE UNIT =====
        logger.info("\nPhase 1.2: Distributing data to federated clients by care unit...")
        
        # Split into train/val/test first (reusing Phase 0 splits)
        X_train = X[:45691]  # Phase 0 train size
        y_train = y[:45691]
        X_val = X[45691:55482]  # Phase 0 val size
        y_val = y[45691:55482]
        X_test = X[55482:]  # Phase 0 test size
        y_test = y[55482:]
        
        care_units_train = df_full.iloc[:45691]['first_careunit']
        
        # Create federated clients from training data
        clients = distribute_by_care_unit(
            X_train, y_train, care_units_train,
            min_patients_per_unit=100
        )
        
        logger.info(f"\n✓ Created {len(clients)} federated clients")
        
        # Display client statistics
        logger.info("\nClient Statistics:")
        logger.info("-" * 70)
        for unit_name, (X_c, y_c) in sorted(clients.items(), key=lambda x: -len(x[1][0])):
            logger.info(
                f"  {unit_name:40} {len(X_c):6} patients, "
                f"{int(y_c.sum()):4} deaths ({100*y_c.mean():5.1f}%)"
            )
        
        # ===== PHASE 1.3: TRAIN CENTRALIZED BASELINE =====
        logger.info("\n" + "=" * 70)
        logger.info("Phase 1.3: Training Centralized Baseline (for comparison)...")
        logger.info("=" * 70)
        
        cent_model, cent_scaler, cent_metrics = train_centralized_baseline_simple(
            X_train, y_train, X_val, y_val, X_test, y_test, seed=seed
        )
        
        logger.info(f"\nCentralized Train AUROC: {cent_metrics['train_auroc']:.4f}")
        logger.info(f"Centralized Val AUROC:   {cent_metrics['val_auroc']:.4f}")
        logger.info(f"Centralized Test AUROC:  {cent_metrics['test_auroc']:.4f}")
        logger.info(
            f"Centralized calibrated threshold: {cent_metrics['decision_threshold']:.2f} "
            f"(val recall={cent_metrics['val_recall']:.2%}, val precision={cent_metrics['val_precision']:.2%})"
        )
        logger.info(
            f"Centralized Test Recall:    {cent_metrics['test_recall']:.2%}"
        )
        logger.info(
            f"Centralized Test Precision: {cent_metrics['test_precision']:.2%}"
        )
        
        # ===== PHASE 1.4: TRAIN FEDERATED MODEL =====
        logger.info("\n" + "=" * 70)
        logger.info("Phase 1.4: Training Federated Learning Model...")
        logger.info("=" * 70)

        # Allow runtime override of DP epsilon in config
        if args.epsilon is not None:
            try:
                import src.config.config as cfg
                cfg.DP_EPSILON = float(args.epsilon)
                logger.info(f"Overriding DP epsilon to {cfg.DP_EPSILON}")
            except Exception:
                logger.warning("Could not override src.config.config.DP_EPSILON")

        if args.clipping is not None:
            try:
                import src.config.config as cfg
                cfg.CLIPPING_THRESHOLD = float(args.clipping)
                logger.info(f"Overriding clipping threshold to {cfg.CLIPPING_THRESHOLD}")
            except Exception:
                logger.warning("Could not override src.config.config.CLIPPING_THRESHOLD")

        # Import FederatedTrainer after possible config overrides so DP settings take effect
        from src.training.federated import FederatedTrainer

        trainer = FederatedTrainer(
            clients=clients,
            val_data=(X_val, y_val),
            test_data=(X_test, y_test),
            num_rounds=args.rounds,
            use_dp=args.use_dp,
            random_seed=seed
        )
        
        fed_results = trainer.train()

        # Recall-oriented clinical threshold calibration on the federated model
        fed_weights = fed_results['final_weights']
        def scores_from_weights(X_input):
            X_scaled = trainer.scaler.transform(X_input)
            logits = X_scaled @ fed_weights['coef'] + fed_weights['intercept']
            return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

        fed_val_scores = scores_from_weights(X_val)
        fed_test_scores = scores_from_weights(X_test)
        fed_threshold, fed_threshold_metrics = select_recall_calibrated_threshold(y_val, fed_val_scores)
        fed_val_clinical = evaluate_threshold_metrics(y_val, fed_val_scores, fed_threshold)
        fed_test_clinical = evaluate_threshold_metrics(y_test, fed_test_scores, fed_threshold)
        
        # ===== PHASE 1.5: COMPARISON & VALIDATION =====
        logger.info("\n" + "=" * 70)
        logger.info("Phase 1.5: Results Comparison")
        logger.info("=" * 70)
        
        fed_auroc = fed_results['test_auroc']
        cent_auroc = cent_metrics['test_auroc']
        
        logger.info(f"\nTest AUROC Comparison:")
        logger.info(f"  Centralized: {cent_auroc:.4f}")
        logger.info(f"  Federated:   {fed_auroc:.4f}")
        logger.info(f"  Divergence:  {abs(cent_auroc - fed_auroc):.4f}")
        logger.info(
            f"\nClinical operating point (threshold={fed_threshold:.2f}):"
        )
        logger.info(
            f"  Federated Recall:    {fed_test_clinical['recall']:.2%}"
        )
        logger.info(
            f"  Federated Precision: {fed_test_clinical['precision']:.2%}"
        )
        logger.info(
            f"  Federated Val Recall: {fed_val_clinical['recall']:.2%}"
        )
        logger.info(
            f"  Federated Val Precision: {fed_val_clinical['precision']:.2%}"
        )
        
        # Check Phase 1 success criteria
        success = True
        if fed_auroc < 0.85:
            logger.warning(f"⚠ Federated AUROC ({fed_auroc:.4f}) < 0.85 target")
            success = False
        else:
            logger.info(f"✓ Federated AUROC ({fed_auroc:.4f}) ≥ 0.85 target")
        
        if abs(cent_auroc - fed_auroc) > 0.05:
            logger.warning(
                f"⚠ Divergence between centralized and federated "
                f"({abs(cent_auroc - fed_auroc):.4f}) > 0.05"
            )
            success = False
        else:
            logger.info(f"✓ Divergence ({abs(cent_auroc - fed_auroc):.4f}) within 0.05")

        if fed_test_clinical['recall'] < 0.80:
            logger.warning(
                f"⚠ Federated recall ({fed_test_clinical['recall']:.2%}) < 80% clinical target"
            )
            success = False
        else:
            logger.info(
                f"✓ Federated recall ({fed_test_clinical['recall']:.2%}) ≥ 80% clinical target"
            )
        
        logger.info("\n" + "=" * 70)
        if success:
            logger.info("✅ EXPERIMENT 1 PASSED - Ready for Phase 2")
        else:
            logger.info("⚠️  EXPERIMENT 1 PARTIAL - Manual review recommended")
        logger.info("=" * 70)
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"\n❌ Experiment 1 failed:")
        logger.error(f"{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
