"""Apply Platt scaling (sigmoid / CalibratedClassifierCV) to FedAvg outputs.

This script recreates the data split consistent with `exp1_baseline.py`,
trains (or reruns) the federated trainer to obtain the final FedAvg weights,
wraps those weights into a predict_proba-compatible estimator and fits
`sklearn.calibration.CalibratedClassifierCV(method='sigmoid', cv='prefit')`
on the validation set. It then evaluates calibrated probabilities on the
test set at the centralized decision threshold (0.39) and writes a small
CSV summarizing the calibrated metrics.

Usage:
  python experiments/apply_platt_scaling.py --rounds 5 --seed 42 --use-dp

The script is intentionally small and reuses existing data-loading and
trainer code from the repo to preserve identical preprocessing and splits.
"""
import sys
import os
from pathlib import Path
import argparse
import logging

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, recall_score, precision_score, fbeta_score

# Add parent directory to import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_dataset_with_df
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.training.federated import FederatedTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_threshold_metrics(y_true, y_scores, threshold):
    y_pred = (y_scores >= threshold).astype(int)
    return {
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f2': fbeta_score(y_true, y_pred, beta=2, zero_division=0),
    }


class FedEstimator:
    """Simple sklearn-like estimator wrapping federated linear weights."""
    def __init__(self, coef: np.ndarray, intercept: float, scaler: StandardScaler):
        self.coef = np.asarray(coef)
        self.intercept = float(intercept)
        self.scaler = scaler
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        logits = Xs @ self.coef + self.intercept
        logits = np.clip(logits, -30.0, 30.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-dp', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.39, help='Target centralized decision threshold')
    args = parser.parse_args()

    seed = args.seed

    # Load dataset and create reproducible splits matching exp1_baseline
    df_full, X, y = load_dataset_with_df(use_cache=True)
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=seed, stratify=y)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed, stratify=y[temp_idx])

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    care_units_train = df_full.iloc[train_idx]['first_careunit']
    from src.data.split import distribute_by_care_unit

    clients = distribute_by_care_unit(X_train, y_train, care_units_train, min_patients_per_unit=100)

    logger.info(f"Created {len(clients)} federated clients; training rounds={args.rounds}")

    trainer = FederatedTrainer(
        clients=clients,
        val_data=(X_val, y_val),
        test_data=(X_test, y_test),
        num_rounds=args.rounds,
        use_dp=args.use_dp,
        random_seed=seed
    )

    fed_results = trainer.train()
    fed_weights = fed_results['final_weights']

    # Wrap into estimator and calibrate on validation set with Platt scaling
    base_est = FedEstimator(fed_weights['coef'], fed_weights['intercept'], trainer.scaler)

    logger.info("Fitting Platt scaling (sigmoid) on validation logits with 1D logistic regression")
    # Compute uncalibrated logits for val/test using fed weights and global scaler
    X_val_scaled = trainer.scaler.transform(X_val)
    X_test_scaled = trainer.scaler.transform(X_test)
    logits_val = X_val_scaled @ fed_weights['coef'] + fed_weights['intercept']
    logits_test = X_test_scaled @ fed_weights['coef'] + fed_weights['intercept']

    from sklearn.linear_model import LogisticRegression as _LR
    platt = _LR(max_iter=1000)
    platt.fit(logits_val.reshape(-1, 1), y_val)
    calibrated_proba_test = platt.predict_proba(logits_test.reshape(-1, 1))[:, 1]
    auroc = roc_auc_score(y_test, calibrated_proba_test)
    thresh = args.threshold
    metrics_at_threshold = evaluate_threshold_metrics(y_test, calibrated_proba_test, thresh)

    logger.info(f"Calibrated Test AUROC: {auroc:.4f}")
    logger.info(f"At threshold={thresh:.2f}: Recall={metrics_at_threshold['recall']:.3f}, Precision={metrics_at_threshold['precision']:.3f}")

    # Save a small CSV summary
    out_dir = Path('results/summary')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'calibrated_fed_results_rounds{args.rounds}_seed{seed}_dp{int(args.use_dp)}.csv'
    import pandas as pd
    df = pd.DataFrame([{
        'rounds': args.rounds,
        'seed': seed,
        'use_dp': args.use_dp,
        'calibrated_test_auroc': auroc,
        'threshold': thresh,
        'test_recall': metrics_at_threshold['recall'],
        'test_precision': metrics_at_threshold['precision'],
        'test_f2': metrics_at_threshold['f2']
    }])
    df.to_csv(out_file, index=False)
    logger.info(f"Wrote calibrated summary to {out_file}")


if __name__ == '__main__':
    main()
