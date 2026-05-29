#!/usr/bin/env python3
"""
Phase 2.2 Core Experiments: Baseline + Federated + Calibration on Multi-Dataset

Runs three core experiments to address reviewer's concern about single-dataset evaluation:
1. Centralized logistic regression baseline
2. Federated FedAvg across clients  
3. Calibration validation (ECE, Platt scaling)

Supports both MIMIC-IV and eICU-CRD datasets.

Usage:
    python scripts/phase2_core_experiments.py --dataset eicu_crd [--seed 42]
    python scripts/phase2_core_experiments.py --dataset mimic_iv
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score, fbeta_score
from sklearn.calibration import CalibratedClassifierCV

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.multi_dataset import load_dataset, distribute_by_partition, get_dataset_metadata
from src.config.config import GCP_PROJECT_ID, TEST_SIZE, RANDOM_SEED

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_threshold_metrics(y_true, y_scores, threshold):
    """Evaluate metrics at a specific threshold"""
    y_pred = (y_scores >= threshold).astype(int)
    
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    
    return {
        'recall': recall,
        'precision': precision,
        'f2': f2,
    }


def distribute_eicu_hospitals(X, y, hospital_ids, top_n=7, min_positive_samples=2):
    """
    Distribute eICU data to top N hospitals as federated clients
    
    Args:
        X: Feature matrix
        y: Target vector
        hospital_ids: Hospital ID for each sample
        top_n: Number of top hospitals to use
        min_positive_samples: Minimum positive samples per hospital
        
    Returns:
        Dict: {hospital_id: (X_hospital, y_hospital)}
    """
    client_data = {}
    hospital_counts = hospital_ids.value_counts()
    top_hospitals = hospital_counts.head(top_n).index.tolist()
    
    logger.info(f"Using top {top_n} hospitals by sample count")
    
    for hospital_id in top_hospitals:
        mask = hospital_ids == hospital_id
        X_hospital = X[mask]
        y_hospital = y[mask]
        
        if y_hospital.sum() >= min_positive_samples:
            client_data[hospital_id] = (X_hospital, y_hospital)
            mortality = y_hospital.mean()
            logger.info(f"  Hospital {hospital_id}: {len(X_hospital)} samples, "
                       f"{int(y_hospital.sum())} deaths ({mortality*100:.1f}%)")
        else:
            logger.warning(f"  Hospital {hospital_id}: Skipped ({int(y_hospital.sum())} positive samples)")
    
    total_samples = sum(len(X_c[0]) for X_c in client_data.values())
    logger.info(f"Total clients: {len(client_data)}, total samples: {total_samples}")
    
    return client_data


def select_recall_calibrated_threshold(y_val, y_scores_val):
    """Select threshold that maximizes F2 score (recall-oriented)"""
    thresholds = np.linspace(0.01, 0.99, 50)
    best = None
    best_threshold = 0.5
    
    for threshold in thresholds:
        metrics = evaluate_threshold_metrics(y_val, y_scores_val, threshold)
        candidate = (
            -metrics['recall'] * 2 + metrics['precision'],
            threshold,
            metrics['recall'],
            metrics['precision'],
            metrics['f2'],
        )
        
        if best is None or candidate < best:
            best = candidate
            best_threshold = threshold
    
    return best_threshold, {
        'recall': best[2],
        'precision': best[3],
        'f2': best[4],
    }


def train_centralized_baseline(X_train, y_train, X_val, y_val, X_test, y_test, seed=42):
    """Train centralized logistic regression baseline"""
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
        'val_auroc': roc_auc_score(y_val, val_proba),
        'test_auroc': roc_auc_score(y_test, test_proba),
        'decision_threshold': float(threshold),
        'val_recall': float(val_metrics['recall']),
        'val_precision': float(val_metrics['precision']),
        'val_f2': float(val_metrics['f2']),
        'test_recall': float(test_metrics['recall']),
        'test_precision': float(test_metrics['precision']),
        'test_f2': float(test_metrics['f2']),
    }
    
    return model, scaler, metrics


def train_federated_model(clients, X_val, y_val, X_test, y_test, seed=42, num_rounds=5):
    """Train federated model using simple averaging (FedAvg)"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Training Federated Model ({len(clients)} clients, {num_rounds} rounds)")
    logger.info(f"{'='*70}")
    
    scaler = StandardScaler()
    
    # Pre-fit scaler on aggregated training data
    all_X = np.vstack([X_c[0] for X_c in clients.values()])
    scaler.fit(all_X)
    
    for round_num in range(1, num_rounds + 1):
        logger.info(f"\nRound {round_num}/{num_rounds}:")
        
        client_coefs = []
        client_intercepts = []
        
        # Train local models
        for client_id, (X_c, y_c) in clients.items():
            X_c_scaled = scaler.transform(X_c)
            
            local_model = LogisticRegression(max_iter=1000, random_state=seed, class_weight='balanced')
            local_model.fit(X_c_scaled, y_c)
            
            client_coefs.append(local_model.coef_[0])
            client_intercepts.append(local_model.intercept_[0])
        
        # FedAvg: aggregate coefficients
        global_coef = np.mean(client_coefs, axis=0)
        global_intercept = np.mean(client_intercepts, axis=0)
    
    # Create final model with aggregated weights
    # Fit on all data to initialize model properly
    all_X_scaled = scaler.transform(all_X)
    all_y = np.hstack([y_c[1] for y_c in clients.values()])
    
    model = LogisticRegression(max_iter=1000, random_state=seed, class_weight='balanced')
    model.fit(all_X_scaled, all_y)
    
    # Replace coefficients with aggregated values
    model.coef_ = global_coef.reshape(1, -1)
    model.intercept_ = global_intercept.reshape(1)
    
    # Evaluate on validation and test sets
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    val_proba = model.predict_proba(X_val_scaled)[:, 1]
    test_proba = model.predict_proba(X_test_scaled)[:, 1]
    threshold, threshold_metrics = select_recall_calibrated_threshold(y_val, val_proba)
    
    val_metrics = evaluate_threshold_metrics(y_val, val_proba, threshold)
    test_metrics = evaluate_threshold_metrics(y_test, test_proba, threshold)
    
    metrics = {
        'train_auroc': roc_auc_score(all_y, model.predict_proba(all_X_scaled)[:, 1]),
        'val_auroc': roc_auc_score(y_val, val_proba),
        'test_auroc': roc_auc_score(y_test, test_proba),
        'decision_threshold': float(threshold),
        'val_recall': float(val_metrics['recall']),
        'val_precision': float(val_metrics['precision']),
        'val_f2': float(val_metrics['f2']),
        'test_recall': float(test_metrics['recall']),
        'test_precision': float(test_metrics['precision']),
        'test_f2': float(test_metrics['f2']),
    }
    
    return model, scaler, metrics


def validate_calibration(model, scaler, X_val, y_val, X_test, y_test, seed=42):
    """Validate model calibration using Platt scaling and ECE"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Validating Calibration")
    logger.info(f"{'='*70}")
    
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Get original probabilities
    test_proba_original = model.predict_proba(X_test_scaled)[:, 1]
    
    # Fit Platt scaling model on validation set
    val_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    platt_model = LogisticRegression(max_iter=1000, random_state=seed)
    platt_model.fit(val_proba.reshape(-1, 1), y_val)
    
    # Get calibrated probabilities
    test_proba_calibrated = platt_model.predict_proba(test_proba_original.reshape(-1, 1))[:, 1]
    
    # Compute Expected Calibration Error (ECE)
    def compute_ece(y_true, y_proba, num_bins=10):
        bins = np.linspace(0, 1, num_bins + 1)
        ece = 0
        
        for i in range(num_bins):
            mask = (y_proba >= bins[i]) & (y_proba < bins[i+1])
            if mask.sum() > 0:
                predicted_prob = y_proba[mask].mean()
                true_prob = y_true[mask].mean()
                ece += abs(predicted_prob - true_prob) * mask.sum() / len(y_true)
        
        return ece
    
    ece_original = compute_ece(y_test, test_proba_original)
    ece_calibrated = compute_ece(y_test, test_proba_calibrated)
    
    metrics = {
        'ece_original': float(ece_original),
        'ece_calibrated': float(ece_calibrated),
        'ece_improvement': float(ece_original - ece_calibrated),
    }
    
    logger.info(f"ECE (original): {ece_original:.4f}")
    logger.info(f"ECE (calibrated): {ece_calibrated:.4f}")
    logger.info(f"ECE improvement: {ece_original - ece_calibrated:.4f}")
    
    return platt_model, metrics


def run_experiments(dataset_name='mimic_iv', seed=42, num_rounds=5):
    """Run all three experiments"""
    logger.info(f"\n{'='*80}")
    logger.info(f"PHASE 2.2: CORE EXPERIMENTS ON {dataset_name.upper()}")
    logger.info(f"{'='*80}")
    
    # Load dataset
    logger.info(f"\nLoading {dataset_name} dataset...")
    df, X, y = load_dataset(dataset_name, use_cache=True, billing_project=GCP_PROJECT_ID)
    
    logger.info(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    logger.info(f"Target: {y.sum()} deaths ({y.mean()*100:.1f}%)")
    
    # Split data
    logger.info(f"\nSplitting data (70% train, 15% val, 15% test)...")
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=seed, stratify=y)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed, stratify=y[temp_idx])
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    logger.info(f"  Train: {len(X_train)} samples ({y_train.mean()*100:.1f}% mortality)")
    logger.info(f"  Val: {len(X_val)} samples ({y_val.mean()*100:.1f}% mortality)")
    logger.info(f"  Test: {len(X_test)} samples ({y_test.mean()*100:.1f}% mortality)")
    
    # Distribute to clients
    logger.info(f"\nDistributing training data to federated clients...")
    df_train = df.iloc[train_idx].reset_index(drop=True)
    
    # For eICU, use top 7 hospitals; for MIMIC, use care units
    if dataset_name == 'eicu_crd':
        clients = distribute_eicu_hospitals(X_train, y_train, df_train['hospitalid'], top_n=7)
    else:
        clients = distribute_by_partition(X_train, y_train, df_train, dataset_name)
    
    logger.info(f"Created {len(clients)} federated clients")
    
    # Experiment 1: Centralized Baseline
    logger.info(f"\n{'='*70}")
    logger.info(f"EXPERIMENT 1: Centralized Baseline")
    logger.info(f"{'='*70}")
    
    cent_model, cent_scaler, cent_metrics = train_centralized_baseline(
        X_train, y_train, X_val, y_val, X_test, y_test, seed=seed
    )
    
    logger.info(f"\nCentralized Results:")
    logger.info(f"  Test AUROC: {cent_metrics['test_auroc']:.4f}")
    logger.info(f"  Test Recall: {cent_metrics['test_recall']:.2%}")
    logger.info(f"  Test Precision: {cent_metrics['test_precision']:.2%}")
    
    # Experiment 2: Federated Learning
    logger.info(f"\n{'='*70}")
    logger.info(f"EXPERIMENT 2: Federated Learning (FedAvg)")
    logger.info(f"{'='*70}")
    
    fed_model, fed_scaler, fed_metrics = train_federated_model(
        clients, X_val, y_val, X_test, y_test, seed=seed, num_rounds=num_rounds
    )
    
    logger.info(f"\nFederated Results:")
    logger.info(f"  Test AUROC: {fed_metrics['test_auroc']:.4f}")
    logger.info(f"  Test Recall: {fed_metrics['test_recall']:.2%}")
    logger.info(f"  Test Precision: {fed_metrics['test_precision']:.2%}")
    
    # Calculate AUROC loss
    auroc_loss = (cent_metrics['test_auroc'] - fed_metrics['test_auroc']) / cent_metrics['test_auroc'] * 100
    logger.info(f"\n  AUROC Loss (Federated vs Centralized): {auroc_loss:.2f}%")
    
    # Experiment 3: Calibration
    logger.info(f"\n{'='*70}")
    logger.info(f"EXPERIMENT 3: Calibration Validation")
    logger.info(f"{'='*70}")
    
    cal_model, cal_metrics = validate_calibration(
        cent_model, cent_scaler, X_val, y_val, X_test, y_test, seed=seed
    )
    
    # Save results
    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'seed': seed,
        'num_clients': len(clients),
        'num_rounds': num_rounds,
        'centralized': cent_metrics,
        'federated': {**fed_metrics, 'auroc_loss_pct': auroc_loss},
        'calibration': cal_metrics,
    }
    
    output_dir = Path(__file__).parent.parent / "results" / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"phase2_core_{dataset_name}_{int(datetime.now().timestamp())}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {output_path}")
    
    # Summary
    logger.info(f"\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\nDataset: {dataset_name}")
    logger.info(f"Centralized Test AUROC: {cent_metrics['test_auroc']:.4f}")
    logger.info(f"Federated Test AUROC: {fed_metrics['test_auroc']:.4f}")
    logger.info(f"AUROC Loss: {auroc_loss:.2f}%")
    logger.info(f"ECE (original): {cal_metrics['ece_original']:.4f}")
    logger.info(f"ECE (calibrated): {cal_metrics['ece_calibrated']:.4f}")
    
    # Validation
    status = "✅" if auroc_loss < 3 else "⚠️" if auroc_loss < 5 else "❌"
    logger.info(f"\n{status} AUROC Loss Target: <3% → Achieved: {auroc_loss:.2f}%")
    
    status = "✅" if cal_metrics['ece_calibrated'] < 0.02 else "⚠️" if cal_metrics['ece_calibrated'] < 0.03 else "❌"
    logger.info(f"{status} Calibration Target: ECE <0.02 → Achieved: {cal_metrics['ece_calibrated']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Phase 2.2 Core Experiments')
    parser.add_argument('--dataset', choices=['mimic_iv', 'eicu_crd'], default='eicu_crd',
                        help='Dataset to run experiments on')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--rounds', type=int, default=5,
                        help='Number of federated rounds')
    
    args = parser.parse_args()
    
    results = run_experiments(args.dataset, seed=args.seed, num_rounds=args.rounds)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
