"""Experiment 1B: Multi-Model Federated Learning Baseline (LR and MLP)

Extends Experiment 1 to support multiple model architectures.
Compares Logistic Regression (baseline) with MLP (neural network).

Each model architecture is evaluated in:
1. Centralized training
2. Federated learning (FedAvg, 5 rounds)
3. (Optional) Differentially private federated learning

Requirements:
- Phase 0 cohort validation completed (baseline AUROC 0.8920 for LR)
- Cached cohort at data/cache/mimic_iv_cohort.csv
- PyTorch installed for MLP models

Output:
- results/plots/multimodel_comparison.csv
  (Model, Centralized_AUROC, Fed_AUROC, DP_AUROC, Centralized_Recall, Fed_Recall, etc.)
"""

import sys
import os
import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, fbeta_score
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_dataset_with_df
from src.data.split import distribute_by_care_unit
from src.models.model import create_model, LogisticRegressionModel, MLPModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def select_recall_calibrated_threshold(y_true, y_scores, target_recall=0.85):
    """Pick the threshold closest to a target recall, preferring precision and F2 on ties."""
    best = None
    best_threshold = 0.5
    for threshold in [i / 100 for i in range(1, 100)]:
        y_pred = (y_scores >= threshold).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        
        candidate = (
            abs(recall - target_recall),
            -precision,
            -f2,
            threshold,
            recall,
            precision,
            f2,
        )
        if best is None or candidate < best:
            best = candidate
            best_threshold = threshold

    return best_threshold, {
        'recall': best[4],
        'precision': best[5],
        'f2': best[6],
    }


def evaluate_threshold_metrics(y_true, y_scores, threshold):
    """Evaluate recall-oriented classification metrics at a fixed threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    return {
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f2': fbeta_score(y_true, y_pred, beta=2, zero_division=0),
    }


def train_centralized_model(model_type, X_train, y_train, X_val, y_val, X_test, y_test, seed=42):
    """Train a centralized model (any type: LR, MLP, etc.)"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    if model_type == 'logistic_regression':
        model = create_model('logistic_regression', random_state=seed)
        model.fit(X_train_scaled, y_train)
    elif model_type == 'mlp':
        model = create_model('mlp', input_dim=31, hidden_layers=[64, 32], 
                            epochs=20, batch_size=32, learning_rate=0.001, random_state=seed)
        model.fit(X_train_scaled, y_train, verbose=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Get probabilities
    val_proba = model.predict_proba(X_val_scaled)
    test_proba = model.predict_proba(X_test_scaled)
    train_proba = model.predict_proba(X_train_scaled)
    
    # Handle different output formats
    if isinstance(val_proba, np.ndarray) and val_proba.ndim == 2 and val_proba.shape[1] == 2:
        # Sklearn format: (n_samples, 2)
        val_proba = val_proba[:, 1]
        test_proba = test_proba[:, 1]
        train_proba = train_proba[:, 1]
    
    # Select threshold
    threshold, _ = select_recall_calibrated_threshold(y_val, val_proba)
    
    # Evaluate
    val_metrics = evaluate_threshold_metrics(y_val, val_proba, threshold)
    test_metrics = evaluate_threshold_metrics(y_test, test_proba, threshold)
    
    metrics = {
        'train_auroc': roc_auc_score(y_train, train_proba),
        'val_auroc': roc_auc_score(y_val, val_proba),
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


def train_federated_model_simple(model_type, clients, X_val, y_val, X_test, y_test, 
                                  scaler, rounds=5, seed=42):
    """
    Train a federated model with simple federated averaging (no DP-SGD).
    Supports any model type that has get_weights/set_weights.
    """
    np.random.seed(seed)
    
    # Initialize global model with zero weights
    if model_type == 'logistic_regression':
        n_features = X_val.shape[1]
        global_weights = {
            'coef': np.zeros(n_features),
            'intercept': np.array([0.0]),
            'classes': np.array([0, 1])
        }
    elif model_type == 'mlp':
        # Create temporary model to get weight shape
        temp_model = create_model('mlp', input_dim=31, hidden_layers=[64, 32], 
                                 epochs=1, random_state=seed)
        dummy_X = np.random.randn(10, 31)
        dummy_y = np.random.randint(0, 2, 10)
        temp_model.fit(dummy_X, dummy_y, verbose=False)
        global_weights = temp_model.get_weights()
        global_weights = np.zeros_like(global_weights)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Federated averaging rounds
    round_losses = []
    for round_num in range(rounds):
        logger.info(f"  Federated Round {round_num + 1}/{rounds}")
        
        client_weights = []
        client_sizes = []
        
        for unit_name, (X_client, y_client) in clients.items():
            X_scaled = scaler.transform(X_client)
            
            # Train local model
            if model_type == 'logistic_regression':
                local_model = create_model('logistic_regression', random_state=seed)
                local_model.fit(X_scaled, y_client)
                weights = local_model.get_weights()
            elif model_type == 'mlp':
                local_model = create_model('mlp', input_dim=31, hidden_layers=[64, 32],
                                          epochs=5, batch_size=32, learning_rate=0.001, random_state=seed)
                local_model.fit(X_scaled, y_client, verbose=False)
                weights = local_model.get_weights()
            
            client_weights.append(weights)
            client_sizes.append(len(X_client))
        
        # Aggregate weights
        if model_type == 'logistic_regression':
            # Average coef and intercept
            avg_coef = np.mean([w['coef'] for w in client_weights], axis=0)
            avg_intercept = np.mean([w['intercept'] for w in client_weights], axis=0)
            global_weights = {
                'coef': avg_coef,
                'intercept': avg_intercept,
                'classes': np.array([0, 1])
            }
        elif model_type == 'mlp':
            # Average weight vectors
            avg_weights = np.mean(client_weights, axis=0)
            global_weights = avg_weights
        
        round_losses.append(0.0)  # Simplified - not tracking loss per round
    
    # Evaluate on validation and test sets
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    if model_type == 'logistic_regression':
        # Use weights directly to compute predictions
        val_logits = X_val_scaled @ global_weights['coef'] + global_weights['intercept']
        test_logits = X_test_scaled @ global_weights['coef'] + global_weights['intercept']
        val_proba = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -30.0, 30.0)))
        test_proba = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -30.0, 30.0)))
    elif model_type == 'mlp':
        # Create model and set weights
        eval_model = create_model('mlp', input_dim=31, hidden_layers=[64, 32], epochs=1, random_state=seed)
        dummy_X = np.random.randn(10, 31)
        dummy_y = np.random.randint(0, 2, 10)
        eval_model.fit(dummy_X, dummy_y, verbose=False)
        eval_model.set_weights(global_weights)
        val_proba = eval_model.predict_proba(X_val_scaled)
        test_proba = eval_model.predict_proba(X_test_scaled)
    
    # Select threshold and evaluate
    threshold, _ = select_recall_calibrated_threshold(y_val, val_proba)
    val_metrics = evaluate_threshold_metrics(y_val, val_proba, threshold)
    test_metrics = evaluate_threshold_metrics(y_test, test_proba, threshold)
    
    metrics = {
        'val_auroc': roc_auc_score(y_val, val_proba),
        'test_auroc': roc_auc_score(y_test, test_proba),
        'decision_threshold': threshold,
        'val_recall': val_metrics['recall'],
        'val_precision': val_metrics['precision'],
        'val_f2': val_metrics['f2'],
        'test_recall': test_metrics['recall'],
        'test_precision': test_metrics['precision'],
        'test_f2': test_metrics['f2']
    }
    
    return metrics


def main():
    """Run multi-model federated learning baseline experiment."""
    logger.info("\n" + "=" * 80)
    logger.info("# EXPERIMENT 1B: MULTI-MODEL FEDERATED LEARNING BASELINE")
    logger.info("=" * 80)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all', 
                       choices=['logistic_regression', 'mlp', 'all'],
                       help='Which model(s) to train')
    parser.add_argument('--rounds', type=int, default=5, help='Number of federated rounds')
    parser.add_argument('--seed', type=int, default=None, help='Random seed override')
    args = parser.parse_args()
    
    seed = args.seed if args.seed is not None else int(os.getenv('RANDOM_SEED', '42'))
    models_to_train = ['logistic_regression', 'mlp'] if args.model == 'all' else [args.model]
    
    try:
        # ===== LOAD DATA =====
        logger.info("\nLoading MIMIC-IV cohort...")
        df_full, X, y = load_dataset_with_df(use_cache=True)
        logger.info(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # ===== SPLIT DATA =====
        indices = np.arange(len(y))
        train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=seed, stratify=y)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed, stratify=y[temp_idx])
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        care_units_train = df_full.iloc[train_idx]['first_careunit']
        
        # ===== CREATE FEDERATED CLIENTS =====
        clients = distribute_by_care_unit(X_train, y_train, care_units_train, min_patients_per_unit=100)
        logger.info(f"Created {len(clients)} federated clients (ICU care units)")
        
        # ===== TRAIN MODELS =====
        results = []
        
        for model_type in models_to_train:
            logger.info("\n" + "=" * 70)
            logger.info(f"Training {model_type.upper()} Model")
            logger.info("=" * 70)
            
            # Centralized
            logger.info(f"\nCentralized training ({model_type})...")
            cent_model, cent_scaler, cent_metrics = train_centralized_model(
                model_type, X_train, y_train, X_val, y_val, X_test, y_test, seed=seed
            )
            logger.info(f"  ✓ Centralized Test AUROC: {cent_metrics['test_auroc']:.4f}")
            logger.info(f"  ✓ Centralized Test Recall: {cent_metrics['test_recall']:.2%}")
            
            # Federated
            logger.info(f"\nFederated training ({model_type}, {args.rounds} rounds)...")
            fed_metrics = train_federated_model_simple(
                model_type, clients, X_val, y_val, X_test, y_test, cent_scaler, 
                rounds=args.rounds, seed=seed
            )
            logger.info(f"  ✓ Federated Test AUROC: {fed_metrics['test_auroc']:.4f}")
            logger.info(f"  ✓ Federated Test Recall: {fed_metrics['test_recall']:.2%}")
            
            # Record results
            result_row = {
                'model': model_type,
                'centralized_auroc': cent_metrics['test_auroc'],
                'centralized_recall': cent_metrics['test_recall'],
                'centralized_precision': cent_metrics['test_precision'],
                'federated_auroc': fed_metrics['test_auroc'],
                'federated_recall': fed_metrics['test_recall'],
                'federated_precision': fed_metrics['test_precision'],
                'centralized_threshold': cent_metrics['decision_threshold'],
                'federated_threshold': fed_metrics['decision_threshold'],
            }
            results.append(result_row)
        
        # ===== SAVE RESULTS =====
        logger.info("\n" + "=" * 70)
        logger.info("Results Summary")
        logger.info("=" * 70)
        
        import pandas as pd
        results_df = pd.DataFrame(results)
        
        results_file = Path(__file__).parent.parent / 'results' / 'plots' / 'multimodel_comparison.csv'
        results_file.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_file, index=False)
        logger.info(f"\n✓ Results saved to {results_file}")
        
        # Print summary
        logger.info("\nModel Comparison:")
        logger.info(results_df.to_string(index=False))
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ EXPERIMENT 1B COMPLETED")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Experiment 1B failed:")
        logger.error(f"{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
