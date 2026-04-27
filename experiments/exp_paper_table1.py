#!/usr/bin/env python3
"""
Master Verification Experiment for Paper Table 1
=================================================
Produces ALL numbers cited in the paper with 5-seed averaging.
Every configuration uses the SAME threshold (0.30), preprocessing,
and feature engineering for fair comparison.

Configurations:
  A) Centralized Baseline (threshold=0.30, feature eng)
  B) FL Baseline (FedAvg, No DP, 5 clients, 10 rounds)
  C) FL + DP-SGD (multiple epsilon values for privacy-utility curve)
  D) FL + DP-SGD + Krum aggregation
  E) FL + DP-SGD + Median aggregation

All configs run with seeds [42, 123, 456, 789, 2024], results averaged with std.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from src.data.loader import load_dataset_with_df
from src.data.preprocess import DataPreprocessor
from src.data.split import train_test_split_data, distribute_non_iid
from src.models.model import LogisticRegressionModel
from src.utils.feature_engineering import HealthcareFeatureEngineer
from src.fl.strategy import FedAvgAggregator
from src.fl.privacy import DifferentialPrivacyMechanism
from src.fl.robust_aggregation import RobustAggregator
from src.config.config import MAX_ITER, DECISION_THRESHOLD

# =====================================================================
# CONFIGURATION
# =====================================================================
SEEDS = [42, 123, 456, 789, 2024]
NUM_SEEDS = len(SEEDS)
THRESHOLD = 0.30
N_CLIENTS = 5
N_ROUNDS = 10
ALPHA = 0.5

# DP configurations for privacy-utility curve
DP_EPSILON_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0]
DP_DELTA = 0.01
DP_CLIPPING_NORM = 1.0


def load_and_preprocess():
    """Load, preprocess, split, and feature-engineer the data.
    Uses fixed random_state=42 for the train/test split (same as config).
    """
    df, X, y = load_dataset_with_df()
    preprocessor = DataPreprocessor()
    X_processed = preprocessor.preprocess(df.iloc[:, :-1], fit=True)
    feature_names = list(df.columns[:-1])

    # train_test_split_data uses random_state=42 from config — this is fixed
    X_train, X_test, y_train, y_test = train_test_split_data(X_processed, y)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Feature engineering (deterministic — no randomness)
    engineer = HealthcareFeatureEngineer()
    X_train_eng, feature_names_eng = engineer.engineer_all_features(X_train, feature_names)

    # Apply same to test set
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    for feat1, feat2 in engineer.interaction_pairs:
        if feat1 in feature_names and feat2 in feature_names:
            X_test_df[f"{feat1}_x_{feat2}"] = X_test_df[feat1] * X_test_df[feat2]
    for feat in ['Glucose', 'BMI', 'Age', 'BloodPressure']:
        if feat in feature_names:
            X_test_df[f"{feat}_squared"] = X_test_df[feat] ** 2
    for feat1, feat2 in [('Glucose', 'Insulin'), ('BloodPressure', 'Age')]:
        if feat1 in feature_names and feat2 in feature_names:
            denominator = X_test_df[feat2] + 1e-6
            X_test_df[f"{feat1}_per_{feat2}"] = X_test_df[feat1] / denominator
    X_test_eng = X_test_df[feature_names_eng].values

    return X_train_eng, X_test_eng, y_train, y_test, feature_names_eng


def evaluate_predictions(y_test, y_pred):
    """Evaluate predictions and return metrics dict."""
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'missed_patients': int(fn),
        'total_positive': int(tp + fn),
    }


def run_centralized(X_train, X_test, y_train, y_test, seed):
    """Config A: Centralized baseline with threshold=0.30."""
    model = LogisticRegressionModel(
        max_iter=MAX_ITER, class_weight='balanced', random_state=seed
    )
    model.set_decision_threshold(THRESHOLD)
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict(X_test)
    return evaluate_predictions(y_test, y_pred)


def robust_aggregate_weights(client_weights_list, client_sample_sizes, method):
    """
    Apply robust aggregation (krum/median) to dict-format weights.
    Bridges between the dict-based weight format and RobustAggregator's array interface.
    """
    robust_agg = RobustAggregator(method=method, verbose=False)

    # Aggregate coefficients
    coef_arrays = [np.array(w['coef']).flatten() for w in client_weights_list]
    agg_coef = robust_agg.aggregate(
        [c.reshape(1, -1) for c in coef_arrays],
        client_sample_sizes
    )

    # Aggregate intercepts
    intercept_arrays = [np.array(w['intercept']).flatten() for w in client_weights_list]
    agg_intercept = robust_agg.aggregate(
        [i.reshape(1, -1) for i in intercept_arrays],
        client_sample_sizes
    )

    return {
        'coef': agg_coef.flatten(),
        'intercept': agg_intercept.flatten(),
        'classes': client_weights_list[0]['classes'].copy()
    }


def run_fl(X_train, X_test, y_train, y_test, seed,
           dp_epsilon=None, aggregation='fedavg'):
    """
    Configs B-E: Federated Learning with various configurations.

    Args:
        dp_epsilon: If not None, apply DP-SGD with this epsilon
        aggregation: 'fedavg', 'krum', or 'median'
    """
    np.random.seed(seed)

    # Distribute data to clients (Non-IID)
    client_data_dict = distribute_non_iid(X_train, y_train, N_CLIENTS, ALPHA)

    client_data = []
    for cid in range(N_CLIENTS):
        X_c, y_c = client_data_dict[cid]
        client_data.append({
            'X_train': X_c, 'y_train': y_c,
            'n_samples': len(X_c)
        })

    # Initialize global model from first client
    init_model = LogisticRegressionModel(
        max_iter=MAX_ITER, class_weight='balanced', random_state=seed
    )
    init_model.set_decision_threshold(THRESHOLD)
    init_model.fit(client_data[0]['X_train'], client_data[0]['y_train'], verbose=False)
    global_weights = init_model.get_weights()

    # Initialize DP mechanism if needed
    dp_mechanism = None
    if dp_epsilon is not None:
        dp_mechanism = DifferentialPrivacyMechanism(
            epsilon=dp_epsilon,
            delta=DP_DELTA,
            clipping_norm=DP_CLIPPING_NORM,
            num_samples=len(X_train)
        )

    # Federated learning rounds
    for round_num in range(N_ROUNDS):
        client_weights_list = []
        client_sample_sizes = []

        for client in client_data:
            # Create local model with global weights
            local_model = LogisticRegressionModel(
                max_iter=MAX_ITER, class_weight='balanced', random_state=seed
            )
            local_model.set_weights(deepcopy(global_weights))
            local_model.set_decision_threshold(THRESHOLD)

            # Local training
            try:
                local_model.fit(client['X_train'], client['y_train'], verbose=False)
                local_weights = local_model.get_weights()
            except ValueError:
                local_weights = deepcopy(global_weights)

            # Apply DP if enabled
            if dp_mechanism is not None:
                local_weights, _ = dp_mechanism.privatize_weights(local_weights)

            client_weights_list.append(local_weights)
            client_sample_sizes.append(client['n_samples'])

        # Aggregate
        if aggregation in ('krum', 'median'):
            global_weights = robust_aggregate_weights(
                client_weights_list, client_sample_sizes, aggregation
            )
        else:
            global_weights = FedAvgAggregator.aggregate(
                client_weights_list, client_sample_sizes
            )

    # Final evaluation
    final_model = LogisticRegressionModel(
        max_iter=MAX_ITER, class_weight='balanced', random_state=seed
    )
    final_model.set_weights(global_weights)
    final_model.set_decision_threshold(THRESHOLD)
    y_pred = final_model.predict(X_test)
    return evaluate_predictions(y_test, y_pred)


def aggregate_seed_results(seed_results):
    """Compute mean ± std across seed runs."""
    metrics = {}
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score',
                   'missed_patients', 'total_positive']
    for key in metric_keys:
        values = [r[key] for r in seed_results]
        metrics[f'{key}_mean'] = float(np.mean(values))
        metrics[f'{key}_std'] = float(np.std(values))
    metrics['per_seed'] = seed_results
    return metrics


def print_result_row(label, result):
    """Pretty-print a result row."""
    acc = result['accuracy_mean']
    acc_std = result['accuracy_std']
    rec = result['recall_mean']
    rec_std = result['recall_std']
    f1 = result['f1_score_mean']
    f1_std = result['f1_score_std']
    missed = result['missed_patients_mean']
    missed_std = result['missed_patients_std']
    safe = "✅" if rec >= 0.80 else "❌"
    print(f"  {label:<40s}  "
          f"Acc={acc:6.2%}±{acc_std:.2%}  "
          f"Rec={rec:6.2%}±{rec_std:.2%}  "
          f"F1={f1:6.2%}±{f1_std:.2%}  "
          f"Missed={missed:.1f}±{missed_std:.1f}  {safe}")


def main():
    print("\n" + "=" * 110)
    print("MASTER VERIFICATION EXPERIMENT — Paper Table 1")
    print("=" * 110)
    print(f"Seeds: {SEEDS}")
    print(f"Threshold: {THRESHOLD}, Clients: {N_CLIENTS}, Rounds: {N_ROUNDS}, Alpha: {ALPHA}")
    print(f"DP Epsilons: {DP_EPSILON_VALUES}, Delta: {DP_DELTA}, Clipping: {DP_CLIPPING_NORM}")
    print()

    # Load data once
    X_train, X_test, y_train, y_test, feat_names = load_and_preprocess()
    print(f"\nData: {len(X_train)} train, {len(X_test)} test, {len(feat_names)} features")
    print(f"Test positives: {y_test.sum()}, Test negatives: {(y_test == 0).sum()}")

    all_results = {}

    # =================================================================
    # CONFIG A: Centralized Baseline
    # =================================================================
    print("\n" + "-" * 110)
    print("Config A: Centralized Baseline (threshold=0.30, feature eng)")
    print("-" * 110)
    seed_results = []
    for seed in SEEDS:
        r = run_centralized(X_train, X_test, y_train, y_test, seed)
        seed_results.append(r)
        print(f"    seed={seed}: Acc={r['accuracy']:.4f}, Rec={r['recall']:.4f}, Missed={r['missed_patients']}")
    all_results['centralized'] = aggregate_seed_results(seed_results)
    print_result_row("→ MEAN: Centralized", all_results['centralized'])

    # =================================================================
    # CONFIG B: FL Baseline (FedAvg, No DP)
    # =================================================================
    print("\n" + "-" * 110)
    print("Config B: FL Baseline (FedAvg, No DP)")
    print("-" * 110)
    seed_results = []
    for seed in SEEDS:
        r = run_fl(X_train, X_test, y_train, y_test, seed,
                   dp_epsilon=None, aggregation='fedavg')
        seed_results.append(r)
        print(f"    seed={seed}: Acc={r['accuracy']:.4f}, Rec={r['recall']:.4f}, Missed={r['missed_patients']}")
    all_results['fl_fedavg_no_dp'] = aggregate_seed_results(seed_results)
    print_result_row("→ MEAN: FL (FedAvg, No DP)", all_results['fl_fedavg_no_dp'])

    # =================================================================
    # CONFIG C: FL + DP-SGD (privacy-utility curve)
    # =================================================================
    print("\n" + "-" * 110)
    print("Config C: FL + DP-SGD (Privacy-Utility Curve)")
    print("-" * 110)
    for eps in DP_EPSILON_VALUES:
        seed_results = []
        for seed in SEEDS:
            r = run_fl(X_train, X_test, y_train, y_test, seed,
                       dp_epsilon=eps, aggregation='fedavg')
            seed_results.append(r)
        key = f'fl_dp_eps{eps}'
        all_results[key] = aggregate_seed_results(seed_results)
        print_result_row(f"FL + DP (ε={eps})", all_results[key])

    # =================================================================
    # CONFIG D: FL + DP-SGD + Krum
    # =================================================================
    print("\n" + "-" * 110)
    print("Config D: FL + DP-SGD (ε=1.0) + Krum")
    print("-" * 110)
    seed_results = []
    for seed in SEEDS:
        r = run_fl(X_train, X_test, y_train, y_test, seed,
                   dp_epsilon=1.0, aggregation='krum')
        seed_results.append(r)
        print(f"    seed={seed}: Acc={r['accuracy']:.4f}, Rec={r['recall']:.4f}, Missed={r['missed_patients']}")
    all_results['fl_dp_krum'] = aggregate_seed_results(seed_results)
    print_result_row("→ MEAN: FL + DP + Krum", all_results['fl_dp_krum'])

    # =================================================================
    # CONFIG E: FL + DP-SGD + Median
    # =================================================================
    print("\n" + "-" * 110)
    print("Config E: FL + DP-SGD (ε=1.0) + Median")
    print("-" * 110)
    seed_results = []
    for seed in SEEDS:
        r = run_fl(X_train, X_test, y_train, y_test, seed,
                   dp_epsilon=1.0, aggregation='median')
        seed_results.append(r)
        print(f"    seed={seed}: Acc={r['accuracy']:.4f}, Rec={r['recall']:.4f}, Missed={r['missed_patients']}")
    all_results['fl_dp_median'] = aggregate_seed_results(seed_results)
    print_result_row("→ MEAN: FL + DP + Median", all_results['fl_dp_median'])

    # =================================================================
    # SUMMARY TABLE
    # =================================================================
    print("\n" + "=" * 110)
    print("VERIFIED TABLE 1 — FOR PAPER")
    print("=" * 110)
    print()
    header = f"  {'Configuration':<40s}  {'Accuracy':<18s}  {'Recall':<18s}  {'F1':<18s}  {'Missed':<12s}"
    print(header)
    print("  " + "-" * 108)

    table_rows = [
        ('Centralized Baseline', 'centralized'),
        ('FL Baseline (FedAvg)', 'fl_fedavg_no_dp'),
    ]
    for eps in DP_EPSILON_VALUES:
        table_rows.append((f'FL + DP-SGD (ε={eps})', f'fl_dp_eps{eps}'))
    table_rows.append(('FL + DP (ε=1.0) + Krum', 'fl_dp_krum'))
    table_rows.append(('FL + DP (ε=1.0) + Median', 'fl_dp_median'))

    for label, key in table_rows:
        r = all_results[key]
        acc_s = f"{r['accuracy_mean']:.2%} ± {r['accuracy_std']:.2%}"
        rec_s = f"{r['recall_mean']:.2%} ± {r['recall_std']:.2%}"
        f1_s = f"{r['f1_score_mean']:.2%} ± {r['f1_score_std']:.2%}"
        miss_s = f"{r['missed_patients_mean']:.1f} ± {r['missed_patients_std']:.1f}"
        print(f"  {label:<40s}  {acc_s:<18s}  {rec_s:<18s}  {f1_s:<18s}  {miss_s:<12s}")

    # =================================================================
    # SAVE RESULTS
    # =================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(__file__).parent.parent / 'results' / f'paper_table1_verified_{timestamp}.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        'metadata': {
            'experiment': 'paper_table1_verification',
            'timestamp': datetime.now().isoformat(),
            'seeds': SEEDS,
            'threshold': THRESHOLD,
            'n_clients': N_CLIENTS,
            'n_rounds': N_ROUNDS,
            'alpha': ALPHA,
            'dp_delta': DP_DELTA,
            'dp_clipping_norm': DP_CLIPPING_NORM,
            'dp_epsilon_values': DP_EPSILON_VALUES,
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'features': int(len(feat_names)),
        },
        'results': {}
    }
    for key, val in all_results.items():
        save_data['results'][key] = {
            k: v for k, v in val.items()
        }

    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")
    print("=" * 110)

    return all_results


if __name__ == '__main__':
    results = main()
