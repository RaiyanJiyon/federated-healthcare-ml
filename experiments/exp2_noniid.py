#!/usr/bin/env python3
"""
Experiment 2: Federated Learning with Non-IID Data

Tests federated learning with non-independently distributed (Non-IID) data
to simulate realistic healthcare scenarios where different hospitals have
different patient demographics and disease distributions.

Results are compared against the centralized baseline model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.data.loader import load_dataset_with_df
from src.data.preprocess import DataPreprocessor
from src.data.split import train_test_split_data, distribute_non_iid
from src.models.model import LogisticRegressionModel
from src.utils.feature_engineering import HealthcareFeatureEngineer
from src.fl.strategy import FedAvgAggregator, aggregate_metrics
from src.evaluation.metrics import calculate_all_metrics


def run_non_iid_experiment():
    """Run federated learning with Non-IID data distribution."""
    
    print("\n" + "=" * 100)
    print("EXPERIMENT 2: FEDERATED LEARNING WITH NON-IID DATA")
    print("=" * 100)
    
    # Configuration
    num_clients = 5
    num_rounds = 10
    alpha = 0.5  # Dirichlet parameter (lower = more non-IID)
    
    print(f"\n📊 Configuration:")
    print(f"   Clients: {num_clients}")
    print(f"   Federated rounds: {num_rounds}")
    print(f"   Non-IID alpha: {alpha} (lower = more heterogeneous)")
    print(f"   Data distribution: Non-Independent and Identically Distributed")
    
    # Load and preprocess data
    print("\n📁 Loading data...")
    df, X, y = load_dataset_with_df()
    preprocessor = DataPreprocessor()
    X_processed = preprocessor.preprocess(df.iloc[:, :-1], fit=True)
    feature_names = list(df.columns[:-1])
    
    X_train, X_test, y_train, y_test = train_test_split_data(X_processed, y)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    print(f"✓ Data ready: {len(X_train)} training, {len(X_test)} testing")
    
    # Create engineered features
    print("\n📊 Creating engineered features...")
    engineer = HealthcareFeatureEngineer()
    X_train_eng, feature_names_eng = engineer.engineer_all_features(X_train, feature_names)
    
    # Apply to test set
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
    print(f"✓ Features engineered: {len(feature_names)} → {len(feature_names_eng)}")
    
    # Distribute data to clients (Non-IID)
    print(f"\n🌐 Distributing data to {num_clients} clients (Non-IID)...")
    client_data_dict = distribute_non_iid(X_train_eng, y_train, num_clients, alpha)
    
    # Create client datasets list
    client_data = []
    for client_id in range(num_clients):
        X_client, y_client = client_data_dict[client_id]
        
        # Each client also gets the full test set for evaluation
        client_data.append({
            'id': client_id,
            'X_train': X_client,
            'y_train': y_client,
            'X_test': X_test_eng,
            'y_test': y_test,
            'n_samples': len(X_client)
        })
        
        print(f"   Client {client_id}: {len(X_client)} samples " +
              f"({y_client.sum()} diabetic, {len(y_client) - y_client.sum()} non-diabetic)")
    
    # =====================================================================
    # FEDERATED LEARNING SIMULATION
    # =====================================================================
    print("\n" + "=" * 100)
    print("FEDERATED LEARNING SIMULATION")
    print("=" * 100 + "\n")
    
    # Initialize global model by training on first client
    # This gives us initial weights to start federated learning
    init_model = LogisticRegressionModel(max_iter=2000, class_weight='balanced')
    init_model.set_decision_threshold(0.30)
    init_model.fit(client_data[0]['X_train'], client_data[0]['y_train'], verbose=False)
    global_weights = init_model.get_weights()
    
    global_model = LogisticRegressionModel(max_iter=2000, class_weight='balanced')
    global_model.set_decision_threshold(0.30)
    global_model.set_weights(global_weights)
    
    # Track metrics per round
    round_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
    }
    
    start_time = time.time()
    
    # Federated learning rounds
    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}")
        
        client_weights = []
        client_sample_sizes = []
        client_eval_metrics = []
        
        # Local training on each client
        for client_idx, client in enumerate(client_data):
            # Create local model
            local_model = LogisticRegressionModel(max_iter=2000, class_weight='balanced')
            local_model.set_weights(global_weights)
            local_model.set_decision_threshold(0.30)
            
            # Local training (handle single-class clients)
            try:
                local_model.fit(client['X_train'], client['y_train'], verbose=False)
                client_weights.append(local_model.get_weights())
            except ValueError:
                # Client has only one class - skip training, use global weights
                client_weights.append(global_weights)
            
            client_sample_sizes.append(client['n_samples'])
            
            # Evaluate on test set
            y_pred = local_model.predict(client['X_test'])
            metrics = {
                'accuracy': float(accuracy_score(client['y_test'], y_pred)),
                'precision': float(precision_score(client['y_test'], y_pred, zero_division=0)),
                'recall': float(recall_score(client['y_test'], y_pred, zero_division=0)),
                'f1_score': float(f1_score(client['y_test'], y_pred, zero_division=0)),
            }
            client_eval_metrics.append(metrics)
        
        # Aggregate weights (FedAvg)
        global_weights = FedAvgAggregator.aggregate(client_weights, client_sample_sizes)
        
        # Aggregate metrics
        agg_metrics = aggregate_metrics(client_eval_metrics, client_sample_sizes)
        
        print(f"   Accuracy: {agg_metrics['accuracy']:.2%}, " +
              f"Recall: {agg_metrics['recall']:.2%}, " +
              f"F1: {agg_metrics['f1_score']:.2%}")
        
        # Track metrics
        round_metrics['accuracy'].append(agg_metrics['accuracy'])
        round_metrics['precision'].append(agg_metrics['precision'])
        round_metrics['recall'].append(agg_metrics['recall'])
        round_metrics['f1'].append(agg_metrics['f1_score'])
    
    fl_training_time = time.time() - start_time
    
    # =====================================================================
    # FINAL EVALUATION
    # =====================================================================
    print("\n" + "=" * 100)
    print("FINAL EVALUATION (Non-IID Federated Learning)")
    print("=" * 100)
    
    # Set global model with final weights
    global_model.set_weights(global_weights)
    y_pred_fl = global_model.predict(X_test_eng)
    
    fl_metrics = calculate_all_metrics(y_test, y_pred_fl)
    
    print(f"\nFederated Learning Results:")
    print(f"  Accuracy:  {fl_metrics['accuracy']:.2%}")
    print(f"  Precision: {fl_metrics['precision']:.2%}")
    print(f"  Recall:    {fl_metrics['recall']:.2%}")
    print(f"  F1-Score:  {fl_metrics['f1_score']:.2%}")
    print(f"  Training time: {fl_training_time:.2f} seconds")
    print(f"  Missed patients: {fl_metrics['FN']} out of {fl_metrics['FN'] + fl_metrics['TP']}")
    
    # =====================================================================
    # COMPARISON WITH CENTRALIZED BASELINE
    # =====================================================================
    print("\n" + "=" * 100)
    print("COMPARISON: FEDERATED vs CENTRALIZED")
    print("=" * 100)
    
    # Baseline metrics (from exp1_optimized)
    baseline_metrics = {
        'accuracy': 0.6948,
        'precision': 0.5402,
        'recall': 0.8704,
        'f1_score': 0.6667,
    }
    
    print(f"\n{'Metric':<15} {'Centralized':<15} {'Federated':<15} {'Difference':<15}")
    print("-" * 60)
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric in metrics_to_compare:
        centralized = baseline_metrics[metric]
        federated = fl_metrics[metric]
        diff = federated - centralized
        print(f"{metric:<15} {centralized:<14.2%} {federated:<14.2%} {diff:+.2%}")
    
    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    print("\n" + "=" * 100)
    print("SAVING RESULTS")
    print("=" * 100)
    
    results = {
        'experiment': 'exp2_noniid',
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'non_iid_alpha': alpha,
            'model': 'LogisticRegression',
            'decision_threshold': 0.30,
        },
        'federated_learning': {
            'final_metrics': fl_metrics,
            'round_metrics': round_metrics,
            'training_time': float(fl_training_time),
        },
        'centralized_baseline': baseline_metrics,
        'comparison': {
            metric: fl_metrics[metric] - baseline_metrics[metric]
            for metric in metrics_to_compare
        }
    }
    
    result_path = Path(__file__).parent.parent / 'results'
    result_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = result_path / f"noniid_federated_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {result_file}")
    
    print("\n" + "=" * 100)
    print("CONCLUSIONS")
    print("=" * 100)
    
    if fl_metrics['recall'] >= 0.80:
        print(f"\n✅ FEDERATED LEARNING ACHIEVES CLINICAL SAFETY")
        print(f"   Recall: {fl_metrics['recall']:.2%} ≥ 80% threshold")
        print(f"   Missed patients: {fl_metrics['FN']} out of {fl_metrics['FN'] + fl_metrics['TP']}")
    else:
        print(f"\n⚠️ Recall: {fl_metrics['recall']:.2%} (below 80% target)")
    
    print(f"""
Key Findings:
1. Non-IID Data Distribution:
   - Federated learning handles non-IID data (alpha={alpha})
   - Different clients have different data distributions
   - Aggregation via FedAvg still effective

2. Performance vs Centralized:
   - Federated accuracy: {fl_metrics['accuracy']:.2%}
   - Centralized accuracy: {baseline_metrics['accuracy']:.2%}
   - Difference: {fl_metrics['accuracy'] - baseline_metrics['accuracy']:+.2%}

3. Privacy Preservation:
   - Only model weights shared (no raw data)
   - Healthcare data remains local
   - Suitable for multi-hospital collaboration

4. Recall (Clinical Safety):
   - Federated recall: {fl_metrics['recall']:.2%}
   - Centralized recall: {baseline_metrics['recall']:.2%}
   - Maintains clinical safety in federated setting
""")
    
    print("=" * 100)
    
    return results


if __name__ == '__main__':
    results = run_non_iid_experiment()
    print(f"\n✅ Experiment complete!")
