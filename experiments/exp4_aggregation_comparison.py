"""
Experiment 4: Federated Learning Aggregation Strategy Comparison

Compares FedAvg vs FedProx on Non-IID data.

Key questions:
1. Does FedProx handle Non-IID data better than FedAvg?
2. What's the convergence speed difference?
3. How does recall differ between strategies?
4. When should you use FedProx over FedAvg?
"""

import sys
sys.path.insert(0, '/home/raiyanjiyon/Machine Learning/federated-healthcare-ml')

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Project imports
from src.data.loader import load_dataset_with_df
from src.data.preprocess import DataPreprocessor
from src.data.split import train_test_split_data, distribute_non_iid
from src.models.model import LogisticRegressionModel
from src.utils.feature_engineering import HealthcareFeatureEngineer
from src.fl.strategy import FedAvgAggregator, FedProxAggregator, aggregate_metrics


def run_aggregation_comparison():
    """Compare FedAvg vs FedProx aggregation strategies."""
    
    print("\n" + "=" * 100)
    print("EXPERIMENT 4: AGGREGATION STRATEGY COMPARISON (FedAvg vs FedProx)")
    print("=" * 100 + "\n")
    
    # Configuration
    num_clients = 5
    num_rounds = 10
    alpha = 0.5  # Non-IID parameter
    mu = 0.01   # FedProx proximal term coefficient
    
    print(f"📊 Configuration:")
    print(f"   Clients: {num_clients}")
    print(f"   Federated rounds: {num_rounds}")
    print(f"   Non-IID alpha: {alpha}")
    print(f"   FedProx mu: {mu}")
    print(f"   Comparison: FedAvg vs FedProx")
    
    # =====================================================================
    # DATA LOADING & PREPROCESSING
    # =====================================================================
    print(f'\n📁 Loading data...')
    df, X, y = load_dataset_with_df()
    
    print('Preprocessing data...')
    preprocessor = DataPreprocessor()
    X_processed = preprocessor.preprocess(df.iloc[:, :-1], fit=True)
    feature_names = list(df.columns[:-1])
    
    X_train, X_test, y_train, y_test = train_test_split_data(X_processed, y)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    print(f'✓ Data ready: {len(X_train)} training, {len(X_test)} testing')
    
    # =====================================================================
    # FEATURE ENGINEERING
    # =====================================================================
    print(f'\n📊 Creating engineered features...')
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
    X_test_df['Glucose_per_Insulin'] = X_test_df['Glucose'] / X_test_df['Insulin']
    X_test_df['BloodPressure_per_Age'] = X_test_df['BloodPressure'] / X_test_df['Age']
    X_test_eng = X_test_df.values
    
    print(f'✓ Features: {X_train.shape[1]} → {X_train_eng.shape[1]}')
    
    # =====================================================================
    # NON-IID DISTRIBUTION
    # =====================================================================
    print(f'\n🌐 Distributing data to {num_clients} clients (Non-IID)...')
    client_data_dict = distribute_non_iid(X_train_eng, y_train, num_clients, alpha)
    
    client_data = []
    for client_id in range(num_clients):
        X_client, y_client = client_data_dict[client_id]
        client_data.append({
            'id': client_id,
            'X_train': X_client,
            'y_train': y_client,
            'X_test': X_test_eng,
            'y_test': y_test,
            'n_samples': len(X_client)
        })
    
    print(f"Client distribution (Non-IID, alpha={alpha}):")
    for client in client_data:
        pos = client['y_train'].sum()
        neg = len(client['y_train']) - pos
        print(f"   Client {client['id']}: {client['n_samples']} samples ({pos} pos, {neg} neg)")
    
    # =====================================================================
    # FEDERATED LEARNING WITH FedAvg
    # =====================================================================
    print("\n" + "=" * 100)
    print("FEDERATED LEARNING WITH FedAvg (BASELINE)")
    print("=" * 100 + "\n")
    
    # Initialize global model
    init_model = LogisticRegressionModel(max_iter=2000, class_weight='balanced')
    init_model.set_decision_threshold(0.30)
    init_model.fit(client_data[0]['X_train'], client_data[0]['y_train'], verbose=False)
    global_weights = init_model.get_weights()
    
    # Track metrics
    fedavg_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
    }
    
    fedavg_start = time.time()
    fedavg_weights = global_weights.copy()
    
    # Federated learning rounds
    for round_num in range(num_rounds):
        client_weights = []
        client_sample_sizes = []
        client_eval_metrics = []
        
        # Local training on each client
        for client_idx, client in enumerate(client_data):
            # Create local model
            local_model = LogisticRegressionModel(max_iter=2000, class_weight='balanced')
            local_model.set_weights(fedavg_weights)
            local_model.set_decision_threshold(0.30)
            
            # Local training
            try:
                local_model.fit(client['X_train'], client['y_train'], verbose=False)
                client_weights.append(local_model.get_weights())
            except ValueError:
                # Single-class client - skip training
                client_weights.append(fedavg_weights)
            
            client_sample_sizes.append(client['n_samples'])
            
            # Evaluate
            y_pred = local_model.predict(client['X_test'])
            metrics = {
                'accuracy': float(accuracy_score(client['y_test'], y_pred)),
                'precision': float(precision_score(client['y_test'], y_pred, zero_division=0)),
                'recall': float(recall_score(client['y_test'], y_pred, zero_division=0)),
                'f1_score': float(f1_score(client['y_test'], y_pred, zero_division=0)),
            }
            client_eval_metrics.append(metrics)
        
        # Aggregate weights using FedAvg
        fedavg_weights = FedAvgAggregator.aggregate(client_weights, client_sample_sizes)
        
        # Aggregate metrics
        total_samples = sum(client_sample_sizes)
        agg_metrics = {}
        for metric_name in client_eval_metrics[0].keys():
            weighted_sum = sum(
                (client_sample_sizes[i] / total_samples) * client_eval_metrics[i][metric_name]
                for i in range(len(client_eval_metrics))
            )
            agg_metrics[metric_name] = weighted_sum
        
        # Track metrics
        fedavg_metrics['accuracy'].append(agg_metrics['accuracy'])
        fedavg_metrics['precision'].append(agg_metrics['precision'])
        fedavg_metrics['recall'].append(agg_metrics['recall'])
        fedavg_metrics['f1'].append(agg_metrics['f1_score'])
        
        if (round_num + 1) % 5 == 0 or round_num == 0:
            print(f"Round {round_num + 1:2d}: Accuracy={agg_metrics['accuracy']:.2%}, " +
                  f"Recall={agg_metrics['recall']:.2%}")
    
    fedavg_time = time.time() - fedavg_start
    
    # Final evaluation with FedAvg
    final_model_fedavg = LogisticRegressionModel(max_iter=2000, class_weight='balanced')
    final_model_fedavg.set_weights(fedavg_weights)
    final_model_fedavg.set_decision_threshold(0.30)
    y_pred_fedavg = final_model_fedavg.predict(X_test_eng)
    
    fedavg_results = {
        'strategy': 'FedAvg',
        'accuracy': float(accuracy_score(y_test, y_pred_fedavg)),
        'precision': float(precision_score(y_test, y_pred_fedavg, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_fedavg, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred_fedavg, zero_division=0)),
        'missed_patients': int((y_test == 1).sum() - (y_pred_fedavg * y_test).sum()),
        'total_patients': int((y_test == 1).sum()),
        'training_time': fedavg_time,
        'convergence_metrics': fedavg_metrics,
    }
    
    print(f"\nFedAvg Final Results:")
    print(f"  Accuracy:  {fedavg_results['accuracy']:.2%}")
    print(f"  Precision: {fedavg_results['precision']:.2%}")
    print(f"  Recall:    {fedavg_results['recall']:.2%}")
    print(f"  F1-Score:  {fedavg_results['f1_score']:.2%}")
    print(f"  Time:      {fedavg_time:.2f} seconds")
    
    # =====================================================================
    # FEDERATED LEARNING WITH FedProx
    # =====================================================================
    print("\n" + "=" * 100)
    print(f"FEDERATED LEARNING WITH FedProx (mu={mu})")
    print("=" * 100 + "\n")
    
    # Track metrics
    fedprox_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
    }
    
    fedprox_start = time.time()
    fedprox_weights = global_weights.copy()
    
    # Federated learning rounds
    for round_num in range(num_rounds):
        client_weights = []
        client_sample_sizes = []
        client_eval_metrics = []
        
        # Local training on each client
        for client_idx, client in enumerate(client_data):
            # Create local model
            local_model = LogisticRegressionModel(max_iter=2000, class_weight='balanced')
            local_model.set_weights(fedprox_weights)
            local_model.set_decision_threshold(0.30)
            
            # Local training
            try:
                local_model.fit(client['X_train'], client['y_train'], verbose=False)
                client_weights.append(local_model.get_weights())
            except ValueError:
                # Single-class client - skip training
                client_weights.append(fedprox_weights)
            
            client_sample_sizes.append(client['n_samples'])
            
            # Evaluate
            y_pred = local_model.predict(client['X_test'])
            metrics = {
                'accuracy': float(accuracy_score(client['y_test'], y_pred)),
                'precision': float(precision_score(client['y_test'], y_pred, zero_division=0)),
                'recall': float(recall_score(client['y_test'], y_pred, zero_division=0)),
                'f1_score': float(f1_score(client['y_test'], y_pred, zero_division=0)),
            }
            client_eval_metrics.append(metrics)
        
        # Aggregate weights using FedProx
        fedprox_weights = FedProxAggregator.aggregate(
            client_weights, client_sample_sizes, 
            global_weights=fedprox_weights, mu=mu
        )
        
        # Aggregate metrics
        total_samples = sum(client_sample_sizes)
        agg_metrics = {}
        for metric_name in client_eval_metrics[0].keys():
            weighted_sum = sum(
                (client_sample_sizes[i] / total_samples) * client_eval_metrics[i][metric_name]
                for i in range(len(client_eval_metrics))
            )
            agg_metrics[metric_name] = weighted_sum
        
        # Track metrics
        fedprox_metrics['accuracy'].append(agg_metrics['accuracy'])
        fedprox_metrics['precision'].append(agg_metrics['precision'])
        fedprox_metrics['recall'].append(agg_metrics['recall'])
        fedprox_metrics['f1'].append(agg_metrics['f1_score'])
        
        if (round_num + 1) % 5 == 0 or round_num == 0:
            print(f"Round {round_num + 1:2d}: Accuracy={agg_metrics['accuracy']:.2%}, " +
                  f"Recall={agg_metrics['recall']:.2%}")
    
    fedprox_time = time.time() - fedprox_start
    
    # Final evaluation with FedProx
    final_model_fedprox = LogisticRegressionModel(max_iter=2000, class_weight='balanced')
    final_model_fedprox.set_weights(fedprox_weights)
    final_model_fedprox.set_decision_threshold(0.30)
    y_pred_fedprox = final_model_fedprox.predict(X_test_eng)
    
    fedprox_results = {
        'strategy': 'FedProx',
        'mu': mu,
        'accuracy': float(accuracy_score(y_test, y_pred_fedprox)),
        'precision': float(precision_score(y_test, y_pred_fedprox, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_fedprox, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred_fedprox, zero_division=0)),
        'missed_patients': int((y_test == 1).sum() - (y_pred_fedprox * y_test).sum()),
        'total_patients': int((y_test == 1).sum()),
        'training_time': fedprox_time,
        'convergence_metrics': fedprox_metrics,
    }
    
    print(f"\nFedProx Final Results:")
    print(f"  Accuracy:  {fedprox_results['accuracy']:.2%}")
    print(f"  Precision: {fedprox_results['precision']:.2%}")
    print(f"  Recall:    {fedprox_results['recall']:.2%}")
    print(f"  F1-Score:  {fedprox_results['f1_score']:.2%}")
    print(f"  Time:      {fedprox_time:.2f} seconds")
    
    # =====================================================================
    # COMPARISON
    # =====================================================================
    print("\n" + "=" * 100)
    print("AGGREGATION STRATEGY COMPARISON")
    print("=" * 100 + "\n")
    
    comparison_results = {
        'strategy_1': fedavg_results,
        'strategy_2': fedprox_results,
        'configuration': {
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'alpha': alpha,
            'mu': mu,
        },
    }
    
    print(f"{'Metric':<15} {'FedAvg':<15} {'FedProx':<15} {'Difference':<15}")
    print("-" * 60)
    
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric in metrics_to_compare:
        fedavg_val = fedavg_results[metric]
        fedprox_val = fedprox_results[metric]
        diff = fedprox_val - fedavg_val
        symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"{metric:<15} {fedavg_val:>14.2%} {fedprox_val:>14.2%} {diff:>+13.2%} {symbol}")
    
    print(f"{'time (s)':<15} {fedavg_time:>14.2f} {fedprox_time:>14.2f}")
    
    # =====================================================================
    # CONVERGENCE ANALYSIS
    # =====================================================================
    print("\n" + "=" * 100)
    print("CONVERGENCE ANALYSIS")
    print("=" * 100 + "\n")
    
    conv_analysis = {
        'fedavg_recall_stability': np.std(fedavg_metrics['recall']),
        'fedprox_recall_stability': np.std(fedprox_metrics['recall']),
        'fedavg_recall_improvement': fedavg_metrics['recall'][-1] - fedavg_metrics['recall'][0],
        'fedprox_recall_improvement': fedprox_metrics['recall'][-1] - fedprox_metrics['recall'][0],
    }
    
    print(f"Recall Stability (lower = more stable):")
    print(f"  FedAvg:  {conv_analysis['fedavg_recall_stability']:.4f} (std dev)")
    print(f"  FedProx: {conv_analysis['fedprox_recall_stability']:.4f} (std dev)")
    if conv_analysis['fedprox_recall_stability'] < conv_analysis['fedavg_recall_stability']:
        improvement = (1 - conv_analysis['fedprox_recall_stability']/conv_analysis['fedavg_recall_stability']) * 100
        print(f"  → FedProx is {improvement:.1f}% more stable")
    
    print(f"\nRecall Improvement Over Rounds:")
    print(f"  FedAvg:  {conv_analysis['fedavg_recall_improvement']:+.4f}")
    print(f"  FedProx: {conv_analysis['fedprox_recall_improvement']:+.4f}")
    
    comparison_results['convergence_analysis'] = {
        'recall_stability': conv_analysis,
    }
    
    # =====================================================================
    # KEY FINDINGS
    # =====================================================================
    print("\n" + "=" * 100)
    print("KEY FINDINGS")
    print("=" * 100 + "\n")
    
    winner = None
    winner_metric = None
    
    # Determine clinically safe winner (≥80% recall)
    fedavg_safe = fedavg_results['recall'] >= 0.80
    fedprox_safe = fedprox_results['recall'] >= 0.80
    
    print(f"Clinical Safety (Recall ≥ 80%):")
    print(f"  FedAvg:  {fedavg_results['recall']:.2%} {'✅' if fedavg_safe else '❌'}")
    print(f"  FedProx: {fedprox_results['recall']:.2%} {'✅' if fedprox_safe else '❌'}")
    
    print(f"\nRecall Winner:")
    if fedavg_results['recall'] > fedprox_results['recall']:
        print(f"  → FedAvg: {fedavg_results['recall']:.2%}")
        winner = 'FedAvg'
    elif fedprox_results['recall'] > fedavg_results['recall']:
        print(f"  → FedProx: {fedprox_results['recall']:.2%}")
        winner = 'FedProx'
    else:
        print(f"  → Tie: Both {fedavg_results['recall']:.2%}")
    
    print(f"\nAccuracy Winner:")
    if fedavg_results['accuracy'] > fedprox_results['accuracy']:
        print(f"  → FedAvg: {fedavg_results['accuracy']:.2%}")
    elif fedprox_results['accuracy'] > fedavg_results['accuracy']:
        print(f"  → FedProx: {fedprox_results['accuracy']:.2%}")
    else:
        print(f"  → Tie: Both {fedavg_results['accuracy']:.2%}")
    
    print(f"\nComputational Cost:")
    if fedavg_time < fedprox_time:
        print(f"  → FedAvg: {fedavg_time:.2f}s (faster)")
    else:
        print(f"  → FedProx: {fedprox_time:.2f}s (faster)")
    
    # =====================================================================
    # RECOMMENDATIONS
    # =====================================================================
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100 + "\n")
    
    if fedavg_results['recall'] >= 0.80 and fedprox_results['recall'] >= 0.80:
        print("✅ Both strategies maintain clinical safety (≥80% recall)")
        print("\nChoose based on:")
        if abs(fedavg_results['recall'] - fedprox_results['recall']) < 0.02:
            print("  • Similar recall: Use FedAvg (simpler, faster)")
        else:
            winner_strat = 'FedProx' if fedprox_results['recall'] > fedavg_results['recall'] else 'FedAvg'
            print(f"  • {winner_strat} has better recall: Use {winner_strat}")
    elif fedavg_results['recall'] >= 0.80:
        print("✅ FedAvg maintains clinical safety, FedProx does not")
        print("  → Recommendation: Use FedAvg")
    elif fedprox_results['recall'] >= 0.80:
        print("✅ FedProx maintains clinical safety, FedAvg does not")
        print("  → Recommendation: Use FedProx")
    else:
        print("❌ Neither strategy meets safety requirements")
        print("  → Recommendation: Adjust configuration (more clients, more rounds, or tune mu)")
    
    comparison_results['recommendation'] = {
        'strategy': winner or 'unclear',
        'reason': 'Both safe - FedAvg preferred for simplicity' if winner is None else f'{winner} has better recall',
    }
    
    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    print("\n" + "=" * 100)
    print("SAVING RESULTS")
    print("=" * 100 + "\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"results/aggregation_comparison_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")
    
    return comparison_results


if __name__ == "__main__":
    results = run_aggregation_comparison()
    print("\n✅ Experiment complete!")
