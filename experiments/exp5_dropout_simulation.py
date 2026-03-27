"""
Experiment 5: Client Dropout Simulation

Tests federated learning robustness when clients drop out during training.
Realistic scenario: hospitals may go offline or disconnect mid-training.

Key questions:
1. How does FL perform with 5%, 10%, 20%, 30% client dropout?
2. At what dropout rate does performance degrade below 80% recall?
3. Is FedAvg robust to dropout, or does it need modifications?
4. How many minimum clients are needed to maintain safety?
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
from src.fl.strategy import FedAvgAggregator, aggregate_metrics


def run_dropout_experiment():
    """Test FL robustness to client dropout."""
    
    print("\n" + "=" * 100)
    print("EXPERIMENT 5: CLIENT DROPOUT SIMULATION")
    print("=" * 100 + "\n")
    
    # Configuration
    num_clients = 10
    num_rounds = 10
    alpha = 0.5
    dropout_rates = [0.0, 0.05, 0.10, 0.20, 0.30]  # 0%, 5%, 10%, 20%, 30%
    random_seed = 42
    
    print(f"📊 Configuration:")
    print(f"   Total clients: {num_clients}")
    print(f"   Federated rounds: {num_rounds}")
    print(f"   Non-IID alpha: {alpha}")
    print(f"   Dropout rates to test: {dropout_rates}")
    print(f"   Random seed: {random_seed}")
    
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
    np.random.seed(random_seed)
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
            'n_samples': len(X_client),
            'dropout': False,  # Track dropout status
        })
    
    print(f"Client distribution created: {num_clients} clients")
    
    # =====================================================================
    # DROPOUT EXPERIMENTS
    # =====================================================================
    dropout_results = {}
    
    for dropout_rate in dropout_rates:
        print("\n" + "=" * 100)
        print(f"TESTING DROPOUT RATE: {dropout_rate:.0%}")
        print("=" * 100 + "\n")
        
        # Reset dropout status
        for client in client_data:
            client['dropout'] = False
        
        # Randomly select clients to drop out
        num_dropouts = int(num_clients * dropout_rate)
        if num_dropouts > 0:
            dropout_indices = np.random.choice(num_clients, num_dropouts, replace=False)
            for idx in dropout_indices:
                client_data[idx]['dropout'] = True
        
        active_clients = [c for c in client_data if not c['dropout']]
        print(f"Active clients: {len(active_clients)}/{num_clients}")
        print(f"Dropped out: {num_dropouts} clients")
        
        if len(active_clients) == 0:
            print("⚠️ No active clients - skipping this dropout rate")
            continue
        
        # Initialize global model
        init_model = LogisticRegressionModel(max_iter=2000, class_weight='balanced')
        init_model.set_decision_threshold(0.30)
        init_model.fit(active_clients[0]['X_train'], active_clients[0]['y_train'], verbose=False)
        global_weights = init_model.get_weights()
        
        # Track metrics
        round_metrics = {
            'accuracy': [],
            'recall': [],
            'f1': [],
        }
        
        start_time = time.time()
        
        # Federated learning rounds
        for round_num in range(num_rounds):
            client_weights = []
            client_sample_sizes = []
            client_eval_metrics = []
            
            # Local training on active clients only
            for client in active_clients:
                # Create local model
                local_model = LogisticRegressionModel(max_iter=2000, class_weight='balanced')
                local_model.set_weights(global_weights)
                local_model.set_decision_threshold(0.30)
                
                # Local training
                try:
                    local_model.fit(client['X_train'], client['y_train'], verbose=False)
                    client_weights.append(local_model.get_weights())
                except ValueError:
                    # Single-class client
                    client_weights.append(global_weights)
                
                client_sample_sizes.append(client['n_samples'])
                
                # Evaluate
                y_pred = local_model.predict(client['X_test'])
                metrics = {
                    'accuracy': float(accuracy_score(client['y_test'], y_pred)),
                    'recall': float(recall_score(client['y_test'], y_pred, zero_division=0)),
                    'f1_score': float(f1_score(client['y_test'], y_pred, zero_division=0)),
                }
                client_eval_metrics.append(metrics)
            
            # Aggregate
            global_weights = FedAvgAggregator.aggregate(client_weights, client_sample_sizes)
            agg_metrics = aggregate_metrics(client_eval_metrics, client_sample_sizes)
            
            # Track metrics
            round_metrics['accuracy'].append(agg_metrics['accuracy'])
            round_metrics['recall'].append(agg_metrics['recall'])
            round_metrics['f1'].append(agg_metrics['f1_score'])
            
            if (round_num + 1) % 5 == 0 or round_num == 0:
                print(f"Round {round_num + 1:2d}: Accuracy={agg_metrics['accuracy']:.2%}, " +
                      f"Recall={agg_metrics['recall']:.2%}")
        
        elapsed_time = time.time() - start_time
        
        # Final evaluation
        final_model = LogisticRegressionModel(max_iter=2000, class_weight='balanced')
        final_model.set_weights(global_weights)
        final_model.set_decision_threshold(0.30)
        y_pred = final_model.predict(X_test_eng)
        
        results = {
            'dropout_rate': dropout_rate,
            'active_clients': len(active_clients),
            'total_clients': num_clients,
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'missed_patients': int((y_test == 1).sum() - (y_pred * y_test).sum()),
            'total_patients': int((y_test == 1).sum()),
            'training_time': elapsed_time,
            'convergence_metrics': round_metrics,
        }
        
        dropout_results[f"{dropout_rate:.0%}"] = results
        
        print(f"\nResults for {dropout_rate:.0%} dropout:")
        print(f"  Active: {len(active_clients)}/{num_clients}")
        print(f"  Accuracy:  {results['accuracy']:.2%}")
        print(f"  Recall:    {results['recall']:.2%} {'✅' if results['recall'] >= 0.80 else '❌'}")
        print(f"  Time:      {elapsed_time:.2f}s")
    
    # =====================================================================
    # ANALYSIS
    # =====================================================================
    print("\n" + "=" * 100)
    print("DROPOUT IMPACT ANALYSIS")
    print("=" * 100 + "\n")
    
    print(f"{'Dropout':<10} {'Clients':<10} {'Accuracy':<12} {'Recall':<12} {'Safety':<10} {'Time':<8}")
    print("-" * 60)
    
    for rate_str, results in dropout_results.items():
        safety = '✅' if results['recall'] >= 0.80 else '❌'
        print(f"{rate_str:<10} {results['active_clients']}/{results['total_clients']:<8} " +
              f"{results['accuracy']:>10.2%}  {results['recall']:>10.2%}  {safety:<10} {results['training_time']:>6.2f}s")
    
    # =====================================================================
    # KEY FINDINGS
    # =====================================================================
    print("\n" + "=" * 100)
    print("KEY FINDINGS")
    print("=" * 100 + "\n")
    
    safe_rates = [rate for rate, res in dropout_results.items() if res['recall'] >= 0.80]
    unsafe_rates = [rate for rate, res in dropout_results.items() if res['recall'] < 0.80]
    
    if safe_rates:
        print(f"✅ Safe dropout rates (recall ≥ 80%):")
        for rate in safe_rates:
            res = dropout_results[rate]
            print(f"   {rate}: {res['recall']:.2%} recall ({res['active_clients']} active clients)")
    
    if unsafe_rates:
        print(f"\n❌ Unsafe dropout rates (recall < 80%):")
        for rate in unsafe_rates:
            res = dropout_results[rate]
            print(f"   {rate}: {res['recall']:.2%} recall ({res['active_clients']} active clients)")
    
    # Find maximum safe dropout rate
    if dropout_results:
        sorted_rates = sorted(dropout_results.items(), 
                            key=lambda x: float(x[0].rstrip('%')) / 100)
        
        max_safe_rate = None
        for rate_str, results in sorted_rates:
            if results['recall'] >= 0.80:
                max_safe_rate = rate_str
        
        if max_safe_rate:
            max_res = dropout_results[max_safe_rate]
            print(f"\n🎯 Maximum safe dropout rate: {max_safe_rate}")
            print(f"   Active clients: {max_res['active_clients']}/{num_clients}")
            print(f"   Recall: {max_res['recall']:.2%}")
        else:
            print(f"\n⚠️ No dropout rate maintains 80% recall with {num_clients} clients")
            print(f"   Recommendation: Use more clients or adjust configuration")
    
    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    print("\n" + "=" * 100)
    print("SAVING RESULTS")
    print("=" * 100 + "\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"results/dropout_simulation_{timestamp}.json")
    
    all_results = {
        'configuration': {
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'alpha': alpha,
            'dropout_rates': dropout_rates,
        },
        'dropout_results': dropout_results,
        'analysis': {
            'safe_rates': safe_rates,
            'unsafe_rates': unsafe_rates,
            'max_safe_rate': max_safe_rate if 'max_safe_rate' in locals() else None,
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    results = run_dropout_experiment()
    print("\n✅ Experiment complete!")
