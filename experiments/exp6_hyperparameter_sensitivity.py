"""
Experiment 6: Hyperparameter Sensitivity Analysis

Tests FL model sensitivity to hyperparameter changes.

For scikit-learn LogisticRegression, key hyperparameters:
- max_iter: Model convergence iterations (affects training stability)
- C: Regularization strength inverse (affects overfitting)
- num_rounds: Federated rounds (affects aggregation convergence)

Key questions:
1. How does max_iter affect recall and convergence?
2. How sensitive is the model to regularization (C)?
3. How many FL rounds are needed for stable convergence?
4. What's the optimal hyperparameter configuration?
"""

import sys
sys.path.insert(0, '/home/raiyanjiyon/Machine Learning/federated-healthcare-ml')

import json
import time
from datetime import datetime
from pathlib import Path
from itertools import product

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


def run_hyperparameter_sensitivity():
    """Test federated learning hyperparameter sensitivity."""
    
    print("\n" + "=" * 100)
    print("EXPERIMENT 6: HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("=" * 100 + "\n")
    
    # Configuration
    num_clients = 5
    alpha = 0.5
    
    # Hyperparameters to test
    test_configs = {
        'max_iter': [100, 500, 2000, 5000],
        'C': [0.1, 1.0, 10.0, 100.0],
        'num_rounds': [5, 10, 15, 20],
    }
    
    print(f"📊 Configuration:")
    print(f"   Base: {num_clients} clients, Non-IID (alpha={alpha})")
    print(f"   Testing:")
    for param, values in test_configs.items():
        print(f"     {param}: {values}")
    
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
            'n_samples': len(X_client),
        })
    
    print(f"✓ {num_clients} clients created")
    
    # =====================================================================
    # HYPERPARAMETER SWEEP
    # =====================================================================
    results_all = {}
    total_configs = len(test_configs['max_iter']) * len(test_configs['C']) * len(test_configs['num_rounds'])
    config_count = 0
    
    for max_iter in test_configs['max_iter']:
        for C in test_configs['C']:
            for num_rounds in test_configs['num_rounds']:
                config_count += 1
                config_name = f"max_iter={max_iter}_C={C}_rounds={num_rounds}"
                
                print(f"\n[{config_count}/{total_configs}] Testing: {config_name}")
                
                # Initialize global model
                init_model = LogisticRegressionModel(max_iter=max_iter, class_weight='balanced')
                init_model.model.C = C  # Set regularization
                init_model.set_decision_threshold(0.30)
                init_model.fit(client_data[0]['X_train'], client_data[0]['y_train'], verbose=False)
                global_weights = init_model.get_weights()
                
                start_time = time.time()
                
                # Federated learning rounds
                for round_num in range(num_rounds):
                    client_weights = []
                    client_sample_sizes = []
                    client_eval_metrics = []
                    
                    # Local training on each client
                    for client in client_data:
                        # Create local model
                        local_model = LogisticRegressionModel(max_iter=max_iter, class_weight='balanced')
                        local_model.model.C = C
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
                
                elapsed_time = time.time() - start_time
                
                # Final evaluation
                final_model = LogisticRegressionModel(max_iter=max_iter, class_weight='balanced')
                final_model.model.C = C
                final_model.set_weights(global_weights)
                final_model.set_decision_threshold(0.30)
                y_pred = final_model.predict(X_test_eng)
                
                results = {
                    'max_iter': max_iter,
                    'C': C,
                    'num_rounds': num_rounds,
                    'accuracy': float(accuracy_score(y_test, y_pred)),
                    'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                    'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                    'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
                    'training_time': elapsed_time,
                }
                
                results_all[config_name] = results
                
                print(f"  Accuracy: {results['accuracy']:.2%}, Recall: {results['recall']:.2%}, " +
                      f"Time: {elapsed_time:.2f}s")
    
    # =====================================================================
    # SENSITIVITY ANALYSIS
    # =====================================================================
    print("\n" + "=" * 100)
    print("SENSITIVITY ANALYSIS")
    print("=" * 100 + "\n")
    
    # Analyze impact of each hyperparameter
    sensitivity = {
        'max_iter_impact': {},
        'C_impact': {},
        'rounds_impact': {},
    }
    
    # max_iter impact (keeping C and rounds constant)
    baseline_C = 1.0
    baseline_rounds = 10
    print(f"Impact of max_iter (C={baseline_C}, rounds={baseline_rounds}):")
    print(f"{'max_iter':<10} {'Accuracy':<12} {'Recall':<12} {'Time':<10}")
    print("-" * 44)
    
    for max_iter in test_configs['max_iter']:
        key = f"max_iter={max_iter}_C={baseline_C}_rounds={baseline_rounds}"
        if key in results_all:
            res = results_all[key]
            sensitivity['max_iter_impact'][max_iter] = {
                'accuracy': res['accuracy'],
                'recall': res['recall'],
                'time': res['training_time'],
            }
            print(f"{max_iter:<10} {res['accuracy']:>10.2%}  {res['recall']:>10.2%}  {res['training_time']:>8.2f}s")
    
    # C impact (keeping max_iter and rounds constant)
    baseline_max_iter = 2000
    print(f"\nImpact of C (max_iter={baseline_max_iter}, rounds={baseline_rounds}):")
    print(f"{'C':<10} {'Accuracy':<12} {'Recall':<12} {'Time':<10}")
    print("-" * 44)
    
    for C in test_configs['C']:
        key = f"max_iter={baseline_max_iter}_C={C}_rounds={baseline_rounds}"
        if key in results_all:
            res = results_all[key]
            sensitivity['C_impact'][C] = {
                'accuracy': res['accuracy'],
                'recall': res['recall'],
                'time': res['training_time'],
            }
            print(f"{C:<10.1f} {res['accuracy']:>10.2%}  {res['recall']:>10.2%}  {res['training_time']:>8.2f}s")
    
    # num_rounds impact (keeping max_iter and C constant)
    print(f"\nImpact of num_rounds (max_iter={baseline_max_iter}, C={baseline_C}):")
    print(f"{'Rounds':<10} {'Accuracy':<12} {'Recall':<12} {'Time':<10}")
    print("-" * 44)
    
    for num_rounds in test_configs['num_rounds']:
        key = f"max_iter={baseline_max_iter}_C={baseline_C}_rounds={num_rounds}"
        if key in results_all:
            res = results_all[key]
            sensitivity['rounds_impact'][num_rounds] = {
                'accuracy': res['accuracy'],
                'recall': res['recall'],
                'time': res['training_time'],
            }
            print(f"{num_rounds:<10} {res['accuracy']:>10.2%}  {res['recall']:>10.2%}  {res['training_time']:>8.2f}s")
    
    # =====================================================================
    # OPTIMIZATION RECOMMENDATIONS
    # =====================================================================
    print("\n" + "=" * 100)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 100 + "\n")
    
    # Find best configuration for recall (clinical safety)
    best_recall = max(results_all.items(), key=lambda x: x[1]['recall'])
    print(f"✅ Best for Recall (Clinical Safety):")
    print(f"   Config: {best_recall[0]}")
    print(f"   Recall: {best_recall[1]['recall']:.2%}")
    print(f"   Accuracy: {best_recall[1]['accuracy']:.2%}")
    
    # Find best configuration for accuracy
    best_accuracy = max(results_all.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n✅ Best for Accuracy:")
    print(f"   Config: {best_accuracy[0]}")
    print(f"   Accuracy: {best_accuracy[1]['accuracy']:.2%}")
    print(f"   Recall: {best_accuracy[1]['recall']:.2%}")
    
    # Find best balanced configuration (highest F1)
    best_f1 = max(results_all.items(), key=lambda x: x[1]['f1_score'])
    print(f"\n✅ Best Balanced Configuration (F1):")
    print(f"   Config: {best_f1[0]}")
    print(f"   F1-Score: {best_f1[1]['f1_score']:.2%}")
    print(f"   Recall: {best_f1[1]['recall']:.2%}")
    print(f"   Accuracy: {best_f1[1]['accuracy']:.2%}")
    
    # Find fastest configuration (that maintains 80% recall)
    safe_configs = {k: v for k, v in results_all.items() if v['recall'] >= 0.80}
    if safe_configs:
        fastest_safe = min(safe_configs.items(), key=lambda x: x[1]['training_time'])
        print(f"\n⚡ Fastest Safe Configuration (recall ≥ 80%):")
        print(f"   Config: {fastest_safe[0]}")
        print(f"   Time: {fastest_safe[1]['training_time']:.2f}s")
        print(f"   Recall: {fastest_safe[1]['recall']:.2%}")
        print(f"   Accuracy: {fastest_safe[1]['accuracy']:.2%}")
    else:
        print(f"\n⚠️ No configuration meets 80% recall requirement")
    
    # =====================================================================
    # KEY INSIGHTS
    # =====================================================================
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100 + "\n")
    
    # Convergence analysis
    min_rounds = min(test_configs['num_rounds'])
    max_rounds = max(test_configs['num_rounds'])
    min_round_results = {k: v for k, v in results_all.items() if v['num_rounds'] == min_rounds}
    max_round_results = {k: v for k, v in results_all.items() if v['num_rounds'] == max_rounds}
    
    if min_round_results and max_round_results:
        avg_recall_min = np.mean([v['recall'] for v in min_round_results.values()])
        avg_recall_max = np.mean([v['recall'] for v in max_round_results.values()])
        improvement = avg_recall_max - avg_recall_min
        
        print(f"1. FL Convergence (more rounds = better):")
        print(f"   {min_rounds} rounds: {avg_recall_min:.2%} avg recall")
        print(f"   {max_rounds} rounds: {avg_recall_max:.2%} avg recall")
        print(f"   Improvement: {improvement:+.2%}")
    
    # Regularization impact
    min_C = min(test_configs['C'])
    max_C = max(test_configs['C'])
    min_C_results = {k: v for k, v in results_all.items() if v['C'] == min_C}
    max_C_results = {k: v for k, v in results_all.items() if v['C'] == max_C}
    
    if min_C_results and max_C_results:
        avg_recall_min_C = np.mean([v['recall'] for v in min_C_results.values()])
        avg_recall_max_C = np.mean([v['recall'] for v in max_C_results.values()])
        
        print(f"\n2. Regularization Sensitivity:")
        print(f"   C={min_C} (strong regularization): {avg_recall_min_C:.2%} avg recall")
        print(f"   C={max_C} (weak regularization): {avg_recall_max_C:.2%} avg recall")
        print(f"   Recommendation: {'Use strong regularization' if avg_recall_min_C > avg_recall_max_C else 'Use weak regularization'}")
    
    # Convergence iterations impact
    min_iter = min(test_configs['max_iter'])
    max_iter = max(test_configs['max_iter'])
    min_iter_results = {k: v for k, v in results_all.items() if v['max_iter'] == min_iter}
    max_iter_results = {k: v for k, v in results_all.items() if v['max_iter'] == max_iter}
    
    if min_iter_results and max_iter_results:
        avg_recall_min_iter = np.mean([v['recall'] for v in min_iter_results.values()])
        avg_recall_max_iter = np.mean([v['recall'] for v in max_iter_results.values()])
        
        print(f"\n3. Model Convergence (max_iter):")
        print(f"   {min_iter} iterations: {avg_recall_min_iter:.2%} avg recall")
        print(f"   {max_iter} iterations: {avg_recall_max_iter:.2%} avg recall")
        if avg_recall_max_iter > avg_recall_min_iter:
            print(f"   Recommendation: Use {max_iter} iterations for stable convergence")
        else:
            print(f"   Recommendation: {min_iter} iterations sufficient (faster)")
    
    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    print("\n" + "=" * 100)
    print("SAVING RESULTS")
    print("=" * 100 + "\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"results/hyperparameter_sensitivity_{timestamp}.json")
    
    all_results = {
        'configuration': {
            'num_clients': num_clients,
            'alpha': alpha,
            'test_configs': test_configs,
        },
        'results': results_all,
        'sensitivity_analysis': sensitivity,
        'best_configurations': {
            'recall': best_recall[0],
            'accuracy': best_accuracy[0],
            'f1_score': best_f1[0],
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    results = run_hyperparameter_sensitivity()
    print("\n✅ Experiment complete!")
