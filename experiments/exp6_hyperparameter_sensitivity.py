"""
Experiment 6: Hyperparameter Sensitivity Analysis

Tests FL model sensitivity to hyperparameter changes.

Key hyperparameters tested:
- max_iter: Model convergence iterations (affects training stability)
- C: Regularization strength inverse (affects overfitting)
- num_rounds: Federated rounds (affects aggregation convergence)
- learning_rate: Optimizer step size (affects convergence speed)
- batch_size: Training batch size (affects gradient stability)

Key questions:
1. How does max_iter affect recall and convergence?
2. How sensitive is the model to regularization (C)?
3. How many FL rounds are needed for stable convergence?
4. How does learning rate affect model performance?
5. How does batch size affect training stability and performance?
6. What's the optimal hyperparameter configuration?
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
from sklearn.linear_model import SGDClassifier

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
        'learning_rate': [0.001, 0.01, 0.1, 1.0],
        'batch_size': [8, 16, 32, 64],
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
    # LEARNING RATE SENSITIVITY (SGD-BASED TRAINING)
    # =====================================================================
    print("\n" + "=" * 100)
    print("LEARNING RATE SENSITIVITY (SGD-based)")
    print("=" * 100 + "\n")
    
    results_lr = {}
    baseline_C = 1.0
    baseline_rounds = 10
    baseline_batch_size = 32
    
    print(f"Testing learning rates with SGD solver...")
    print(f"Configuration: C={baseline_C}, rounds={baseline_rounds}, batch_size={baseline_batch_size}")
    print(f"{'Learning Rate':<15} {'Accuracy':<12} {'Recall':<12} {'Time':<10}")
    print("-" * 49)
    
    for lr in test_configs['learning_rate']:
        config_name = f"lr={lr}_C={baseline_C}_rounds={baseline_rounds}_bs={baseline_batch_size}"
        
        # Initialize feature-engineered data
        X_train_eng_lr = X_train_eng if 'X_train_eng' in locals() else X_train
        
        # Redistribute data
        client_data_dict_lr = distribute_non_iid(X_train_eng_lr, y_train, num_clients, alpha)
        client_data_lr = []
        for client_id in range(num_clients):
            X_client, y_client = client_data_dict_lr[client_id]
            client_data_lr.append({
                'id': client_id,
                'X_train': X_client,
                'y_train': y_client,
                'X_test': X_test_eng,
                'y_test': y_test,
                'n_samples': len(X_client),
            })
        
        start_time = time.time()
        
        # Initialize global model with SGD
        init_model_sgd = SGDClassifier(
            loss='log_loss',  # Logistic regression loss
            max_iter=2000,
            random_state=42,
            class_weight='balanced',
            eta0=lr,  # Learning rate
            alpha=1.0 / (baseline_C * X_train_eng_lr.shape[0]),  # L2 penalty
            solver='sgd',
            verbose=0
        )
        init_model_sgd.fit(client_data_lr[0]['X_train'], client_data_lr[0]['y_train'])
        global_coef = init_model_sgd.coef_.flatten().copy()
        global_intercept = init_model_sgd.intercept_.copy()
        
        # Federated learning rounds with SGD
        for round_num in range(baseline_rounds):
            client_coefs = []
            client_intercepts = []
            client_sample_sizes = []
            
            for client in client_data_lr:
                # Create local SGD model
                local_model_sgd = SGDClassifier(
                    loss='log_loss',
                    max_iter=2000,
                    random_state=42,
                    class_weight='balanced',
                    eta0=lr,
                    alpha=1.0 / (baseline_C * client['X_train'].shape[0]),
                    solver='sgd',
                    verbose=0,
                    warm_start=False
                )
                
                # Set initial weights
                local_model_sgd.fit(client['X_train'][:1], client['y_train'][:1])
                local_model_sgd.coef_ = global_coef.reshape(1, -1).copy()
                local_model_sgd.intercept_ = global_intercept.copy()
                
                # Continue training from global weights
                local_model_sgd.partial_fit(client['X_train'], client['y_train'], classes=[0, 1])
                
                client_coefs.append(local_model_sgd.coef_.flatten())
                client_intercepts.append(local_model_sgd.intercept_)
                client_sample_sizes.append(client['n_samples'])
        
            # Aggregate weights
            total_samples = sum(client_sample_sizes)
            global_coef = np.zeros_like(client_coefs[0])
            global_intercept = np.zeros_like(client_intercepts[0])
            
            for i, (coef, intercept, n_samples) in enumerate(zip(client_coefs, client_intercepts, client_sample_sizes)):
                weight = n_samples / total_samples
                global_coef += weight * coef
                global_intercept += weight * intercept
        
        elapsed_time = time.time() - start_time
        
        # Final evaluation
        final_model_sgd = SGDClassifier(
            loss='log_loss',
            max_iter=2000,
            random_state=42,
            class_weight='balanced',
            eta0=lr,
            alpha=1.0 / (baseline_C * X_train_eng_lr.shape[0]),
            solver='sgd',
            verbose=0
        )
        final_model_sgd.fit(X_train_eng_lr[:1], y_train[:1])
        final_model_sgd.coef_ = global_coef.reshape(1, -1)
        final_model_sgd.intercept_ = global_intercept
        y_pred_lr = final_model_sgd.predict(X_test_eng)
        
        results_lr[config_name] = {
            'learning_rate': lr,
            'accuracy': float(accuracy_score(y_test, y_pred_lr)),
            'precision': float(precision_score(y_test, y_pred_lr, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred_lr, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred_lr, zero_division=0)),
            'training_time': elapsed_time,
        }
        
        res = results_lr[config_name]
        print(f"{lr:<15.4f} {res['accuracy']:>10.2%}  {res['recall']:>10.2%}  {elapsed_time:>8.2f}s")
    
    # =====================================================================
    # BATCH SIZE SENSITIVITY (SGD-BASED TRAINING)
    # =====================================================================
    print("\n" + "=" * 100)
    print("BATCH SIZE SENSITIVITY (SGD-based)")
    print("=" * 100 + "\n")
    
    results_bs = {}
    baseline_lr = 0.01
    
    print(f"Testing batch sizes with SGD solver...")
    print(f"Configuration: C={baseline_C}, rounds={baseline_rounds}, learning_rate={baseline_lr}")
    print(f"{'Batch Size':<15} {'Accuracy':<12} {'Recall':<12} {'Time':<10}")
    print("-" * 49)
    
    for batch_size in test_configs['batch_size']:
        config_name = f"bs={batch_size}_C={baseline_C}_rounds={baseline_rounds}_lr={baseline_lr}"
        
        # Re-initialize data
        client_data_dict_bs = distribute_non_iid(X_train_eng, y_train, num_clients, alpha)
        client_data_bs = []
        for client_id in range(num_clients):
            X_client, y_client = client_data_dict_bs[client_id]
            client_data_bs.append({
                'id': client_id,
                'X_train': X_client,
                'y_train': y_client,
                'X_test': X_test_eng,
                'y_test': y_test,
                'n_samples': len(X_client),
            })
        
        start_time = time.time()
        
        # Initialize with SGD
        init_model_bs = SGDClassifier(
            loss='log_loss',
            max_iter=int(np.ceil(2000 * batch_size / 32)),  # Adjust iterations for batch size
            random_state=42,
            class_weight='balanced',
            eta0=baseline_lr,
            alpha=1.0 / (baseline_C * X_train_eng.shape[0]),
            solver='sgd',
            verbose=0
        )
        init_model_bs.fit(client_data_bs[0]['X_train'], client_data_bs[0]['y_train'])
        global_coef_bs = init_model_bs.coef_.flatten().copy()
        global_intercept_bs = init_model_bs.intercept_.copy()
        
        # Federated rounds with different batch sizes
        for round_num in range(baseline_rounds):
            client_coefs_bs = []
            client_intercepts_bs = []
            client_sample_sizes_bs = []
            
            for client in client_data_bs:
                # Create local model with batch size
                local_model_bs = SGDClassifier(
                    loss='log_loss',
                    max_iter=int(np.ceil(2000 * batch_size / 32)),
                    random_state=42,
                    class_weight='balanced',
                    eta0=baseline_lr,
                    alpha=1.0 / (baseline_C * client['X_train'].shape[0]),
                    solver='sgd',
                    verbose=0,
                    warm_start=False
                )
                
                local_model_bs.fit(client['X_train'][:1], client['y_train'][:1])
                local_model_bs.coef_ = global_coef_bs.reshape(1, -1).copy()
                local_model_bs.intercept_ = global_intercept_bs.copy()
                local_model_bs.partial_fit(client['X_train'], client['y_train'], classes=[0, 1])
                
                client_coefs_bs.append(local_model_bs.coef_.flatten())
                client_intercepts_bs.append(local_model_bs.intercept_)
                client_sample_sizes_bs.append(client['n_samples'])
            
            # Aggregate
            total_samples_bs = sum(client_sample_sizes_bs)
            global_coef_bs = np.zeros_like(client_coefs_bs[0])
            global_intercept_bs = np.zeros_like(client_intercepts_bs[0])
            
            for i, (coef, intercept, n_samples) in enumerate(zip(client_coefs_bs, client_intercepts_bs, client_sample_sizes_bs)):
                weight = n_samples / total_samples_bs
                global_coef_bs += weight * coef
                global_intercept_bs += weight * intercept
        
        elapsed_time = time.time() - start_time
        
        # Final evaluation
        final_model_bs = SGDClassifier(
            loss='log_loss',
            max_iter=int(np.ceil(2000 * batch_size / 32)),
            random_state=42,
            class_weight='balanced',
            eta0=baseline_lr,
            alpha=1.0 / (baseline_C * X_train_eng.shape[0]),
            solver='sgd',
            verbose=0
        )
        final_model_bs.fit(X_train_eng[:1], y_train[:1])
        final_model_bs.coef_ = global_coef_bs.reshape(1, -1)
        final_model_bs.intercept_ = global_intercept_bs
        y_pred_bs = final_model_bs.predict(X_test_eng)
        
        results_bs[config_name] = {
            'batch_size': batch_size,
            'accuracy': float(accuracy_score(y_test, y_pred_bs)),
            'precision': float(precision_score(y_test, y_pred_bs, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred_bs, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred_bs, zero_division=0)),
            'training_time': elapsed_time,
        }
        
        res = results_bs[config_name]
        print(f"{batch_size:<15} {res['accuracy']:>10.2%}  {res['recall']:>10.2%}  {elapsed_time:>8.2f}s")
    print("\n" + "=" * 100)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 100 + "\n")
    
    # Find best configuration for recall (clinical safety)
    best_recall = max(results_all.items(), key=lambda x: x[1]['recall'])
    print(f"✅ Best for Recall (Clinical Safety) - LBFGS:")
    print(f"   Config: {best_recall[0]}")
    print(f"   Recall: {best_recall[1]['recall']:.2%}")
    print(f"   Accuracy: {best_recall[1]['accuracy']:.2%}")
    
    # Find best learning rate
    best_lr = max(results_lr.items(), key=lambda x: x[1]['recall'])
    print(f"\n✅ Best for Recall (Clinical Safety) - SGD (Learning Rate):")
    print(f"   Config: {best_lr[0]}")
    print(f"   Learning Rate: {best_lr[1]['learning_rate']}")
    print(f"   Recall: {best_lr[1]['recall']:.2%}")
    print(f"   Accuracy: {best_lr[1]['accuracy']:.2%}")
    
    # Find best batch size
    best_bs = max(results_bs.items(), key=lambda x: x[1]['recall'])
    print(f"\n✅ Best for Recall (Clinical Safety) - SGD (Batch Size):")
    print(f"   Config: {best_bs[0]}")
    print(f"   Batch Size: {best_bs[1]['batch_size']}")
    print(f"   Recall: {best_bs[1]['recall']:.2%}")
    print(f"   Accuracy: {best_bs[1]['accuracy']:.2%}")
    
    # Find best configuration for accuracy
    best_accuracy = max(results_all.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n✅ Best for Accuracy - LBFGS:")
    print(f"   Config: {best_accuracy[0]}")
    print(f"   Accuracy: {best_accuracy[1]['accuracy']:.2%}")
    print(f"   Recall: {best_accuracy[1]['recall']:.2%}")
    
    # Find best balanced configuration (highest F1)
    best_f1 = max(results_all.items(), key=lambda x: x[1]['f1_score'])
    print(f"\n✅ Best Balanced Configuration (F1) - LBFGS:")
    print(f"   Config: {best_f1[0]}")
    print(f"   F1-Score: {best_f1[1]['f1_score']:.2%}")
    print(f"   Recall: {best_f1[1]['recall']:.2%}")
    print(f"   Accuracy: {best_f1[1]['accuracy']:.2%}")
    
    # Find fastest configuration (that maintains 80% recall)
    safe_configs = {k: v for k, v in results_all.items() if v['recall'] >= 0.80}
    if safe_configs:
        fastest_safe = min(safe_configs.items(), key=lambda x: x[1]['training_time'])
        print(f"\n⚡ Fastest Safe Configuration (recall ≥ 80%) - LBFGS:")
        print(f"   Config: {fastest_safe[0]}")
        print(f"   Time: {fastest_safe[1]['training_time']:.2f}s")
        print(f"   Recall: {fastest_safe[1]['recall']:.2%}")
        print(f"   Accuracy: {fastest_safe[1]['accuracy']:.2%}")
    else:
        print(f"\n⚠️ No LBFGS configuration meets 80% recall requirement")
    
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
    
    # Learning Rate impact (SGD)
    if results_lr:
        min_lr = min(test_configs['learning_rate'])
        max_lr = max(test_configs['learning_rate'])
        min_lr_results = [v['recall'] for k, v in results_lr.items() if v['learning_rate'] == min_lr]
        max_lr_results = [v['recall'] for k, v in results_lr.items() if v['learning_rate'] == max_lr]
        
        if min_lr_results and max_lr_results:
            avg_recall_min_lr = np.mean(min_lr_results)
            avg_recall_max_lr = np.mean(max_lr_results)
            
            print(f"\n4. Learning Rate Sensitivity (SGD):")
            print(f"   LR={min_lr} (conservative): {avg_recall_min_lr:.2%} avg recall")
            print(f"   LR={max_lr} (aggressive): {avg_recall_max_lr:.2%} avg recall")
            print(f"   Recommendation: {'Use conservative LR' if avg_recall_min_lr > avg_recall_max_lr else 'Use aggressive LR' if avg_recall_max_lr > avg_recall_min_lr else 'Learning rate has minimal impact'}")
    
    # Batch Size impact (SGD)
    if results_bs:
        min_bs = min(test_configs['batch_size'])
        max_bs = max(test_configs['batch_size'])
        min_bs_results = [v['recall'] for k, v in results_bs.items() if v['batch_size'] == min_bs]
        max_bs_results = [v['recall'] for k, v in results_bs.items() if v['batch_size'] == max_bs]
        
        if min_bs_results and max_bs_results:
            avg_recall_min_bs = np.mean(min_bs_results)
            avg_recall_max_bs = np.mean(max_bs_results)
            
            print(f"\n5. Batch Size Sensitivity (SGD):")
            print(f"   BS={min_bs} (small batches): {avg_recall_min_bs:.2%} avg recall")
            print(f"   BS={max_bs} (large batches): {avg_recall_max_bs:.2%} avg recall")
            print(f"   Recommendation: {'Use small batches' if avg_recall_min_bs > avg_recall_max_bs else 'Use large batches' if avg_recall_max_bs > avg_recall_min_bs else 'Batch size has minimal impact'}")
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
        'results_lbfgs': results_all,
        'results_learning_rate': results_lr,
        'results_batch_size': results_bs,
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
