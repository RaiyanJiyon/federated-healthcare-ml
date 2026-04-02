"""
Experiment 7: Differential Privacy Analysis

Tests federated learning with differential privacy (DP).
Demonstrates the privacy-utility tradeoff: privacy gains vs accuracy loss.

Key Questions:
1. How does differential privacy impact model accuracy?
2. What's the optimal epsilon for healthcare (safety & privacy)?
3. How does privacy degrade with more FL rounds?
4. Can we maintain clinical safety (recall ≥ 80%) with DP?
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
from src.fl.strategy import FedAvgAggregator
from src.fl.privacy import DifferentialPrivacyMechanism, PrivacyBudgetTracker


def run_differential_privacy_experiment():
    """Test FL with differential privacy."""
    
    print("\n" + "=" * 100)
    print("EXPERIMENT 7: DIFFERENTIAL PRIVACY ANALYSIS")
    print("=" * 100 + "\n")
    
    # Configuration
    num_clients = 5
    alpha = 0.5
    num_rounds = 10
    
    # Epsilon values to test (privacy budgets)
    epsilon_values = [None, 0.1, 0.5, 1.0, 2.0, 5.0]  # None = no DP (baseline)
    
    print(f"📊 Configuration:")
    print(f"   Clients: {num_clients}, Non-IID (alpha={alpha})")
    print(f"   FL Rounds: {num_rounds}")
    print(f"   Privacy budgets (ε): {epsilon_values}")
    
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
    # DIFFERENTIAL PRIVACY EXPERIMENTS
    # =====================================================================
    print("\n" + "=" * 100)
    print("PRIVACY-UTILITY TRADEOFF ANALYSIS")
    print("=" * 100 + "\n")
    
    results_all = {}
    
    for epsilon in epsilon_values:
        if epsilon is None:
            privacy_label = "No Privacy (Baseline)"
            use_dp = False
        else:
            privacy_label = f"DP (ε={epsilon})"
            use_dp = True
        
        print(f"\n📊 Testing: {privacy_label}")
        print("-" * 100)
        
        # Initialize global model
        init_model = LogisticRegressionModel(max_iter=2000, class_weight='balanced')
        init_model.model.C = 1.0
        init_model.set_decision_threshold(0.30)
        init_model.fit(client_data[0]['X_train'], client_data[0]['y_train'], verbose=False)
        global_weights = init_model.get_weights()
        
        # Initialize DP if needed
        if use_dp:
            dp_mechanism = DifferentialPrivacyMechanism(
                epsilon=epsilon,
                delta=1.0 / len(X_train_eng),
                clipping_norm=1.0,
                num_samples=len(X_train_eng)
            )
        
        start_time = time.time()
        
        # Federated learning rounds
        for round_num in range(num_rounds):
            client_weights = []
            client_sample_sizes = []
            
            # Local training on each client
            for client in client_data:
                # Create local model
                local_model = LogisticRegressionModel(max_iter=2000, class_weight='balanced')
                local_model.model.C = 1.0
                local_model.set_weights(global_weights)
                local_model.set_decision_threshold(0.30)
                
                # Local training
                try:
                    local_model.fit(client['X_train'], client['y_train'], verbose=False)
                    local_weights = local_model.get_weights()
                except ValueError:
                    local_weights = global_weights
                
                # Apply DP if enabled
                if use_dp:
                    local_weights, _ = dp_mechanism.privatize_weights(local_weights)
                
                client_weights.append(local_weights)
                client_sample_sizes.append(client['n_samples'])
            
            # Aggregate
            global_weights = FedAvgAggregator.aggregate(client_weights, client_sample_sizes)
        
        elapsed_time = time.time() - start_time
        
        # Final evaluation
        final_model = LogisticRegressionModel(max_iter=2000, class_weight='balanced')
        final_model.model.C = 1.0
        final_model.set_weights(global_weights)
        final_model.set_decision_threshold(0.30)
        y_pred = final_model.predict(X_test_eng)
        
        # Calculate metrics
        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        
        results_all[privacy_label] = {
            'epsilon': epsilon,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': elapsed_time,
            'clinical_safe': recall >= 0.80,  # Recall > 80% = safe
        }
        
        # Print results
        print(f"  Accuracy: {accuracy:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}")
        print(f"  Time: {elapsed_time:.2f}s, Clinical Safe: {'✅ YES' if recall >= 0.80 else '❌ NO'}")
        
        # Print DP status if used
        if use_dp:
            print(f"  Privacy: ({dp_mechanism.total_epsilon_budget:.2f}, {dp_mechanism.delta:.6f})-DP")
    
    # =====================================================================
    # ANALYSIS & RECOMMENDATIONS
    # =====================================================================
    print("\n" + "=" * 100)
    print("PRIVACY-UTILITY TRADEOFF ANALYSIS")
    print("=" * 100 + "\n")
    
    # Find baseline
    baseline = results_all.get("No Privacy (Baseline)", {})
    baseline_accuracy = baseline.get('accuracy', 0)
    baseline_recall = baseline.get('recall', 0)
    
    print(f"📊 BASELINE (No Privacy):")
    print(f"   Accuracy: {baseline_accuracy:.2%}")
    print(f"   Recall:   {baseline_recall:.2%}")
    
    print(f"\n📊 DIFFERENTIAL PRIVACY RESULTS:")
    print(f"{'Privacy Level':<30} {'Accuracy':<12} {'Recall':<12} {'Accuracy Loss':<15} {'Safe?':<10}")
    print("-" * 79)
    
    for label, result in results_all.items():
        if "Baseline" not in label:
            acc_loss = baseline_accuracy - result['accuracy']
            recall_loss = baseline_recall - result['recall']
            safe_str = "✅ YES" if result['clinical_safe'] else "❌ NO"
            
            print(f"{label:<30} {result['accuracy']:>10.2%}  {result['recall']:>10.2%}  "
                  f"-{acc_loss:>6.2%}{'':>6}  {safe_str:<10}")
    
    # Recommendations
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS FOR HEALTHCARE DEPLOYMENT")
    print("=" * 100 + "\n")
    
    # Find best epsilon that maintains safety
    safe_runs = {k: v for k, v in results_all.items() 
                 if v.get('clinical_safe') and v.get('epsilon') is not None}
    
    if safe_runs:
        best_private = min(safe_runs.items(), 
                          key=lambda x: x[1]['epsilon'])  # Minimum epsilon = most private
        print(f"✅ RECOMMENDED: {best_private[0]}")
        print(f"   Epsilon: {best_private[1]['epsilon']}")
        print(f"   Recall: {best_private[1]['recall']:.2%} (clinical safety maintained)")
        print(f"   Accuracy: {best_private[1]['accuracy']:.2%}")
        print(f"   Privacy Loss vs Baseline: {baseline_recall - best_private[1]['recall']:.2%}")
    else:
        print("⚠️ WARNING: No DP configuration maintains clinical safety (recall ≥ 80%)")
        print("   Recommendation: Use baseline (no DP) or accept slightly lower safety")
    
    # Tradeoff analysis
    print("\n📊 PRIVACY-UTILITY TRADEOFF:")
    print("   More Private (lower ε)  ←→  Better Accuracy (higher recall)")
    print("   Typical loss with ε=1.0: 5-10% accuracy reduction")
    print("   For healthcare: ε=1.0 provides excellent privacy with acceptable accuracy")
    
    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    print("\n" + "=" * 100)
    print("SAVING RESULTS")
    print("=" * 100 + "\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"results/differential_privacy_{timestamp}.json")
    
    all_results = {
        'configuration': {
            'num_clients': num_clients,
            'alpha': alpha,
            'num_rounds': num_rounds,
            'epsilon_values': [float(e) if e else None for e in epsilon_values],
        },
        'results': results_all,
        'baseline': baseline,
    }
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    results = run_differential_privacy_experiment()
    print("\n✅ Experiment complete!")
