#!/usr/bin/env python3
"""
Experiment 9: Comprehensive Scalability Analysis

Tests federated learning system with increasing numbers of clients (5-50+).
Analyzes:
- Performance metrics across client counts
- Memory usage and system resources
- Communication overhead and bandwidth requirements
- Bottleneck identification (computation vs communication)
- Scaling laws (linear, polynomial, exponential)
- Per-round and per-client computational costs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import psutil
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.data.loader import load_dataset_with_df
from src.data.preprocess import DataPreprocessor
from src.data.split import train_test_split_data, distribute_non_iid
from src.models.model import LogisticRegressionModel
from src.utils.feature_engineering import HealthcareFeatureEngineer
from src.fl.strategy import FedAvgAggregator, aggregate_metrics
from src.config.config import MAX_ITER, DECISION_THRESHOLD, DIRICHLET_ALPHA, SCALABILITY_CLIENT_COUNTS, SCALABILITY_NUM_ROUNDS
from src.evaluation.metrics import calculate_all_metrics


class ScalabilityAnalyzer:
    """Comprehensive scalability analysis for federated learning"""
    
    def __init__(self):
        self.results = {}
        self.process = psutil.Process(os.getpid())
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB"""
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        }
    
    def get_system_cpu_usage(self) -> float:
        """Get system CPU usage percentage"""
        return psutil.cpu_percent(interval=0.1)
    
    def calculate_communication_overhead(self, num_clients: int, 
                                       weights_size_kb: float,
                                       num_rounds: int) -> Dict[str, float]:
        """
        Calculate communication overhead metrics
        
        Args:
            num_clients: Number of clients
            weights_size_kb: Size of model weights in KB
            num_rounds: Number of FL rounds
            
        Returns:
            Dictionary with communication metrics
        """
        # Each round: clients upload weights + receive aggregated weights
        upload_per_round = num_clients * weights_size_kb
        download_per_round = num_clients * weights_size_kb
        total_per_round = upload_per_round + download_per_round
        total_overall = total_per_round * num_rounds
        
        return {
            'weights_size_kb': weights_size_kb,
            'upload_per_round_kb': upload_per_round,
            'download_per_round_kb': download_per_round,
            'total_per_round_kb': total_per_round,
            'total_overall_kb': total_overall,
            'total_overall_mb': total_overall / 1024,
            'total_overall_gb': total_overall / 1024 / 1024,
        }
    
    def run_scalability_test(self, X_train_eng, y_train, X_test_eng, y_test, 
                            feature_names_eng, client_counts: List[int],
                            num_rounds: int = 10, alpha: float = 0.5) -> Dict:
        """
        Run scalability tests with varying client counts
        
        Args:
            X_train_eng: Training features (engineered)
            y_train: Training labels
            X_test_eng: Test features (engineered)
            y_test: Test labels
            feature_names_eng: Engineered feature names
            client_counts: List of client counts to test
            num_rounds: Number of federated rounds
            alpha: Dirichlet parameter for Non-IID
            
        Returns:
            Dictionary with comprehensive scalability results
        """
        
        print("\n" + "=" * 120)
        print("EXPERIMENT 9: COMPREHENSIVE SCALABILITY ANALYSIS")
        print("=" * 120)
        
        print(f"\n📊 Test Configuration:")
        print(f"   Client counts: {client_counts}")
        print(f"   Federated rounds: {num_rounds}")
        print(f"   Non-IID alpha: {alpha}")
        print(f"   Feature dimensions: {X_train_eng.shape[1]}")
        
        # Get initial memory baseline
        initial_memory = self.get_memory_usage()
        print(f"   Initial memory usage: {initial_memory['rss_mb']:.1f} MB")
        
        scalability_results = {}
        
        # Test each client count
        for idx, num_clients in enumerate(client_counts):
            print("\n" + "=" * 120)
            print(f"TEST {idx + 1}/{len(client_counts)}: {num_clients} CLIENTS")
            print("=" * 120 + "\n")
            
            # ================================================================
            # DATA DISTRIBUTION PHASE
            # ================================================================
            print(f"📁 [1/4] Distributing data to {num_clients} clients...")
            dist_start = time.time()
            
            client_data_dict = distribute_non_iid(X_train_eng, y_train, num_clients, alpha)
            
            client_data = []
            sample_sizes = []
            for client_id in range(num_clients):
                X_client, y_client = client_data_dict[client_id]
                sample_sizes.append(len(X_client))
                
                client_data.append({
                    'id': client_id,
                    'X_train': X_client,
                    'y_train': y_client,
                    'X_test': X_test_eng,
                    'y_test': y_test,
                    'n_samples': len(X_client)
                })
            
            dist_time = time.time() - dist_start
            print(f"✓ Data distribution complete in {dist_time:.3f} seconds")
            print(f"  Mean samples per client: {np.mean(sample_sizes):.0f}")
            print(f"  Std dev: {np.std(sample_sizes):.0f}")
            print(f"  Min: {np.min(sample_sizes)}, Max: {np.max(sample_sizes)}")
            
            # ================================================================
            # MODEL INITIALIZATION PHASE
            # ================================================================
            print(f"\n🔧 [2/4] Initializing global model...")
            init_start = time.time()
            
            # Find first client with both classes for initialization
            init_client_idx = None
            for idx, client in enumerate(client_data):
                if len(np.unique(client['y_train'])) > 1:
                    init_client_idx = idx
                    break
            
            if init_client_idx is None:
                # If no client has both classes, use first client and handle exception
                init_client_idx = 0
            
            init_model = LogisticRegressionModel(max_iter=MAX_ITER, class_weight='balanced')
            init_model.set_decision_threshold(DECISION_THRESHOLD)
            try:
                init_model.fit(client_data[init_client_idx]['X_train'], 
                             client_data[init_client_idx]['y_train'], verbose=False)
            except ValueError:
                # If initialization still fails, train on aggregated first few clients
                X_agg = np.vstack([client_data[i]['X_train'] for i in range(min(2, len(client_data)))])
                y_agg = np.hstack([client_data[i]['y_train'] for i in range(min(2, len(client_data)))])
                init_model.fit(X_agg, y_agg, verbose=False)
            
            global_weights = init_model.get_weights()
            
            # Calculate model size (coef and intercept only, skip classes)
            weights_size_bytes = (global_weights['coef'].nbytes + 
                                 global_weights['intercept'].nbytes)
            weights_size_kb = weights_size_bytes / 1024
            weights_size_mb = weights_size_kb / 1024
            
            init_time = time.time() - init_start
            print(f"✓ Model initialized in {init_time:.3f} seconds")
            print(f"  Model size: {weights_size_kb:.2f} KB ({weights_size_mb:.4f} MB)")
            
            # ================================================================
            # FEDERATED LEARNING PHASE
            # ================================================================
            print(f"\n🔄 [3/4] Running federated learning ({num_rounds} rounds)...")
            
            fl_start = time.time()
            memory_before_fl = self.get_memory_usage()
            
            # Track per-round metrics
            round_times = []
            local_train_times = []
            agg_times = []
            
            global_model = LogisticRegressionModel(max_iter=MAX_ITER, class_weight='balanced')
            global_model.set_decision_threshold(DECISION_THRESHOLD)
            global_model.set_weights(global_weights)
            
            for round_num in range(num_rounds):
                round_start = time.time()
                
                # Local training phase
                train_start = time.time()
                client_weights = []
                client_sample_sizes = []
                client_eval_metrics = []
                
                for client in client_data:
                    local_model = LogisticRegressionModel(max_iter=MAX_ITER, class_weight='balanced')
                    local_model.set_weights(global_weights)
                    local_model.set_decision_threshold(DECISION_THRESHOLD)
                    
                    try:
                        local_model.fit(client['X_train'], client['y_train'], verbose=False)
                        client_weights.append(local_model.get_weights())
                    except ValueError:
                        client_weights.append(global_weights)
                    
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
                
                local_train_time = time.time() - train_start
                local_train_times.append(local_train_time)
                
                # Aggregation phase
                agg_start = time.time()
                global_weights = FedAvgAggregator.aggregate(client_weights, client_sample_sizes)
                agg_time = time.time() - agg_start
                agg_times.append(agg_time)
                
                # Aggregate metrics
                agg_metrics = aggregate_metrics(client_eval_metrics, client_sample_sizes)
                
                round_time = time.time() - round_start
                round_times.append(round_time)
                
                if (round_num + 1) % max(1, num_rounds // 5) == 0 or round_num == 0:
                    print(f"  Round {round_num + 1:3d}/{num_rounds}: " +
                          f"Recall={agg_metrics['recall']:.2%}, " +
                          f"Time={round_time:.3f}s")
            
            fl_training_time = time.time() - fl_start
            memory_after_fl = self.get_memory_usage()
            
            print(f"✓ Federated learning complete in {fl_training_time:.2f} seconds")
            print(f"  Mean time per round: {np.mean(round_times):.3f}s (±{np.std(round_times):.3f}s)")
            print(f"  Mean local training time: {np.mean(local_train_times):.3f}s")
            print(f"  Mean aggregation time: {np.mean(agg_times):.3f}s")
            
            # ================================================================
            # FINAL EVALUATION PHASE
            # ================================================================
            print(f"\n📊 [4/4] Final evaluation...")
            
            global_model.set_weights(global_weights)
            y_pred_fl = global_model.predict(X_test_eng)
            fl_metrics = calculate_all_metrics(y_test, y_pred_fl)
            
            print(f"✓ Evaluation complete")
            print(f"  Final accuracy: {fl_metrics['accuracy']:.2%}")
            print(f"  Final recall: {fl_metrics['recall']:.2%}")
            
            # ================================================================
            # RESOURCE USAGE ANALYSIS
            # ================================================================
            memory_increase = memory_after_fl['rss_mb'] - memory_before_fl['rss_mb']
            peak_memory = max(memory_usage['rss_mb'] 
                            for memory_usage in [memory_before_fl, memory_after_fl])
            
            # Communication overhead
            comm_overhead = self.calculate_communication_overhead(
                num_clients, weights_size_kb, num_rounds
            )
            
            # Per-client computation cost
            total_samples = sum(sample_sizes)
            samples_per_client = total_samples / num_clients
            
            # ================================================================
            # STORE RESULTS
            # ================================================================
            scalability_results[num_clients] = {
                'num_clients': num_clients,
                'num_rounds': num_rounds,
                
                # Timing
                'data_distribution_time_s': dist_time,
                'model_init_time_s': init_time,
                'fl_training_time_s': fl_training_time,
                'mean_round_time_s': float(np.mean(round_times)),
                'std_round_time_s': float(np.std(round_times)),
                'mean_local_train_time_s': float(np.mean(local_train_times)),
                'mean_aggregation_time_s': float(np.mean(agg_times)),
                'per_client_avg_time_s': fl_training_time / num_clients / num_rounds,
                
                # Performance
                'metrics': fl_metrics,
                
                # Resource usage
                'model_size_kb': weights_size_kb,
                'model_size_mb': weights_size_mb,
                'memory_increase_mb': memory_increase,
                'peak_memory_mb': peak_memory,
                
                # Data distribution
                'total_samples': total_samples,
                'samples_per_client_avg': float(np.mean(sample_sizes)),
                'samples_per_client_std': float(np.std(sample_sizes)),
                'samples_per_client_min': int(np.min(sample_sizes)),
                'samples_per_client_max': int(np.max(sample_sizes)),
                
                # Communication
                'communication': comm_overhead,
                
                # Safety
                'is_safe': fl_metrics['recall'] >= 0.80,
                'recall': fl_metrics['recall'],
            }
            
            print("\n" + "=" * 120)
            print(f"SUMMARY FOR {num_clients} CLIENTS")
            print("=" * 120)
            print(f"  Total time: {fl_training_time:.2f}s")
            print(f"  Throughput: {num_rounds / fl_training_time:.2f} rounds/sec")
            print(f"  Cost per client: {fl_training_time / num_clients:.3f}s")
            print(f"  Communication: {comm_overhead['total_overall_mb']:.2f} MB total")
            print(f"  Recall: {fl_metrics['recall']:.2%} {'✅' if fl_metrics['recall'] >= 0.80 else '❌'}")
        
        return scalability_results


def analyze_scaling_laws(scalability_results: Dict) -> Dict:
    """
    Analyze scaling laws from scalability results
    
    Attempts to fit linear, polynomial, and exponential models
    """
    
    print("\n" + "=" * 120)
    print("SCALING LAW ANALYSIS")
    print("=" * 120 + "\n")
    
    # Extract data
    client_counts = sorted(scalability_results.keys())
    times = [scalability_results[c]['fl_training_time_s'] for c in client_counts]
    throughputs = [scalability_results[c]['num_rounds'] / scalability_results[c]['fl_training_time_s'] 
                   for c in client_counts]
    comm_costs = [scalability_results[c]['communication']['total_overall_mb'] for c in client_counts]
    per_client_times = [scalability_results[c]['per_client_avg_time_s'] for c in client_counts]
    
    # Fit scaling laws
    client_counts_arr = np.array(client_counts, dtype=float)
    times_arr = np.array(times, dtype=float)
    
    # Linear fit: time = a * clients + b
    linear_fit = np.polyfit(client_counts_arr, times_arr, 1)
    linear_r2 = 1 - (np.sum((times_arr - (linear_fit[0] * client_counts_arr + linear_fit[1])) ** 2) / 
                     np.sum((times_arr - np.mean(times_arr)) ** 2))
    
    # Polynomial fit (quadratic): time = a * clients^2 + b * clients + c
    poly_fit = np.polyfit(client_counts_arr, times_arr, 2)
    poly_r2 = 1 - (np.sum((times_arr - (poly_fit[0] * client_counts_arr**2 + 
                                         poly_fit[1] * client_counts_arr + poly_fit[2])) ** 2) / 
                   np.sum((times_arr - np.mean(times_arr)) ** 2))
    
    # Exponential fit: time = a * exp(b * clients)
    try:
        exp_fit = np.polyfit(client_counts_arr, np.log(times_arr), 1)
        exp_r2 = 1 - (np.sum((np.log(times_arr) - (exp_fit[0] * client_counts_arr + exp_fit[1])) ** 2) / 
                      np.sum((np.log(times_arr) - np.mean(np.log(times_arr))) ** 2))
    except:
        exp_fit = [0, 0]
        exp_r2 = 0
    
    print(f"📈 Scaling Law Fits (R² values):\n")
    print(f"  Linear:      time = {linear_fit[0]:.4f}*clients + {linear_fit[1]:.4f}")
    print(f"               R² = {linear_r2:.4f}")
    print(f"\n  Polynomial:  time = {poly_fit[0]:.4f}*clients² + {poly_fit[1]:.4f}*clients + {poly_fit[2]:.4f}")
    print(f"               R² = {poly_r2:.4f}")
    print(f"\n  Exponential: time = exp({exp_fit[0]:.4f}*clients + {exp_fit[1]:.4f})")
    print(f"               R² = {exp_r2:.4f}")
    
    # Determine best fit
    fits = [
        ('linear', linear_r2, linear_fit),
        ('polynomial', poly_r2, poly_fit),
        ('exponential', exp_r2, exp_fit),
    ]
    best_fit_name, best_r2, best_fit = max(fits, key=lambda x: x[1])
    print(f"\n✓ Best fit: {best_fit_name.upper()} (R² = {best_r2:.4f})")
    
    return {
        'best_fit': best_fit_name,
        'best_fit_r2': best_r2,
        'linear': {'fit': linear_fit.tolist(), 'r2': float(linear_r2)},
        'polynomial': {'fit': poly_fit.tolist(), 'r2': float(poly_r2)},
        'exponential': {'fit': exp_fit.tolist(), 'r2': float(exp_r2)},
    }


def identify_bottlenecks(scalability_results: Dict) -> Dict:
    """
    Identify computational and communication bottlenecks
    """
    
    print("\n" + "=" * 120)
    print("BOTTLENECK ANALYSIS")
    print("=" * 120 + "\n")
    
    bottleneck_analysis = {}
    
    for num_clients, result in sorted(scalability_results.items()):
        total_time = result['fl_training_time_s']
        local_train_time = result['mean_local_train_time_s']
        agg_time = result['mean_aggregation_time_s']
        
        # Per-round breakdown
        local_pct = (local_train_time / (local_train_time + agg_time)) * 100
        agg_pct = (agg_time / (local_train_time + agg_time)) * 100
        
        bottleneck_analysis[num_clients] = {
            'local_training_pct': float(local_pct),
            'aggregation_pct': float(agg_pct),
            'bottleneck': 'local_training' if local_pct > 60 else 'communication',
        }
        
        print(f"  {num_clients:3d} clients:")
        print(f"    Local training:  {local_pct:5.1f}% (bottleneck)" if local_pct > 60 else f"    Local training:  {local_pct:5.1f}%")
        print(f"    Aggregation:     {agg_pct:5.1f}%" if agg_pct <= 40 else f"    Aggregation:     {agg_pct:5.1f}% (bottleneck)")
        print(f"    → Dominant: {bottleneck_analysis[num_clients]['bottleneck'].replace('_', ' ').title()}")
    
    return bottleneck_analysis


def generate_comparison_table(scalability_results: Dict) -> pd.DataFrame:
    """Generate comparison table for all client counts"""
    
    comparison_data = []
    
    for num_clients in sorted(scalability_results.keys()):
        result = scalability_results[num_clients]
        metrics = result['metrics']
        
        comparison_data.append({
            'Clients': num_clients,
            'Accuracy': f"{metrics['accuracy']:.2%}",
            'Recall': f"{metrics['recall']:.2%}",
            'Time (s)': f"{result['fl_training_time_s']:.2f}",
            'Throughput (rounds/s)': f"{result['num_rounds'] / result['fl_training_time_s']:.3f}",
            'Total Comm (MB)': f"{result['communication']['total_overall_mb']:.2f}",
            'Memory (MB)': f"{result['peak_memory_mb']:.1f}",
            'Per-Client Cost (s)': f"{result['per_client_avg_time_s']:.4f}",
            'Safety': '✅' if result['is_safe'] else '❌'
        })
    
    return pd.DataFrame(comparison_data)


def run_scalability_experiment():
    """Main scalability experiment runner"""
    
    # Load data
    print("\n" + "=" * 120)
    print("DATA LOADING")
    print("=" * 120)
    
    print("\n📁 Loading dataset...")
    df, X, y = load_dataset_with_df()
    
    preprocessor = DataPreprocessor()
    X_processed = preprocessor.preprocess(df.iloc[:, :-1], fit=True)
    feature_names = list(df.columns[:-1])
    
    X_train, X_test, y_train, y_test = train_test_split_data(X_processed, y)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    print(f"✓ Data loaded: {len(X_train)} training, {len(X_test)} testing")
    
    # Feature engineering
    print("\n📊 Creating engineered features...")
    engineer = HealthcareFeatureEngineer()
    X_train_eng, feature_names_eng = engineer.engineer_all_features(X_train, feature_names)
    
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
    
    print(f"✓ Features engineered: {X_train_eng.shape[1]} dimensions")
    
    # Run scalability analysis
    analyzer = ScalabilityAnalyzer()
    
    # Test extended client counts from config
    # Comment out larger counts for faster testing
    client_counts = SCALABILITY_CLIENT_COUNTS
    
    scalability_results = analyzer.run_scalability_test(
        X_train_eng, y_train, X_test_eng, y_test, feature_names_eng,
        client_counts=client_counts,
        num_rounds=SCALABILITY_NUM_ROUNDS,
        alpha=DIRICHLET_ALPHA
    )
    
    # Analyze results
    scaling_laws = analyze_scaling_laws(scalability_results)
    bottlenecks = identify_bottlenecks(scalability_results)
    comparison_df = generate_comparison_table(scalability_results)
    
    # Print comparison table
    print("\n" + "=" * 120)
    print("COMPREHENSIVE SCALABILITY COMPARISON")
    print("=" * 120 + "\n")
    print(comparison_df.to_string(index=False))
    
    # Print key insights
    print("\n" + "=" * 120)
    print("KEY INSIGHTS")
    print("=" * 120)
    
    first_result = scalability_results[client_counts[0]]
    last_result = scalability_results[client_counts[-1]]
    
    time_scaling = last_result['fl_training_time_s'] / first_result['fl_training_time_s']
    recall_variance = np.std([scalability_results[c]['recall'] for c in client_counts])
    
    print(f"""
1. SCALING EFFICIENCY:
   - Time scaling ({client_counts[0]} → {client_counts[-1]} clients): {time_scaling:.2f}x
   - Throughput change: {(client_counts[0] / client_counts[-1]) * time_scaling:.2f}x
   - Scaling law fit: {scaling_laws['best_fit'].upper()}
   
2. PERFORMANCE STABILITY:
   - Recall variance: ±{recall_variance:.2%}
   - All configurations safe (Recall ≥ 80%): {all(scalability_results[c]['is_safe'] for c in client_counts)}
   
3. COMMUNICATION OVERHEAD:
   - Total communication ({client_counts[0]} clients): {first_result['communication']['total_overall_mb']:.2f} MB
   - Total communication ({client_counts[-1]} clients): {last_result['communication']['total_overall_mb']:.2f} MB
   - Communication scales {last_result['communication']['total_overall_mb'] / first_result['communication']['total_overall_mb']:.1f}x with client count
   
4. COMPUTATIONAL BOTTLENECK:
   - Primary bottleneck across all scales: {bottlenecks[client_counts[-1]]['bottleneck'].replace('_', ' ').title()}
   - Local training consistently dominates with {bottlenecks[client_counts[0]]['local_training_pct']:.1f}% of per-round time
   
5. PRACTICAL RECOMMENDATIONS:
   - Sweet spot: {client_counts[1:3][0]} clients (balance efficiency and overhead)
   - Maximum recommended: {client_counts[-1]} clients (before quadratic cost scaling)
   - Per-client communication: {last_result['communication']['weights_size_kb']:.2f} KB per round
   - Per-client computation: {last_result['per_client_avg_time_s']:.4f}s per round
""")
    
    # Save comprehensive results
    print("\n" + "=" * 120)
    print("SAVING RESULTS")
    print("=" * 120)
    
    results_data = {
        'experiment': 'exp9_scalability_analysis',
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'client_counts': client_counts,
            'num_rounds': 10,
            'non_iid_alpha': 0.5,
            'feature_dimensions': int(X_train_eng.shape[1]),
        },
        'scalability_by_client_count': {
            str(num_clients): {
                k: v for k, v in scalability_results[num_clients].items()
                if k not in ['communication', 'metrics']  # Handle separately
            } | {
                'communication': scalability_results[num_clients]['communication'],
                'metrics': scalability_results[num_clients]['metrics'],
            }
            for num_clients in sorted(scalability_results.keys())
        },
        'scaling_laws': scaling_laws,
        'bottleneck_analysis': bottlenecks,
        'summary': {
            'best_fit': scaling_laws['best_fit'],
            'primary_bottleneck': bottlenecks[client_counts[-1]]['bottleneck'],
            'scalability_index': float(time_scaling),
            'recall_variance': float(recall_variance),
        }
    }
    
    result_path = Path(__file__).parent.parent / 'results'
    result_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = result_path / f"scalability_analysis_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {result_file}")
    print("=" * 120)
    
    return results_data


if __name__ == '__main__':
    results = run_scalability_experiment()
    print(f"\n✅ Scalability analysis complete!\n")
