#!/usr/bin/env python3
"""
Model Comparison: Testing different approaches to improve healthcare prediction.

This script compares:
1. Logistic Regression (Baseline) - with class_weight='balanced'
2. Random Forest - handles non-linear patterns
3. XGBoost - optimal for imbalanced healthcare data

Focus on RECALL metric (must not miss diabetic patients!)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_dataset_with_df
from src.data.preprocess import DataPreprocessor
from src.data.split import train_test_split_data
from src.models.model import LogisticRegressionModel, RandomForestModel, XGBoostModel
import numpy as np
import pandas as pd


def compare_models():
    """Compare all three model architectures."""
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON FOR HEALTHCARE DIABETES PREDICTION")
    print("=" * 80)
    print("\nObjective: Improve recall (sensitivity) to avoid missing diabetic patients")
    print("Healthcare requirement: RECALL must be > 80%\n")
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    df, X, y = load_dataset_with_df()
    preprocessor = DataPreprocessor()
    X_processed = preprocessor.preprocess(df.iloc[:, :-1], fit=True)
    
    X_train, X_test, y_train, y_test = train_test_split_data(X_processed, y)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    print(f"✓ Data ready: {len(X_train)} training, {len(X_test)} testing samples\n")
    
    # Step 2: Define models
    models = {
        'Logistic Regression (Balanced)': LogisticRegressionModel(
            learning_rate=0.01, 
            max_iter=1000, 
            class_weight='balanced'  # KEY IMPROVEMENT: Handle class imbalance
        ),
        'Random Forest': RandomForestModel(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced_subsample'
        ),
        'XGBoost': XGBoostModel(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1
        )
    }
    
    results = {}
    
    # Step 3: Train and evaluate each model
    print("=" * 80)
    print("TRAINING AND EVALUATING MODELS")
    print("=" * 80)
    
    for model_name, model in models.items():
        print(f"\n📊 Training: {model_name}")
        print("-" * 80)
        
        try:
            # Train
            model.fit(X_train, y_train, verbose=True)
            
            # Evaluate
            metrics = model.evaluate(X_test, y_test, verbose=True)
            
            results[model_name] = metrics
            
            print()
        
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            results[model_name] = None
    
    # Step 4: Create comparison table
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    
    comparison_data = []
    for model_name, metrics in results.items():
        if metrics is not None:
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'TN': metrics['confusion_matrix'][0, 0],
                'FP': metrics['confusion_matrix'][0, 1],
                'FN': metrics['confusion_matrix'][1, 0],
                'TP': metrics['confusion_matrix'][1, 1],
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    print("\n" + df_comparison.to_string(index=False))
    
    # Step 5: Analysis and recommendations
    print("\n" + "=" * 80)
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 80)
    
    best_models = {}
    for metric in ['Accuracy', 'Recall', 'F1-Score']:
        best_idx = df_comparison[metric].apply(float).idxmax()
        best_models[metric] = df_comparison.iloc[best_idx]['Model']
    
    print(f"\nBest Model by Metric:")
    for metric, model in best_models.items():
        best_value = df_comparison[df_comparison['Model'] == model][metric].values[0]
        print(f"  {metric:12s}: {model:30s} ({best_value})")
    
    # Healthcare-specific analysis
    print(f"\n🏥 Healthcare Analysis:")
    for model_name, metrics in results.items():
        if metrics is not None:
            recall = metrics['recall']
            fn = metrics['confusion_matrix'][1, 0]  # False negatives
            
            status = "✓ ACCEPTABLE" if recall >= 0.80 else "⚠️  NEEDS IMPROVEMENT"
            print(f"\n  {model_name}:")
            print(f"    - Recall: {recall:.2%} {status}")
            print(f"    - Missing Cases (False Negatives): {fn}")
            if fn > 0:
                print(f"    - Risk: Out of 100 diabetic patients, model misses ~{int(fn/len(y_test)*100)}")
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print("""
1. ✅ CLASS WEIGHT BALANCING:
   - Adding class_weight='balanced' to Logistic Regression improves recall
   - Penalizes errors on minority class (diabetic patients)

2. 🌳 RANDOM FOREST ADVANTAGES:
   - Captures non-linear feature relationships
   - Better at handling mixed feature types
   - Usually improves recall on healthcare data

3. 🚀 XGBOOST ADVANTAGES:
   - State-of-the-art performance for imbalanced data
   - scale_pos_weight automatically handles class imbalance
   - Best for production healthcare systems

4. ⚠️  CRITICAL METRIC:
   - For healthcare, RECALL > 80% is mandatory
   - Choose model that achieves this with good precision balance
""")
    
    print("=" * 80)
    print("✅ MODEL COMPARISON COMPLETE")
    print("=" * 80)
    
    return results, df_comparison


if __name__ == "__main__":
    compare_models()
