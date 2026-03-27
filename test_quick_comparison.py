#!/usr/bin/env python3
"""
Quick Model Comparison: Logistic Regression with Class Balancing vs Random Forest
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_dataset_with_df
from src.data.preprocess import DataPreprocessor
from src.data.split import train_test_split_data
from src.models.model import LogisticRegressionModel, RandomForestModel
import pandas as pd

print("\n" + "=" * 80)
print("IMPROVED MODEL COMPARISON - HEALTHCARE DIABETES PREDICTION")
print("=" * 80)
print("\nKey Improvement: Added class_weight='balanced' to handle class imbalance\n")

# Load and preprocess
print("Loading data...")
df, X, y = load_dataset_with_df()
preprocessor = DataPreprocessor()
X_processed = preprocessor.preprocess(df.iloc[:, :-1], fit=True)

X_train, X_test, y_train, y_test = train_test_split_data(X_processed, y)
y_train = y_train.astype(int)
y_test = y_test.astype(int)

print(f"✓ Data ready: {len(X_train)} training, {len(X_test)} testing\n")

# Train models
models = {
    'LR (No Balance)': LogisticRegressionModel(class_weight=None),
    'LR (Balanced)': LogisticRegressionModel(class_weight='balanced'),
    'Random Forest': RandomForestModel(n_estimators=100, max_depth=10)
}

results = {}
print("=" * 80)
print("TRAINING MODELS")
print("=" * 80)

for name, model in models.items():
    print(f"\n{name}...")
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    results[name] = metrics

# Compare
print("\n" + "=" * 80)
print("RESULTS COMPARISON")
print("=" * 80 + "\n")

data = []
for name, metrics in results.items():
    cm = metrics['confusion_matrix']
    data.append({
        'Model': name,
        'Accuracy': f"{metrics['accuracy']:.2%}",
        'Precision': f"{metrics['precision']:.2%}",
        'Recall': f"{metrics['recall']:.2%}",
        'F1': f"{metrics['f1_score']:.2%}",
        'TN': cm[0,0],
        'FP': cm[0,1],
        'FN': cm[1,0],
        'TP': cm[1,1],
    })

df_comp = pd.DataFrame(data)
print(df_comp.to_string(index=False))

# Analysis
print("\n" + "=" * 80)
print("KEY IMPROVEMENTS")
print("=" * 80)

lr_nobal = results['LR (No Balance)']
lr_bal = results['LR (Balanced)']
rf = results['Random Forest']

print(f"\n1️⃣  BASELINE (No Balancing):")
print(f"   Recall: {lr_nobal['recall']:.2%} - ⚠️  INSUFFICIENT (missing {lr_nobal['confusion_matrix'][1,0]} patients)")

print(f"\n2️⃣  WITH CLASS BALANCING:")
recall_improvement = (lr_bal['recall'] - lr_nobal['recall']) / lr_nobal['recall'] * 100
print(f"   Recall: {lr_bal['recall']:.2%} - Improvement: +{recall_improvement:.1f}%")

print(f"\n3️⃣  RANDOM FOREST:")
recall_improvement_rf = (rf['recall'] - lr_nobal['recall']) / lr_nobal['recall'] * 100
print(f"   Recall: {rf['recall']:.2%} - Improvement: +{recall_improvement_rf:.1f}%")
print(f"   Accuracy: {rf['accuracy']:.2%}")

print(f"\n🎯 RECOMMENDATION:")
if rf['recall'] > 0.75:
    print(f"   ✅ Random Forest achieves {rf['recall']:.2%} recall - acceptable for healthcare")
    print(f"   ✅ Use Random Forest as improved baseline for federated learning")
else:
    if lr_bal['recall'] > 0.70:
        print(f"   ✅ Balanced LR achieves {lr_bal['recall']:.2%} recall - better than baseline")
        print(f"   → Consider Random Forest for further improvements")
    else:
        print(f"   ⚠️  Current models still need improvement")
        print(f"   → Feature engineering or hyperparameter tuning needed")

print("\n" + "=" * 80)
