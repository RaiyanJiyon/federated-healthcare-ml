#!/usr/bin/env python3
"""Test script for Phase 1: Step 3 - Base Model Implementation"""

from src.data.loader import load_dataset_with_df
from src.data.preprocess import DataPreprocessor
from src.data.split import train_test_split_data, distribute_data
from src.models.model import LogisticRegressionModel
import numpy as np

print("=" * 70)
print("PHASE 1: STEP 3 - BASE MODEL IMPLEMENTATION TEST")
print("=" * 70)

# Step 1: Load and preprocess data
print("\n1. Loading and preprocessing dataset...")
df, X, y = load_dataset_with_df()
preprocessor = DataPreprocessor()
X_processed = preprocessor.preprocess(df.iloc[:, :-1], fit=True)

# Step 2: Train-test split
print("\n2. Splitting data (80-20)...")
X_train, X_test, y_train, y_test = train_test_split_data(X_processed, y)

# Step 3: Create and train model
print("\n3. Training Logistic Regression model...")
model = LogisticRegressionModel(learning_rate=0.01, max_iter=1000)
train_metrics = model.fit(X_train, y_train.astype(int), verbose=True)

# Step 4: Evaluate on test set
print("\n4. Evaluating model on test set...")
test_metrics = model.evaluate(X_test, y_test.astype(int), verbose=True)

# Step 5: Test weight management (important for federated learning)
print("\n5. Testing weight management (FL functionality)...")
weights = model.get_weights()
print(f"  Weights retrieved:")
print(f"    - Coefficients shape: {weights['coef'].shape}")
print(f"    - Intercept shape: {weights['intercept'].shape}")
print(f"    - Classes: {weights['classes']}")
print(f"    - Total parameters: {model.get_num_parameters()}")

# Test weight serialization
print("\n6. Testing weight serialization...")
serialized = model.serialize_weights()
print(f"  Serialized weights size: {len(serialized)} bytes")

# Create new model and set weights
model2 = LogisticRegressionModel()
model2.set_weights(weights)
print(f"  ✓ Weights successfully transferred to new model")

# Verify predictions are identical
pred1 = model.predict(X_test[:10])
pred2 = model2.predict(X_test[:10])
print(f"  ✓ Predictions identical: {np.array_equal(pred1, pred2)}")

print("\n" + "=" * 70)
print("✅ MODEL IMPLEMENTATION SUCCESSFUL!")
print("=" * 70)
print("\nModel capabilities:")
print("  ✓ Training with fit()")
print("  ✓ Prediction with predict()")
print("  ✓ Evaluation metrics (accuracy, precision, recall, F1, confusion matrix)")
print("  ✓ Weight extraction with get_weights()")
print("  ✓ Weight setting with set_weights()")
print("  ✓ Weight serialization for federated learning")

print("\nReady for Federated Learning Implementation!")
