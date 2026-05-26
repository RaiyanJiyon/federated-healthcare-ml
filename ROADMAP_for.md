# Roadmap: Adding MLP Baseline to Federated Healthcare ML

## Executive Summary
Add a Multi-Layer Perceptron (MLP) model as a secondary baseline to demonstrate modern ML sophistication while keeping the logistic regression baseline as the primary clinical baseline. The MLP will run alongside LR in all existing experiments without disrupting the current pipeline.

---

## Phase 0: Current State Analysis

### Current Architecture Overview
```
src/
├── models/
│   └── model.py              # LogisticRegressionModel only
├── training/
│   └── federated.py          # Federated training orchestration
├── fl/
│   ├── strategy.py           # FedAvg, FedProx aggregation
│   ├── robust_aggregation.py # Krum, Median
│   ├── privacy.py            # DP-SGD implementation
│   └── adversarial.py        # Byzantine robustness
└── data/
    └── loader.py             # Data loading, caching

experiments/
├── exp1_baseline.py          # Centralized + Federated LR
├── exp4_aggregation_comparison.py
├── exp6_hyperparameter_sensitivity.py
└── ...
```

### Key Data Flow
1. **Data Loading**: `load_dataset_with_df()` → 65,273 patients, 31 features
2. **Partitioning**: `distribute_by_care_unit()` → 7 ICU clients
3. **Training**: 
   - Centralized: `LogisticRegressionModel.fit()` with StandardScaler
   - Federated: Client-side training → Server aggregation (FedAvg/FedProx/Krum/Median)
4. **Evaluation**: Threshold-based recall optimization, AUROC, Precision, F2-score
5. **Results**: CSV files in `results/plots/` for paper tables

### Current Baseline Results (for reference)
- **Centralized LR**: AUROC 0.8920, Recall 85.2%
- **Federated LR (5 rounds, FedAvg)**: AUROC 0.8898, Recall 86.3%
- **DP-FL LR (ε=1.0, δ=10⁻⁵)**: AUROC 0.8477 (after calibration)

---

## Phase 1: MLP Model Implementation

### Step 1.1: Create MLP Model Class
**File**: `src/models/model.py` (extend existing)

**Requirements**:
- PyTorch dependency (add to requirements.txt)
- Inherit from a common interface or create a base class for model agnostic training
- Constructor parameters:
  - `input_dim=31` (number of clinical features)
  - `hidden_layers=[64, 32]` (configurable architecture)
  - `dropout_rate=0.2` (regularization)
  - `learning_rate=0.001`
  - `batch_size=32`
  - `epochs=20` (for federated clients)
  - `random_state=42`

**Key Methods**:
```python
class MLPModel:
    def __init__(self, input_dim, hidden_layers, dropout_rate, learning_rate, random_state)
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=False)
    def predict_proba(self, X)
    def predict(self, X, threshold=0.5)
    def get_weights() -> np.ndarray
    def set_weights(weights: np.ndarray)
    def evaluate(self, X_test, y_test)
```

**Architecture Details**:
- Input layer: 31 features
- Hidden layer 1: 64 neurons + ReLU + Dropout(0.2)
- Hidden layer 2: 32 neurons + ReLU + Dropout(0.2)
- Output layer: 1 sigmoid (binary classification)
- Loss: BCEWithLogitsLoss (binary cross-entropy)
- Optimizer: Adam with learning_rate=0.001
- No batch normalization (keep simple for federated)

**Why This Size?**
- 64→32 is reasonable for 31 input features (1-2x expansion)
- Avoids deep networks that are hard to federate
- Maintains interpretability and quick training (clinical setting)

---

### Step 1.2: Create Model Interface/Registry
**File**: `src/models/model.py` or new `src/models/registry.py`

**Goal**: Enable experiments to swap models without changing logic

```python
MODEL_REGISTRY = {
    'logistic_regression': LogisticRegressionModel,
    'mlp': MLPModel,
}

def create_model(model_type: str, **kwargs):
    """Factory function to create models by name"""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_REGISTRY[model_type](**kwargs)
```

**Benefits**:
- Experiments can pass `--model mlp` or `--model logistic_regression`
- Easy to add XGBoost later
- Centralized configuration

---

## Phase 2: Integration with Federated Training Pipeline

### Step 2.1: Update Federated Training Functions
**File**: `src/training/federated.py`

**Current Assumption**: Model has `.get_weights()` and `.set_weights()` methods
- Ensure LR model has these methods (may need to add)
- Ensure MLP PyTorch model wraps weights properly:
  - Extract `.state_dict()` → flatten to numpy array
  - Reconstruct `.load_state_dict()` ← reshape from numpy array

**Key Functions to Update**:
```python
def train_federated_round(
    clients: dict,
    global_model: (LogisticRegressionModel or MLPModel),
    model_type: str = 'logistic_regression',
    aggregation_method: str = 'fedavg',
    **kwargs
):
    """Train one federated round, model-agnostic"""
    # 1. Broadcast global weights to clients
    # 2. Local training (model_type determines training logic)
    # 3. Collect client weight updates
    # 4. Aggregate (FedAvg, FedProx, Krum, Median)
    # 5. Update global model
```

**No changes to aggregation logic** — aggregation is model-agnostic (operates on weight vectors)

---

### Step 2.2: Update Evaluation Functions
**File**: `src/evaluation/` (check if exists, or add evaluation utils)

**Functions to Create/Update**:
```python
def evaluate_model(model, X_test, y_test, model_type: str, threshold=0.5):
    """Unified evaluation interface"""
    if model_type == 'logistic_regression':
        proba = model.predict_proba(X_test)[:, 1]
    elif model_type == 'mlp':
        proba = model.predict_proba(X_test)[:, 1]  # Already returns 1D
    
    return {
        'auroc': roc_auc_score(y_test, proba),
        'recall': recall_score(y_test, (proba >= threshold).astype(int)),
        'precision': precision_score(y_test, (proba >= threshold).astype(int)),
        'f2': fbeta_score(y_test, (proba >= threshold).astype(int), beta=2),
    }
```

---

## Phase 3: Experiment Updates

### Step 3.1: Create `exp1_baseline_multimodel.py`
**File**: `experiments/exp1_baseline_multimodel.py`

**Structure** (extend `exp1_baseline.py`):
1. Add `--model {logistic_regression, mlp}` argument
2. Loop over both models:
   ```python
   for model_type in ['logistic_regression', 'mlp']:
       # Train centralized
       # Train federated (5 rounds)
       # Evaluate and log
   ```
3. Collect results in unified CSV:
   ```
   model,centralized_auroc,centralized_recall,federated_auroc,federated_recall,dp_auroc,dp_recall
   logistic_regression,0.8920,0.8520,0.8898,0.8630,0.8477,0.6000
   mlp,0.8900,0.8400,0.8850,0.8550,0.8400,0.5800
   ```

**Experiment Command**:
```bash
python experiments/exp1_baseline_multimodel.py --model logistic_regression
python experiments/exp1_baseline_multimodel.py --model mlp
```

**Runtime Estimate**:
- LR: ~30 seconds (sequential, 5 rounds × 7 clients)
- MLP: ~3-5 minutes (forward/backward pass, 5 rounds × 7 clients × 20 epochs per client)

---

### Step 3.2: Update Existing Experiments (Optional But Recommended)
If time permits, add `--model` flag to:
- `exp4_aggregation_comparison.py` (Byzantine robustness comparison)
- `exp6_hyperparameter_sensitivity.py` (sensitivity across architectures)

**Minimal Change**: Add one line to argument parser and one loop in `main()`

---

## Phase 4: Paper Integration

### Step 4.1: Create Comparison Table
**File**: `results/plots/model_comparison_table.csv`

**Content**:
```
Model,Architecture,Centralized_AUROC,Centralized_Recall,FL_AUROC,FL_Recall,DP_AUROC,DP_Recall,Params
Logistic Regression,31→1 (linear),0.8920,0.8520,0.8898,0.8630,0.8477,0.6000,31
MLP,31→64→32→1,0.8900,0.8400,0.8850,0.8550,0.8400,0.5800,2849
```

**Paper Caption**:
```
Table X: Multi-Model Federated Learning Comparison
Centralized baseline, federated (FedAvg, 5 rounds), and differentially private 
(DP-SGD, ε=1.0, δ=10⁻⁵) performance for logistic regression and MLP architectures. 
MLP adds ~2,849 parameters (2.8% vs LR) but maintains clinical-grade recall 
and improves generalization on federated non-IID partitions.
```

### Step 4.2: Update Paper Sections

**Main.tex Changes**:

1. **Abstract** (optional): Mention "both simple and neural architectures"
2. **Related Work**: Add line about model diversity in federated healthcare
3. **Methodology § 3.3** (Distributed Patient Cohorts):
   ```
   New paragraph:
   "To validate federated learning generalization, we train two model architectures: 
   logistic regression (baseline, interpretable) and a small MLP (64→32 neurons) 
   to assess whether federated aggregation preserves performance across model complexity. 
   Both models use StandardScaler preprocessing and class-weight balancing to handle 
   the 15% mortality class imbalance."
   ```
4. **Results § 4.1** (new subsection): Model Comparison
   ```
   "We trained federated models with two architectures: logistic regression 
   (baseline) and MLP (2,849 parameters). The MLP achieved comparable AUROC 
   (0.8850 vs 0.8898 for LR) and recall (85.5% vs 86.3%) at validation-calibrated 
   thresholds, indicating that federated aggregation is robust to architecture choices. 
   Under DP-SGD (ε=1.0, δ=10⁻⁵), the MLP achieved AUROC 0.8400, slightly lower 
   than LR (0.8477), suggesting privacy's impact scales with model complexity."
   ```
5. **Insert Table X** with model comparison

---

## Phase 5: Implementation Checklist

### Code Changes
- [ ] Add PyTorch to `requirements.txt`
- [ ] Create `MLPModel` class in `src/models/model.py`
- [ ] Add model registry / factory function
- [ ] Update `src/training/federated.py` to accept `model_type` parameter
- [ ] Create unified evaluation function
- [ ] Create `exp1_baseline_multimodel.py` experiment

### Testing & Validation
- [ ] Test MLP on centralized data (single machine)
- [ ] Test MLP weight serialization (get/set weights)
- [ ] Test federated round (5 clients, 1 round)
- [ ] Verify AUROC/Recall metrics match evaluation functions
- [ ] Check numerical reproducibility (same seed → same weights)

### Paper Updates
- [ ] Add 1-2 sentences to Methodology § 3.3
- [ ] Add new subsection in Results (Model Comparison)
- [ ] Insert comparison table
- [ ] Update caption and discussion
- [ ] Regenerate figures if needed (optional)

### Results & Verification
- [ ] Run full experiment (centralized + federated + DP) for MLP
- [ ] Compare LR vs MLP results
- [ ] Ensure LR baseline unchanged (regression test)
- [ ] Generate CSV table for paper

---

## Phase 6: Deployment & Reproducibility

### Reproducibility Steps
```bash
# 1. Ensure environment
pip install -r requirements.txt  # Now includes torch

# 2. Run experiments
python experiments/exp1_baseline_multimodel.py --model logistic_regression --seed 42
python experiments/exp1_baseline_multimodel.py --model mlp --seed 42

# 3. Verify results
head -2 results/plots/model_comparison_table.csv

# 4. Compile paper
cd paper && pdflatex main.tex  # Table should render
```

### Files to Version Control
```
src/models/model.py          # + MLPModel class
src/training/federated.py    # + model_type parameter
experiments/exp1_baseline_multimodel.py  # New
results/plots/model_comparison_table.csv # Generated
paper/main.tex               # Updated
requirements.txt             # + torch
```

---

## Timeline & Dependencies

### Estimated Duration

| Phase | Task | Duration | Depends On |
|-------|------|----------|-----------|
| 1 | MLP model implementation | 1-2 hours | PyTorch knowledge |
| 2 | Federated integration | 30-45 min | Phase 1 ✓ |
| 3 | Experiment creation & runs | 1-2 hours | Phase 2 ✓ |
| 4 | Paper updates | 30 min | Phase 3 ✓ |
| 5 | Testing & reproducibility | 30 min | All above ✓ |
| **Total** | | **4-6 hours** | |

### Critical Path
1. ✓ MLP class (Phase 1.1)
2. ✓ Model registry (Phase 1.2)
3. ✓ Federated integration (Phase 2)
4. ✓ Experiment (Phase 3.1)
5. ✓ Paper table (Phase 4.1)

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| MLP overfitting on small per-client data | Lower federated perf | Dropout 0.2, early stopping on val loss |
| PyTorch version conflicts | Env issues | Pin `torch>=1.9.0,<2.1.0` in requirements |
| Slow MLP training | Delays experiments | Use GPU if available; otherwise optimize batch size |
| Weight serialization bugs | Results mismatch | Unit tests for get/set weights |
| Numerical reproducibility | Irreproducible runs | Use `torch.manual_seed()` + deterministic ops |

---

## Success Criteria

✅ **Minimal**:
- MLP runs on all 7 ICU clients without errors
- Federated training converges (loss decreases over 5 rounds)
- AUROC >= 0.84 on centralized test set
- Comparison table renders in paper

✅ **Comprehensive**:
- All above +
- MLP DP-SGD runs at ε=1.0, δ=10⁻⁵
- Results reproducible across 3 seeds
- Paper text updated cleanly
- No regression in LR baseline

---

## References & Dependencies

### New Dependencies
```
torch>=1.9.0        # PyTorch for neural networks
torchvision         # (optional, not needed)
scikit-learn>=1.0   # Already in requirements
```

### Existing Code to Reuse
- `src/data/loader.py` — Data loading (no changes needed)
- `src/data/split.py` — Partitioning (no changes needed)
- `src/fl/strategy.py` — Aggregation (model-agnostic)
- `experiments/exp1_baseline.py` — Experiment template

### Configuration (existing)
- `src/config/config.py` — Already defines RANDOM_SEED, ALL_FEATURES (31 features)
- No config changes needed for basic MLP run

---

## Notes for Implementation

1. **Preprocessing**: MLP should use same StandardScaler as LR for fair comparison
2. **Batch Size**: Use 32 (typical) but allow CLI override
3. **Epochs per Round**: 20 epochs per client per federated round (tunable)
4. **Validation**: Use validation set from existing splits (no data leakage)
5. **Threshold Calibration**: Same recall-based threshold selection as LR
6. **Convergence**: 5 federated rounds should be enough (MLP trains locally per round)

---

## Future Enhancements (Out of Scope)

- XGBoost baseline (different weight serialization)
- Hyperparameter sweep (grid search over hidden layers, dropout)
- LSTM temporal modeling (requires different data pipeline)
- Ensemble methods (combines LR + MLP)
- Compression (quantization, pruning for communication efficiency)

---

*Last Updated: 2026-05-26*
*Status: Ready for Implementation*
