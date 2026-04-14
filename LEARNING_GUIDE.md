# 🎓 Complete Codebase Learning Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [Learning Phases](#learning-phases)
3. [Codebase Structure](#codebase-structure)
4. [File-by-File Guide](#file-by-file-guide)
5. [How to Run](#how-to-run)
6. [Key Concepts Explained](#key-concepts-explained)
7. [Learning Tips](#learning-tips)
8. [Self-Assessment Questions](#self-assessment-questions)

---

## Project Overview

### What is This Project?
**Federated Learning for Healthcare Diabetes Prediction** - A framework that trains machine learning models across multiple clients (simulating different healthcare facilities) WITHOUT sharing raw patient data.

### The Problem
- Healthcare facilities have patient data but can't share it due to privacy regulations
- Need a collaborative model that learns from multiple facilities
- Classical ML: All data in one place ❌ (privacy risk)
- This project: Keep data local, share only model weights ✅

### The Solution
Federated Learning:
```
Client 1 (Hospital A) ──local training──┐
Client 2 (Hospital B) ──local training──┼──→ Server ──aggregate──→ Better Model
Client 3 (Hospital C) ──local training──┤
```

### Achievement
✅ **87.04% recall** - Correctly identifies diabetic patients with healthcare-grade safety
- Baseline: 70.37% recall (dangerous, misses 16 patients)
- Optimized: 87.04% recall (safe, misses only 7 patients)

---

## Learning Phases

### Phase 1: Understand the Problem (30 minutes)
**Goal:** Understand WHAT this project does and WHY

**Read these files:**
1. [README.md](README.md) - Project overview
2. [PROJECT_STATUS.md](PROJECT_STATUS.md) - Results achieved
3. `data/raw/diabetes.csv` - Actual dataset (open in Excel/text editor)

**Key Questions to Answer:**
- What are the 8 input features?
- What is the output (target variable)?
- Why is recall more important than accuracy here?
- How many patients in the dataset?

**Estimated Time:** 30 mins

---

### Phase 2: Understand Core Building Blocks (1 hour)
**Goal:** Understand HOW data flows through the system

**Read in this order:**

#### 1. **Configuration** - `src/config/config.py`
```python
# All settings in ONE place
DATASET_PATH = "data/raw/diabetes.csv"
NUM_CLIENTS = 5          # Simulate 5 healthcare facilities
NUM_ROUNDS = 10          # Communication rounds
LEARNING_RATE = 0.01
BATCH_SIZE = 32
```
**What to understand:**
- This file controls EVERYTHING
- Modify here to change project behavior

#### 2. **Data Loading** - `src/data/loader.py`
```python
def load_dataset_with_df():
    # Loads CSV file
    # Returns: DataFrame, features (X), target (y)
```
**What to understand:**
- Where data comes from
- Data shape and format

#### 3. **Data Preprocessing** - `src/data/preprocess.py`
```python
class DataPreprocessor:
    def preprocess(self, X):
        # Normalize features (scale to 0-1)
        # Handle missing values
        return X_cleaned
```
**What to understand:**
- Why preprocessing needed
- What normalization does

#### 4. **Models** - `src/models/model.py`
```python
class LogisticRegressionModel:
    def train(self, X, y):
        # Train model
    
    def predict(self, X):
        # Make predictions
```
**What to understand:**
- 3 model types available: LogisticRegression, RandomForest, XGBoost
- Every model has `train()` and `predict()` methods

**Estimated Time:** 1 hour

---

### Phase 3: Run Your First Experiment (10 minutes)
**Goal:** Execute code and see results

```bash
# Activate environment
source venv/bin/activate

# Run baseline experiment (simplest)
python experiments/exp1_baseline.py
```

**What happens:**
1. Loads diabetes data
2. Splits into train/test
3. Trains a single model (no federated learning)
4. Prints metrics: accuracy, precision, recall, F1
5. Saves results to `results/` directory

**Expected output shows:**
```
EXPERIMENT 1: CENTRALIZED BASELINE TRAINING
Accuracy: XX%
Precision: XX%
Recall: XX%
F1-Score: XX%
```

**Estimated Time:** 10 mins

---

### Phase 4: Understand Centralized Training (30 minutes)
**Goal:** Learn the SIMPLEST form of ML (no federated learning)

**File:** `src/training/centralized.py`

**Flow:**
```
Load Data
    ↓
Split (80% train, 20% test)
    ↓
Preprocess Features
    ↓
Train Model
    ↓
Evaluate (Accuracy, Recall, etc.)
```

**Code walkthrough:**
```python
# 1. Load data
df, X, y = load_dataset_with_df()

# 2. Preprocess
preprocessor = DataPreprocessor()
X_processed = preprocessor.preprocess(X)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split_data(X_processed, y)

# 4. Train
model = LogisticRegressionModel()
model.train(X_train, y_train, verbose=True)

# 5. Evaluate
predictions = model.predict(X_test)
accuracy = calculate_metrics(y_test, predictions)
```

**Read:** [src/training/centralized.py](src/training/centralized.py)

**Estimated Time:** 30 mins

---

### Phase 5: Understand Federated Learning (1 hour)
**Goal:** Learn how multiple clients collaborate WITHOUT sharing data

**The 3 Core Components:**

#### 1. **Client** - `src/fl/client.py`
```python
class FlowerClient:
    def fit(self, parameters, config):
        # LOCAL training - each client trains independently
        # Uses only its own data
        return updated_parameters
    
    def evaluate(self, parameters, config):
        # Evaluate on local test set
```

**What to understand:**
- Each client is INDEPENDENT
- Trains on its own data
- Shares only MODEL WEIGHTS (not raw data)

#### 2. **Server** - `src/fl/server.py`
```python
def start_fl_server():
    # 1. Collect model weights from all clients
    # 2. Aggregate (combine) them
    # 3. Send back updated model
    # 4. Repeat for N rounds
```

**What to understand:**
- Server is DUMB (no data, only coordinates)
- Aggregation = simple average of weights

#### 3. **Strategy** - `src/fl/strategy.py`
```python
class FedAvg:
    # Weighted average of all client weights
    # Simple formula: global_model = sum(client_weights) / num_clients
```

**What to understand:**
- Different strategies to combine weights
- FedAvg = simple average (most popular)
- FedProx = slightly more complex variant

**Visual Flow:**
```
Round 1:
  Client 1,2,3,4,5 ──→ Train Locally ──→ Send Weights ──→ Server
  Server ──→ Average Weights ──→ Send Back New Model

Round 2:
  Client 1,2,3,4,5 ──→ Train with New Model ──→ Send Weights ──→ Server
  Server ──→ Average Weights ──→ Send Back New Model

... (Repeat NUM_ROUNDS times)
```

**Run experiment:**
```bash
python experiments/exp3_clients.py
```

**Read files in order:**
1. [src/fl/client.py](src/fl/client.py) - Understand client
2. [src/fl/server.py](src/fl/server.py) - Understand server
3. [src/fl/strategy.py](src/fl/strategy.py) - Understand aggregation

**Estimated Time:** 1 hour

---

### Phase 6: Understand Optimization (30 minutes)
**Goal:** Learn how to improve model accuracy from 70% → 87%

**File:** `src/utils/feature_engineering.py`

**Three techniques used:**

1. **Threshold Adjustment** (Most Important ⭐)
   ```python
   # Instead of: if probability > 0.5 → positive
   # Use: if probability > 0.30 → positive
   # Result: Catch more diabetic patients (higher recall)
   ```

2. **Feature Engineering** (+11 features)
   ```python
   # Interaction features: Glucose × BMI
   # Polynomial features: Glucose²
   # Ratio features: Glucose / Insulin
   ```

3. **Hyperparameter Tuning**
   ```python
   # Learning rate, batch size, iterations
   # class_weight='balanced' for imbalanced data
   ```

**Comparison:**
| Method | Recall | Accuracy | Missed Patients |
|--------|--------|----------|-----------------|
| Baseline | 70.37% | 73.38% | 16 patients ❌ |
| + Features | 75.93% | 73.38% | 13 patients ❌ |
| + Threshold | 88.89% | 70.13% | 6 patients ✅ |
| + Both | 87.04% | 69.48% | 7 patients ✅ |

**Run experiment:**
```bash
python experiments/exp2_optimized.py
```

**Estimated Time:** 30 mins

---

### Phase 7: Advanced Topics (Optional)
**Goal:** Learn privacy & security in federated learning

#### Option A: Privacy (Differential Privacy)
**File:** `src/fl/privacy.py`

**What to understand:**
- Add noise to model weights
- Prevents attackers from reverse-engineering patient data
- Trade-off: Slightly lower accuracy for stronger privacy

**Run:**
```bash
python experiments/exp7_differential_privacy.py
```

#### Option B: Security (Adversarial Robustness)
**Files:** 
- `src/fl/adversarial.py` - Byzantine attack setup
- `src/fl/robust_aggregation.py` - Defense mechanisms

**What to understand:**
- Byzantine attack: Malicious clients send bad weights
- Defense: Use Median or Krum aggregation (robust to attacks)
- Comparison: FedAvg vs Median vs Krum

**Run:**
```bash
python experiments/exp8_adversarial_robustness.py
```

#### Option C: Scalability
**File:** `src/evaluation/visualize.py`

**What to understand:**
- How system performs with more clients
- Communication costs vs accuracy

**Run:**
```bash
python experiments/exp9_scalability_analysis.py
```

**Estimated Time:** 30 mins each

---

## Codebase Structure

### Directory Layout
```
federated-healthcare-ml/
├── src/                          # Main source code
│   ├── config/
│   │   └── config.py            ⭐ ALL SETTINGS HERE
│   ├── data/
│   │   ├── loader.py            # Load CSV data
│   │   ├── preprocess.py         # Clean & normalize
│   │   └── split.py             # Train/test split
│   ├── models/
│   │   └── model.py             # 3 model types
│   ├── fl/                      # Federated learning
│   │   ├── client.py            # Client trains locally
│   │   ├── server.py            # Server aggregates
│   │   ├── strategy.py          # Aggregation methods
│   │   ├── privacy.py           # Differential privacy
│   │   ├── adversarial.py       # Byzantine attacks
│   │   └── robust_aggregation.py # Defense mechanisms
│   ├── training/
│   │   ├── centralized.py       # Single machine training
│   │   └── federated.py         # Multi-client training
│   ├── evaluation/
│   │   ├── metrics.py           # Calculate accuracy, recall, F1
│   │   └── visualize.py         # Plot results
│   └── utils/
│       ├── feature_engineering.py # Create new features
│       └── logger.py             # Log results
├── experiments/                  # Test different configurations
│   ├── exp1_baseline.py         # Centralized baseline
│   ├── exp2_optimized.py        # Optimized model
│   ├── exp2_noniid.py           # Non-IID data distribution
│   ├── exp3_clients.py          # Federated learning
│   ├── exp4_aggregation_comparison.py
│   ├── exp5_dropout_simulation.py
│   ├── exp6_hyperparameter_sensitivity.py
│   ├── exp7_differential_privacy.py
│   ├── exp8_adversarial_robustness.py
│   └── exp9_scalability_analysis.py
├── data/
│   ├── raw/
│   │   └── diabetes.csv         # Dataset (768 rows, 8 features)
│   └── processed/               # Preprocessed data (auto-created)
├── results/                     # Experiment outputs
│   ├── logs/
│   ├── plots/
│   └── *.json
├── notebooks/
│   └── exploration.ipynb        # Jupyter notebook
├── paper/
│   └── draft.md                 # Research paper draft
├── requirements.txt             # Dependencies
├── run.py                       # Main entry point
├── README.md                    # Quick start
├── PROJECT_STATUS.md            # What was achieved
└── LEARNING_GUIDE.md            # This file!
```

---

## File-by-File Guide

### Must-Read Files (Priority Order)

| Priority | File | Lines | Purpose | Time |
|----------|------|-------|---------|------|
| 🔴 Critical | `src/config/config.py` | 40 | All project settings | 5 mins |
| 🔴 Critical | `src/models/model.py` | 500 | Model implementations | 15 mins |
| 🔴 Critical | `src/fl/client.py` | 80 | Client training | 10 mins |
| 🔴 Critical | `src/fl/server.py` | 120 | Server aggregation | 10 mins |
| 🟠 Important | `src/data/loader.py` | 30 | Data loading | 5 mins |
| 🟠 Important | `src/data/preprocess.py` | 50 | Data cleaning | 5 mins |
| 🟠 Important | `src/training/centralized.py` | 80 | Simple training | 10 mins |
| 🟡 Useful | `src/fl/strategy.py` | 100 | Aggregation strategies | 10 mins |
| 🟡 Useful | `src/evaluation/metrics.py` | 100 | Performance metrics | 10 mins |
| 🟢 Optional | `src/fl/privacy.py` | 150 | Differential privacy | 15 mins |
| 🟢 Optional | `src/fl/adversarial.py` | 400 | Byzantine robustness | 20 mins |

### Module Descriptions

#### `src/config/` - Configuration
```python
# Central place for ALL settings
DATASET_PATH          # Where to find data
NUM_CLIENTS           # Number of federated clients
NUM_ROUNDS            # Communication rounds
LEARNING_RATE         # Training speed
BATCH_SIZE            # Samples per batch
MODEL_TYPE            # Which model to use
NON_IID               # Non-identical data distribution
DIRICHLET_ALPHA       # How different client data is
```
**Why important:** Change ONE variable here, entire behavior changes

#### `src/data/` - Data Handling
- **loader.py**: Read CSV → DataFrame
- **preprocess.py**: Normalize features (scale 0-1)
- **split.py**: Train/test split & non-IID distribution

**Data flow:**
```
diabetes.csv → loader → preprocess → split → ready for training
```

#### `src/models/` - Model Definitions
```python
LogisticRegressionModel     # Simple, interpretable
RandomForestModel           # Non-linear, handles interactions
XGBoostModel               # Gradient boosting, best for imbalanced data
```
**Common interface:**
```python
model = SomeModel()
model.train(X_train, y_train)
predictions = model.predict(X_test)
```

#### `src/fl/` - Federated Learning
- **client.py**: Individual training
- **server.py**: Centralized aggregation
- **strategy.py**: How to combine weights
- **privacy.py**: Add differential privacy noise
- **adversarial.py**: Simulate Byzantine attacks
- **robust_aggregation.py**: Defend against attacks

#### `src/training/` - Training Pipelines
- **centralized.py**: Single model on all data (baseline)
- **federated.py**: Multiple clients + server

#### `src/evaluation/` - Metrics & Visualization
- **metrics.py**: Calculate accuracy, precision, recall, F1
- **visualize.py**: Plot confusion matrix, training curves

#### `src/utils/` - Helper Functions
- **feature_engineering.py**: Create interaction features, polynomial features
- **logger.py**: Save results to file

#### `experiments/` - Runnable Scripts
Each file tests ONE thing:
- exp1: Centralized baseline
- exp2: Optimized threshold
- exp3: Federated learning
- exp4-9: Variations (dropout, privacy, attacks, scalability)

---

## How to Run

### Quick Start (5 minutes)

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Run simplest experiment
python experiments/exp1_baseline.py

# 3. Check results
ls results/
cat results/logs/baseline_*.log
```

### Run All Experiments

```bash
# Core federated learning experiments
python experiments/exp1_baseline.py              # Centralized
python experiments/exp2_optimized.py             # Optimized
python experiments/exp3_clients.py               # Federated
python experiments/exp4_aggregation_comparison.py # Compare strategies
python experiments/exp5_dropout_simulation.py     # Test robustness
python experiments/exp6_hyperparameter_sensitivity.py # Tune parameters

# Advanced experiments
python experiments/exp7_differential_privacy.py  # Privacy
python experiments/exp8_adversarial_robustness.py # Security
python experiments/exp9_scalability_analysis.py # Scalability
```

### Or Run Everything

```bash
python run.py  # Integrated pipeline
```

### Modify and Re-run

```bash
# 1. Edit config
nano src/config/config.py
# Change: NUM_CLIENTS = 10 (was 5)
# Change: NUM_ROUNDS = 20 (was 10)

# 2. Run experiment
python experiments/exp3_clients.py

# 3. See how results changed
```

### Check Results

```bash
# All outputs in results/ directory
ls results/
ls results/logs/          # Text logs
ls results/plots/         # Visualizations (PNG, CSV)
ls results/*.json         # Raw data

# View latest log
cat results/logs/latest.log

# View results as CSV
head results/plots/results.csv
```

---

## Key Concepts Explained

### 1. **Federated Learning**
**What:** Train model without sharing data
**Why:** Privacy - patient data stays local
**How:** 
- Each client trains locally
- Sends model weights to server
- Server averages weights
- Repeat

**Real-world example:**
```
Hospital A (Boston)     ──Local Training──┐
Hospital B (NYC)        ──Local Training──┼──→ Central Coordinator ──→ Better Model
Hospital C (LA)         ──Local Training──┤   (Averages Weights)
```

### 2. **Non-IID Data Distribution**
**IID (Identical & Independent):**
```
All clients have same data: [Patient profiles A, B, C, D...]
```

**Non-IID (Different & Correlated):**
```
Hospital A: [Mostly young patients]
Hospital B: [Mostly elderly patients]
Hospital C: [Mostly urban patients]
Different demographics, different patterns
```

**Why it matters:** Non-IID is REALISTIC but HARDER to train

### 3. **Recall vs Accuracy**

**Accuracy = Overall Correctness**
```
If 100 patients: 85 correct predictions
Accuracy = 85%
```

**Recall = Correctly Identified Positives**
```
If 50 patients have diabetes:
If we find 44 of them
Recall = 44/50 = 88%

Miss 6 diabetic patients = DANGEROUS in healthcare!
```

**Why Recall matters in healthcare:**
- ❌ Miss diabetic patient → patient dies
- ⚠️ False alarm → patient gets follow-up test

### 4. **Class Imbalance**
**Normal data:** 50% positive, 50% negative
**Imbalanced:** 10% positive, 90% negative (like diabetes dataset)

```
If we predict "No Diabetes" for EVERYONE:
Accuracy = 90% ✓ (looks good!)
Recall = 0% ❌ (missed ALL diabetic patients!)
```

**Solution:** Weight classes differently
```python
class_weight = 'balanced'  # Penalize missing positives
```

### 5. **Aggregation Methods**

**FedAvg (Simple Average)**
```python
global_weights = (w1 + w2 + w3 + w4 + w5) / 5
```
Fast but vulnerable to poisoning attacks

**Median (Robust)**
```python
global_weights = median(w1, w2, w3, w4, w5)
Takes middle weight, ignores outliers
```
Slower but handles Byzantine attacks

**Krum (Byzantine-Resilient)**
```python
global_weights = weights_closest_to_others
Finds weight vector closest to others
```
Best for Byzantine robustness

### 6. **Differential Privacy**
**Problem:** Can attacker reverse-engineer patient data from model?

**Solution:** Add noise to model weights
```python
noisy_weights = original_weights + random_noise
```

**Trade-off:**
- More noise → More privacy (but lower accuracy)
- Less noise → Less privacy (but higher accuracy)

### 7. **Byzantine Attacks**
**Attack:** Malicious client sends wrong weights
```
Client 1: [0.5, 0.3, 0.2] ✓ Good
Client 2: [0.4, 0.35, 0.25] ✓ Good
Client 3: [100, 100, 100] ❌ Attack! (sends huge values)
Client 4: [0.45, 0.3, 0.25] ✓ Good
Client 5: [-100, -100, -100] ❌ Attack! (sends huge negative)

FedAvg: (0.5+0.4+100+0.45-100)/5 = breaks! ❌
Median: [0.45, 0.3, 0.25] = ignores outliers ✓
```

---

## Learning Tips

### 1. **Run Before Reading**
- Execute `python experiments/exp1_baseline.py` FIRST
- See what output looks like
- THEN read the code
- Concrete examples help understanding

### 2. **Read Code with Questions**
```python
# Instead of just reading:
# model = LogisticRegressionModel()

# Ask yourself:
# Q1: What's being created?
# Q2: What parameters does it take?
# Q3: What methods does it have?
# A: Read docstring and next few lines
```

### 3. **Modify One Thing at a Time**
```bash
# Change 1 setting only
# Run experiment
# See result
# Change another setting
# Run again
# Compare results
```

### 4. **Use Comments as Guide**
All code has comments explaining WHAT and WHY. Read them!

### 5. **Keep Experiment Window Open**
```bash
# Terminal 1: Run experiment
python experiments/exp1_baseline.py

# Terminal 2: Open another file to read
nano src/models/model.py

# See execution + code side-by-side
```

### 6. **Track Key Numbers**
```
Baseline Recall: 70.37%
Optimized Recall: 87.04%
Improvement: +16.67%

Missed Patients:
Before: 16 out of 54 (30% miss rate)
After: 7 out of 54 (13% miss rate)
```

### 7. **Build Mental Model**
```
START → Data Loading → Preprocessing → Model Training → Evaluation → Results ✓

Each arrow = One module to understand
```

---

## Self-Assessment Questions

### After Phase 1 (Problem Understanding)
- [ ] What disease does this project predict?
- [ ] How many input features?
- [ ] What is the main output?
- [ ] Why is recall important in healthcare?
- [ ] What was the baseline recall? Optimized recall?

### After Phase 2 (Building Blocks)
- [ ] Where are all settings defined?
- [ ] How does data load from CSV?
- [ ] What does preprocessing do?
- [ ] Name 3 model types available
- [ ] What methods do all models have?

### After Phase 3 (First Experiment)
- [ ] Which experiment is simplest?
- [ ] What does it do step-by-step?
- [ ] Where are results saved?
- [ ] What metrics are calculated?
- [ ] Can you modify one setting and re-run?

### After Phase 4 (Centralized Training)
- [ ] Explain the training pipeline
- [ ] What's the difference between train/test split?
- [ ] Why preprocess data?
- [ ] How does train() method work?
- [ ] What does predict() return?

### After Phase 5 (Federated Learning)
- [ ] What's federated learning?
- [ ] Why keep data local?
- [ ] What's a client?
- [ ] What's a server?
- [ ] How are weights aggregated?
- [ ] Explain FedAvg in one sentence
- [ ] What happens in each round?

### After Phase 6 (Optimization)
- [ ] What's the most impactful optimization?
- [ ] How does threshold adjustment work?
- [ ] What are engineered features?
- [ ] Why does Glucose×BMI make sense?
- [ ] Trade-offs between recall and accuracy?

### After Phase 7 (Advanced Topics)
- [ ] What's differential privacy?
- [ ] What's a Byzantine attack?
- [ ] How does Median defend?
- [ ] When would you use Krum?
- [ ] How does scalability affect accuracy?

---

## Next Steps

### 🎯 Beginner Path (1-2 days)
1. Read this guide completely
2. Run exp1_baseline.py
3. Read src/config/config.py
4. Read src/models/model.py
5. Run exp3_clients.py
6. Answer Phase 1-3 questions

### 🚀 Intermediate Path (3-5 days)
1. Complete Beginner Path
2. Understand all 9 experiments
3. Modify config and re-run
4. Answer all self-assessment questions
5. Draw your own architecture diagram

### 💪 Advanced Path (1-2 weeks)
1. Complete Intermediate Path
2. Modify source code (add new feature, change algorithm)
3. Read papers related to federated learning
4. Implement new aggregation strategy
5. Write your own experiment

### 🏆 Expert Path (2-4 weeks)
1. Complete Advanced Path
2. Extend project with new dataset
3. Implement new privacy mechanism
4. Contribute improvements
5. Publish results

---

## Common Mistakes to Avoid

❌ **Mistake 1: Reading code without running**
✅ **Fix:** Run experiments first, then read code

❌ **Mistake 2: Changing multiple settings at once**
✅ **Fix:** Modify one variable, observe effect

❌ **Mistake 3: Not checking results/ directory**
✅ **Fix:** Always check logs and outputs after running

❌ **Mistake 4: Skipping config.py**
✅ **Fix:** config.py is the KEY to understanding

❌ **Mistake 5: Confusing accuracy with recall**
✅ **Fix:** Recall = % of positives found (critical in healthcare)

❌ **Mistake 6: Not reading comments in code**
✅ **Fix:** Comments explain WHAT and WHY

---

## Quick Reference

### Commands
```bash
# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run experiment
python experiments/exp1_baseline.py

# Run all
python run.py

# Check results
ls results/logs/
cat results/logs/latest.log
```

### Files to Read
```
Quick Overview:
  README.md
  PROJECT_STATUS.md

Settings:
  src/config/config.py

Core ML:
  src/models/model.py
  src/data/loader.py
  src/data/preprocess.py

Federated Learning:
  src/fl/client.py
  src/fl/server.py
  src/fl/strategy.py

Advanced:
  src/fl/privacy.py
  src/fl/adversarial.py
```

### Key Numbers
```
Dataset: 768 patients, 8 features, binary target
Baseline: 70.37% recall (dangerous)
Optimized: 87.04% recall (safe)
Clients: 5 (configurable)
Rounds: 10 (configurable)
Features: [Pregnancies, Glucose, BP, Skin, Insulin, BMI, DPF, Age]
```

---

## Additional Resources

### For Concepts
- Federated Learning: Read `src/fl/client.py` + `server.py` comments
- Differential Privacy: Read `src/fl/privacy.py` comments
- Byzantine Robustness: Read `src/fl/adversarial.py` comments

### For Implementation
- Run experiments to see code in action
- Modify `src/config/config.py` to change behavior
- Check `results/` for outputs

### For Research
- See `paper/draft.md` for research write-up
- Compare exp1 (baseline) vs exp2 (optimized) vs exp3 (federated)

---

## Getting Help

### Problem: Code won't run
1. Check: `python -m venv venv activated?
2. Check: `pip install -r requirements.txt` run?
3. Check: `data/raw/diabetes.csv` exists?
4. Check: Error message - read it carefully!

### Problem: Don't understand X
1. Find the file from directory structure
2. Read the file with comments
3. Run the experiment that uses it
4. Check results/logs for output
5. Modify one parameter, re-run

### Problem: Want to add feature
1. Start with `src/config/config.py`
2. Look at similar experiment
3. Copy and modify slowly
4. Test with small changes
5. Expand gradually

---

**Happy Learning! 🚀**

Start with Phase 1 and work your way up. You've got this!
