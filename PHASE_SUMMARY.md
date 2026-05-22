# MIMIC-IV Federated Healthcare ML - Complete Phase Summary

## PROJECT OVERVIEW
**Goal**: Implement federated learning on MIMIC-IV clinical data to validate Q1/Q2 journal submission (Phase 0→1→2)
**Repository**: `/home/raiyanjiyon/Projects/federated-healthcare-ml`
**Python**: 3.14.4, venv at `.venv/`
**GCP Project**: `mimic-iv-research-496704` (PhysioNet credentials verified, billing enabled)

---

## PHASE 0: COHORT SETUP & BASELINE ✅ COMPLETED

### 0.1 BigQuery Infrastructure
- **Datasets**: 
  - `physionet-data:mimiciv_3_1_icu` (ICU admissions)
  - `physionet-data:mimiciv_3_1_hosp` (hospital admissions, ICD codes)
- **Key Schema Discovery**:
  - `chartevents`: Uses `(subject_id, hadm_id, charttime, itemid, valuenum)` — NO `stay_id`
  - `labevents`: Uses `(subject_id, hadm_id, charttime, itemid, valuenum)` — NO `stay_id`
  - Only `icustays` has `stay_id` (relates back to `hadm_id`)
- **Feature Window**: First 24 hours of ICU stay (intime to intime+24h) to prevent leakage

### 0.2 Clinical Cohort (65,273 patients)
**Inclusion Criteria**:
- First ICU stay only
- Age ≥18 years
- Length of stay ≥4 hours
- Mortality flag: binary (0=alive, 1=dead)

**Target Distribution**: 
- 7,066 deaths / 58,207 survivors = 10.8% mortality

**Data Splits** (stratified, no patient overlap):
- Training: 45,691 patients (70%)
- Validation: 9,791 patients (15%)
- Test: 9,791 patients (15%)

### 0.3 Clinical Features (31 of 32 available)
**Demographics** (4):
- `age` (years at admission)
- `gender_M` (binary: male=1, female=0)
- `admission_emergency` (binary: emergency=1, scheduled=0)
- `insurance_medicare` (binary: medicare=1, other=0)
- ❌ `admission_type_emergency`: Missing from final query

**Vitals from chartevents** (14):
- `heart_rate_mean`, `heart_rate_min`, `heart_rate_max`
- `sbp_mean`, `sbp_min`
- `mbp_mean`, `mbp_min`
- `resp_rate_mean`, `resp_rate_max`
- `temperature_mean`
- `spo2_mean`, `spo2_min`

**Labs from labevents** (12):
- `glucose`, `creatinine`, `bun`, `sodium_min`, `sodium_max`
- `potassium`, `hemoglobin`, `wbc`, `bicarbonate`, `lactate`, `bilirubin`

**Clinical Scores** (3 — currently hardcoded 0):
- `sofa`, `sapsii`, `charlson` (TODO: implement in Phase 2)

### 0.4 BigQuery Implementation
**File**: `src/data/loader.py`
**Key Functions**:
```python
load_from_bigquery(billing_project)  # Execute 18-CTE SQL, return DataFrame
load_dataset_with_df(use_cache=True, billing_project=None)  # Returns (df, X, y)
load_dataset(use_cache, billing_project)  # Returns (X, y) arrays only
```

**SQL Features**:
- 18 CTEs: `first_icu`, `cohort_base`, 12 vitals CTEs, 4 lab CTEs
- Caching: Checks `data/cache/mimic_iv_cohort.csv` first, falls back to BigQuery
- Fallback: If BigQuery fails, uses synthetic data

### 0.5 Baseline Results ✅
**Centralized Logistic Regression** (sklearn, max_iter=1000, StandardScaler):
- **Train AUROC**: 0.8906
- **Validation AUROC**: 0.8846
- **Test AUROC**: 0.8887 ✅ (exceeds 0.75 target)

---

## PHASE 1: FEDERATED LEARNING ON CARE UNITS ✅ COMPLETED

### 1.1 Care-Unit Partitioning
**File**: `src/data/split.py`

**Primary ICU Units** (7 clients from 65,273 patients):
1. **Medical ICU (MICU)**: 8,791 patients, 1,410 deaths (16.0% mortality)
2. **Cardiac Vascular ICU (CVICU)**: 8,118 patients, 265 deaths (3.3% mortality)
3. **Medical/Surgical ICU (MICU/SICU)**: 6,983 patients, 957 deaths (13.7% mortality)
4. **Surgical ICU (SICU)**: 6,372 patients, 754 deaths (11.8% mortality)
5. **Trauma Surgical ICU (TSICU)**: 5,458 patients, 569 deaths (10.4% mortality)
6. **Coronary Care Unit (CCU)**: 5,113 patients, 643 deaths (12.6% mortality)
7. **Neuro Surgical ICU (Neuro SICU)**: 907 patients, 256 deaths (28.2% mortality)

**Skipped Units** (<100 patients each):
- Neuro Intermediate: 3,035 patients
- Neuro Stepdown: 784 patients
- Others: <100 each

**Federated Training Data**: 41,742 patients (from Phase 0 training set)

**Key Function**:
```python
distribute_by_care_unit(X, y, care_units, min_patients_per_unit=100)
# Returns: {unit_name: (X_unit, y_unit)} for each of 7 clients
```

### 1.2 Federated Learning Framework
**File**: `src/training/federated.py`

**FederatedTrainer Class** (Manual FedAvg, no Flower):
```python
__init__(clients, val_data, test_data, num_rounds=10, learning_rate=0.01, use_dp=True)
train_client_local(unit_name, global_weights=None, epochs=1)  # Local training
aggregate_weights(client_results)  # FedAvg (weighted by n_samples)
federated_round(round_num)  # Execute one FL round
train()  # Full training loop
evaluate(weights, data)  # Compute AUROC on dataset
```

**Key Design Decisions**:
1. **Manual Aggregation**: No Flower dependency (simpler, fewer imports)
2. **Global Scaler**: StandardScaler fitted on ALL training data once (prevents client-specific bias)
3. **Weighted FedAvg**: Weight by `n_samples` per client
4. **Evaluation**: AUROC computed per round on validation set

**Privacy Integration** (src/fl/privacy.py):
- `DifferentialPrivacyMechanism` class (Gaussian mechanism)
- Parameters: ε=1.0, δ=1e-5, clipping_norm=1.0
- Calculates σ = sqrt(2 * ln(1.25/δ)) / ε ≈ 4.7985
- Applied per-client BEFORE aggregation

### 1.3 Experiment 1: Federated Baseline ✅
**File**: `experiments/exp1_baseline.py`

**Pipeline**:
1. Load 65,273 samples, split 70/15/15
2. Create 7 federated clients from training data
3. Train centralized baseline (sklearn LR)
4. Initialize FederatedTrainer with 5 rounds
5. Train federated model
6. Compare AUROC + divergence

### 1.4 RESULTS (WITHOUT DP - Core Algorithm Validation) ✅
```
Centralized Test AUROC:  0.8880
Federated Test AUROC:    0.8850
Divergence:              0.0030 (EXCELLENT)

✓ Federated AUROC (0.8850) ≥ 0.85 target
✓ Divergence (0.0030) < 0.05 target
✅ EXPERIMENT 1 PASSED
```

**DP Results** (use_dp=True):
```
Federated Test AUROC:    0.3934  ❌ (too much noise)
```
**Finding**: DP noise at ε=1.0 too aggressive. Needs tuning for Phase 2.

### 1.5 Configuration
**File**: `src/config/config.py`
- `GCP_PROJECT_ID`: 'mimic-iv-research-496704'
- `COHORT_CACHE_PATH`: data/cache/mimic_iv_cohort.csv
- `DP_EPSILON`: 1.0
- `DP_DELTA`: 1e-5
- `CLIPPING_THRESHOLD`: 1.0
- `MAX_ITER`: 2000 (LogisticRegression)
- `NUM_ROUNDS`: 20 (default, Exp1 uses 5)

---

## CODEBASE STATUS

### Files Implemented/Modified

**Core Data Pipeline**:
- ✅ `src/data/loader.py` — BigQuery cohort loader with caching
- ✅ `src/data/preprocess.py` — Feature engineering (exists, unchanged for Phase 1)
- ✅ `src/data/split.py` — Care-unit partitioning (NEWLY IMPLEMENTED)

**Federated Learning**:
- ✅ `src/training/federated.py` — FederatedTrainer class (NEWLY IMPLEMENTED)
- ✅ `src/training/centralized.py` — Baseline training (unchanged)
- ✅ `src/fl/privacy.py` — Differential privacy (unchanged, working)
- ✅ `src/fl/client.py` — Client abstraction (exists, not used in Phase 1)
- ✅ `src/fl/server.py` — Server abstraction (exists, not used in Phase 1)

**Experiments**:
- ✅ `experiments/exp1_baseline.py` — Phase 1 validation (NEWLY IMPLEMENTED)
- ⏳ `experiments/exp2_noniid.py` — Non-IID experiments (ready for Phase 2)
- ⏳ `experiments/exp3_clients.py` — Variable client count (ready for Phase 2)
- ⏳ `experiments/exp4+.py` — Hyperparameter/robustness (ready for Phase 2)

**Configuration**:
- ✅ `src/config/config.py` — Central config (complete)

### Dependencies
```
google-cloud-bigquery
pandas
numpy
scikit-learn (LogisticRegression, StandardScaler, metrics.roc_auc_score)
```
Note: Flower removed (not needed for manual FedAvg)

---

## KNOWN ISSUES & WORKAROUNDS

### ✅ RESOLVED
1. **BigQuery labevents join error**: Fixed schema (hadm_id, not stay_id)
2. **Missing vitals min/max**: Implemented all aggregations (MEAN, MIN, MAX)
3. **Clinical scores hardcoded**: Temporarily set to 0 (Phase 2 task)
4. **DP noise return value**: Fixed unpacking (add_noise returns tuple)

### 🟡 PENDING (Phase 2)
1. **DP Noise Tuning**: σ=4.7985 too aggressive for ε=1.0 — need lower ε or noise scaling
2. **Clinical Scores**: SOFA, SAPSII, Charlson not yet computed from raw data
3. **Non-IID Distribution**: Dirichlet sampling implemented but not tested in Phase 2 yet

---

## PHASE 2 ROADMAP (NOT YET STARTED)

### 2.1 DP Tuning Experiments
- [ ] Reduce DP noise: test ε ∈ [0.1, 0.5, 1.0, 5.0]
- [ ] Measure privacy-utility tradeoff
- [ ] Target: Federated AUROC ≥0.85 WITH DP

### 2.2 Non-IID Distribution
- [ ] Exp2: Dirichlet-partitioned non-IID clients
- [ ] Validate federated convergence under distribution heterogeneity
- [ ] Compare IID vs Non-IID performance

### 2.3 Hyperparameter Sensitivity
- [ ] Exp3: Variable client counts (2, 4, 7, 14)
- [ ] Exp4: Aggregation methods comparison (FedAvg, FedProx, etc.)
- [ ] Exp5: Dropout simulation (stragglers)

### 2.4 Advanced Features
- [ ] Exp6: Clinical scores implementation
- [ ] Exp7: Model explainability (SHAP)
- [ ] Exp8: Byzantine robustness
- [ ] Exp9: Scalability analysis

---

## HOW TO CONTINUE IN NEW CHAT

### To Resume Work
1. Copy this entire document
2. Paste in new chat
3. Use `read_file` to load codebase files as needed
4. Continue from Phase 2 (or re-run Phase 1 if needed)

### Quick Start Commands
```bash
cd /home/raiyanjiyon/Projects/federated-healthcare-ml
source .venv/bin/activate

# Run Phase 1 validation
python experiments/exp1_baseline.py

# Check results
grep -E "(AUROC|PASSED|FAILED)" /tmp/exp1_output.log
```

### Key Files to Reference
- [src/data/loader.py](src/data/loader.py) — BigQuery integration
- [src/data/split.py](src/data/split.py) — Care-unit partitioning
- [src/training/federated.py](src/training/federated.py) — FederatedTrainer
- [experiments/exp1_baseline.py](experiments/exp1_baseline.py) — Exp1 pipeline
- [src/config/config.py](src/config/config.py) — Central config

---

## SUCCESS CRITERIA

### Phase 0 ✅
- [x] Load MIMIC-IV cohort from BigQuery
- [x] Centralized baseline AUROC ≥0.75
- [x] No data leakage (first 24h only)

### Phase 1 ✅
- [x] Partition into 7 care-unit federated clients
- [x] Federated AUROC ≥0.85 (without DP)
- [x] Divergence from centralized <0.05
- [x] Manual FedAvg aggregation working

### Phase 2 (TODO)
- [ ] DP-enabled federated AUROC ≥0.85
- [ ] Non-IID experiments validated
- [ ] Clinical scores computed

---

## TECHNICAL NOTES FOR NEXT SESSION

### BigQuery Querying
- Always filter first 24h: `WHERE charttime BETWEEN intime AND intime+24h`
- Join chartevents/labevents via `hadm_id` + `intime` lookup (through icustays)
- Use CTEs for intermediate calculations (18 CTEs total in current query)
- Budget: ~2-5GB per full cohort query (~$0.01-0.025 per run)

### Federated Learning Design
- Global scaler fits on all training data (prevent client drift)
- Weighted FedAvg: weight = n_samples / total_samples
- Each round: all 7 clients train locally, then aggregate
- Validation AUROC tracked per round (convergence check)

### DP Integration
- Add noise BEFORE aggregation (per-client)
- Clipping threshold: 1.0 (all gradients clipped to unit norm)
- Current parameters: ε=1.0, δ=1e-5 → σ=4.7985
- Problem: σ too large, reduces AUROC from 0.88 to 0.39
- Solution (Phase 2): Tune ε or implement Renyi DP

### Performance Baselines
- Centralized LR: 0.8880 test AUROC (expected)
- Federated (no DP): 0.8850 test AUROC (matches well)
- Federated (DP, ε=1.0): 0.3934 test AUROC (noise too high)

---

## GCP CREDENTIALS
- **Project**: mimic-iv-research-496704
- **PhysioNet Account**: Already verified
- **Billing**: Enabled
- **BigQuery**: `bq` CLI available in `.venv`

---

Last Updated: 2026-05-22 12:05
