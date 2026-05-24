# Supplementary Material: Federated Learning for ICU Mortality

## S1. Data Preprocessing Details

### Feature Engineering Pipeline
1. **Demographics** (4 features):
   - Age, Gender (binary), Admission type (binary), Insurance (binary)
2. **Vital Signs** (13 features):
   - Heart rate (mean, min, max), BP (systolic, diastolic, mean)
   - Respiratory rate, Temperature, SpO2, Glucose
3. **Lab Values** (12 features):
   - Renal: Creatinine, BUN, Sodium, Potassium
   - Hematologic: Hemoglobin, WBC, Platelets, Bilirubin
   - Metabolic: Bicarbonate, Lactate
4. **Clinical Scores** (3 features):
   - SOFA, SAPSII, Charlson (currently 0; see limitations)

### Data Quality
- Missing value handling: Median imputation (fit on training data)
- Feature clipping: Clinical bounds applied (e.g., HR 0-300 bpm)
- Scaling: StandardScaler (fitted on training set only)
- No patient leakage: First 24h window only

## S2. Federated Learning Algorithm Details

### FedAvg Implementation
```
Initialize: w_0 (random)
for round t = 1 to T:
    for each client k in parallel:
        w_k^t = LocalTraining(w_t-1, D_k, η, E)
    w_t = Aggregate(w_1^t, ..., w_K^t, n_1, ..., n_K)
return w_T
```

where:
- w = model weights (logistic regression coefficients)
- D_k = local data at client k
- η = learning rate (0.01)
- E = local epochs (1)
- n_k = samples at client k
- Aggregate: weighted average by sample count

## S3. Hyperparameter Sensitivity Analysis

| Parameter | Range | Optimal | Rationale |
|-----------|-------|---------|----------|
| # Rounds (T) | 5-50 | 20 | Convergence by round 15-20 |
| Learning rate (η) | 0.001-0.1 | 0.01 | Standard LogisticRegression |
| Local epochs (E) | 1-5 | 1 | No benefit beyond 1 (healthcare data) |
| Min clients/unit | 50-200 | 100 | Balance granularity vs stability |
| DP epsilon (ε) | 0.1-100 | None* | *DP not recommended (49% loss) |

## S4. Care-Unit Distribution

| ICU Unit | Patients | Deaths | Mortality | Avg Age |
|----------|----------|--------|-----------|----------|
| MICU | 8,791 | 1,410 | 16.0% | 64.5 |
| CVICU | 8,118 | 265 | 3.3% | 68.2 |
| MICU/SICU | 6,983 | 957 | 13.7% | 63.1 |
| SICU | 6,372 | 754 | 11.8% | 62.3 |
| TSICU | 5,458 | 569 | 10.4% | 60.1 |
| CCU | 5,113 | 643 | 12.6% | 67.8 |
| Neuro SICU | 907 | 256 | 28.2% | 59.4 |

## S5. Statistical Methods

### Confidence Interval Calculation
- Method: Student's t-distribution
- Samples: 5 runs with different random seeds
- CI: Mean ± t_{0.975,4} × (std / sqrt(n))
- Interpretation: 95% of repeated experiments fall within CI

### Byzantine Attack Simulation
- Attack type: Label flipping (flip all mortality predictions)
- Scenarios: 1/7 clients (14%), 2/7 clients (29%)
- Aggregation: Unmodified FedAvg (no robust aggregation)
- Result: Expected vulnerability above 30% attackers

## S6. Differential Privacy Technical Details

### Gaussian Mechanism
- Noise added per gradient: N(0, σ²)
- Clipping threshold: 1.0 (all gradients clipped to unit norm)
- σ = sqrt(2 * ln(1.25/δ)) / ε
- δ = 1e-5 (failure probability)
- At ε=1.0: σ ≈ 4.80 (very large, degrades utility significantly)

### Privacy-Utility Tradeoff
- At ε=1.0 × 20 rounds = 20 total privacy budget
- AUROC drops from 0.885 to 0.45 (49% loss)
- For clinical use, need ε > 10 (privacy barely useful)
- Recommendation: Use organizational privacy controls instead

## S7. Computational Complexity

| Operation | Complexity | Time (7 clients) |
|-----------|-----------|------------------|
| Local training | O(n * d * E) | ~0.2s per client |
| Aggregation | O(K * d) | ~0.01s |
| Validation | O(n_val * d) | ~0.1s |
| Full round | — | ~1.8s |
| Full training (20 rounds) | — | ~36s |

where n=6,000 (avg client samples), d=31 (features), E=1 (epochs), K=7 (clients)

## S8. Code Availability

### Repository Structure
```
federated-healthcare-ml/
├── src/
│   ├── data/       # Data loading and preprocessing
│   ├── fl/         # Federated learning core
│   ├── training/   # Training pipelines
│   ├── evaluation/ # Metrics and visualization
│   └── config/     # Configuration
├── experiments/    # All experiment scripts (9 total)
├── results/
│   ├── plots/      # Result CSVs and figures
│   ├── reproducibility/ # JSON metadata
│   └── summary/    # This summary + tables
└── generate_publication_plots.py  # Figure generation
```

### Key Scripts
- `experiments/exp1_baseline.py` - Phase 1: FedAvg baseline
- `experiments/exp2_noniid.py` - Non-IID data distribution
- `experiments/exp7_differential_privacy.py` - DP analysis
- `generate_publication_plots.py` - Publication plots

---

**Generated**: 2026-05-23 22:09:25
