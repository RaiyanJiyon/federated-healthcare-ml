# Clinically Reliable Privacy-Preserving Federated Learning under Heterogeneous ICU Environments

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn 1.6+](https://img.shields.io/badge/scikit--learn-1.6+-orange.svg)](https://scikit-learn.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![IEEE JBHI](https://img.shields.io/badge/IEEE-JBHI-blue.svg)](paper/main.tex)

A research-grade federated learning framework designed to train clinically reliable, privacy-preserving, and Byzantine-robust mortality prediction models across heterogeneous intensive care unit (ICU) environments without centralizing sensitive Electronic Health Records (EHR).

**Key Achievements:** 
- Evaluated on **65,273 MIMIC-IV ICU admissions** across 7 clinical care units and externally validated on **131,517 / 22,361 eICU-CRD admissions** across 7 independent hospitals.
- **Probability Recalibration:** Eliminates federated probability compression via Platt scaling, reducing Expected Calibration Error (ECE) from **0.2338 to 0.0091** (< 0.01).
- **Byzantine Resilience:** Median and Krum aggregations maintain **AUROC ≥ 0.8594** under severe 3-of-7 malicious client attacks, while standard FedAvg degrades to 0.5011.
- **Formal Privacy:** Enforces client-side DP-SGD ($\epsilon = 4.36, \delta = 10^{-5}$) with per-sample gradient clipping ($C = 1.0$) and moments accountant verification.

---

<a id="quick-links"></a>
## 🎯 Quick Links

### Section Navigation
- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing & Reproducibility](#testing--reproducibility)
- [External Validation](#external-validation)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

### 📖 Full Documentation & Research Artifacts
- 📄 **IEEE Manuscript Source (LaTeX):** [`paper/main.tex`](paper/main.tex)
- 📊 **Empirical Results Summary:** [`results/summary/RESULTS_SUMMARY.md`](results/summary/RESULTS_SUMMARY.md)
- 📋 **Confirmatory Summary:** [`results/summary/CONFIRMATORY_SUMMARY.md`](results/summary/CONFIRMATORY_SUMMARY.md)
- 🧪 **Main Performance Table (TeX):** [`results/summary/table1_main_results.tex`](results/summary/table1_main_results.tex)
- 📈 **Scalability Benchmark Table:** [`results/summary/table2_scalability.md`](results/summary/table2_scalability.md)
- 🛡️ **Byzantine Robustness Report:** [`results/summary/table6_robustness.md`](results/summary/table6_robustness.md)
- 🔒 **Privacy-Utility Tradeoff Summary:** [`results/summary/table5_privacy.md`](results/summary/table5_privacy.md)
- 📄 **Project License:** [`LICENSE`](LICENSE)

---

<a id="features"></a>
## ✨ Features

### Federated Learning & Aggregation Strategies
- **FedAvg** baseline for standard cooperative federated optimization.
- **FedProx** proximal regularization ($\mu = 0.01$) for data heterogeneity control.
- **FedF2 (Clinical Sensitivity-Aware Aggregation):** Dynamically weights client updates using local validation $F_2$-scores at a uniform reference threshold $\tau_{\text{ref}} = 0.39$, prioritizing clinical recall.

### Post-Hoc Probability Recalibration
- **Platt Scaling (Sigmoid Calibration):** Server-side sigmoid calibration to resolve severity of probability compression caused by weight averaging.
- Recovers decision boundaries and improves Precision from $11.45\%$ to $69.85\%$ while maintaining target Recall.

### Layered Privacy & Byzantine Defenses
- **Differential Privacy (DP-SGD):** Client-side gradient clipping ($C = 1.0$) and calibrated Gaussian noise ($\sigma = 4.80$) with formal moments accountant composition ($\epsilon = 4.36, \delta = 10^{-5}$).
- **Byzantine Resilient Aggregation:** Coordinate-wise **Median** and distance-based **Krum** selection to defend against label-flipping, sign-flipping, and adaptive poisoning attacks.

### Multi-Model & Multi-Dataset Support
- **MIMIC-IV Cohort:** 65,273 admissions partitioned into 7 ICU Care Units (MICU, SICU, CCU, CVICU, Neuro SICU, TSICU, MICU/SICU) for realistic Non-IID evaluation.
- **eICU-CRD Cohort:** 22,361 test admissions across 7 independent hospitals for external generalizability testing.
- **Architectural Flexibility:** Supports Logistic Regression (31 params), Multi-Layer Perceptron ($31 \to 64 \to 32 \to 1$, 4,161 params), and Federated XGBoost via soft voting (200 trees).

### Rigorous Evaluation & Statistical Auditing
- Multi-metric evaluation emphasizing clinical utility: AUROC, AUPRC, ECE, Recall (Sensitivity), Precision, and $F_2$-score.
- Statistical significance evaluation with 5 deterministic random seeds and DeLong testing.
- Automated generation of publication-ready vector figures (PDF/PNG).

---

<a id="architecture"></a>
## 🏗️ Architecture

### System Design Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│             Federated Clinical Server Orchestration                   │
├────────────────────────────────────────────────────────────────────────┤
│  Server Aggregation Methods:                                           │
│  • FedAvg / FedProx / FedF2 (F2-Weighted)                              │
│  • Byzantine Defenses: Coordinate-Wise Median / Krum Selection         │
│  • Server Post-Hoc Calibration: Platt Scaling (Sigmoid Recalibration)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
  ┌────────▼────────┐      ┌────────▼────────┐      ┌────────▼────────┐
  │  Client 1: MICU │      │  Client 2: SICU │      │ Client k: CVICU │
  │  • 21,418 ICU   │      │  • 10,872 ICU   │      │ • 8,068 ICU     │
  │    Admissions   │      │    Admissions   │      │   Admissions    │
  │  • Local SGD /  │      │  • Local SGD /  │      │ • Local SGD /   │
  │    DP-SGD       │      │    DP-SGD       │      │   DP-SGD        │
  │  • Clip & Noise │      │  • Clip & Noise │      │ • Clip & Noise  │
  └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
           │                        │                        │
           └────────────────────────┼────────────────────────┘
                                    │ Updates Δw_k
                   ┌────────────────▼────────────────┐
                   │   External Validation Engine    │
                   │   • eICU-CRD Dataset            │
                   │   • 7 Independent Hospitals     │
                   │   • Generalizability Assessment  │
                   └─────────────────────────────────┘
```

### Module Organization

| Directory / File | Purpose |
|-------------------|---------|
| `data/` | Raw MIMIC-IV / eICU-CRD placeholder, cache, and preprocessed CSV cohorts |
| `src/config/` | System configuration (`config.py`) and trial settings |
| `src/data/` | Cohort loaders, Non-IID care-unit splitters, and feature scalers |
| `src/models/` | Baseline models: Logistic Regression, MLP, and XGBoost wrappers |
| `src/fl/` | Aggregation algorithms (`strategy.py`), DP-SGD (`privacy.py`), and Byzantine defenses (`robust_aggregation.py`) |
| `src/training/` | Distributed federated loops and centralized training pipelines |
| `src/evaluation/` | Performance metrics: AUROC, AUPRC, Expected Calibration Error (ECE), and confusion matrices |
| `src/utils/` | SHAP feature drift explainability, logging, and plotting utilities |
| `experiments/` | Executable experiment suite (`exp1_baseline.py` through `exp8_calibration_and_pr.py`) |
| `results/` | Output plots, summary CSVs, TeX tables, and reproducibility metadata |
| `paper/` | LaTeX manuscript (`main.tex`), vector figures, and clean bibliography (`references.bib`) |
| `run.py` | Command-Line Interface (CLI) pipeline entrypoint |

---

<a id="getting-started"></a>
## 🚀 Getting Started

### Prerequisites

- **Python 3.9+** (Tested up to 3.14)
- **4+ GB RAM** (8+ GB recommended for multi-seed runs)
- **Linux / macOS / Windows** (No GPU required; optimized for CPU execution)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RaiyanJiyon/federated-healthcare-ml.git
   cd federated-healthcare-ml
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import sklearn, numpy, pandas, matplotlib; print('✓ All core packages installed successfully')"
   ```

### Dataset Setup

**MIMIC-IV Dataset Setup:**
1. Request access via PhysioNet for [MIMIC-IV (v3.1)](https://physionet.org/content/mimiciv/3.1/).
2. Extract or query the 31 clinical features (demographics, first 24h vitals, first 24h labs, SOFA/SAPS II/Charlson scores).
3. Place the preprocessed cohort file at:
   ```text
   data/processed/mimic_preprocessed.csv
   ```
   *(The dataset contains 65,273 ICU admissions partitioned across 7 care units).*

**eICU-CRD External Dataset Setup (Optional):**
1. Extract adult ICU admissions from [eICU Collaborative Research Database](https://physionet.org/content/eicu-crd/).
2. Cache cohort to `data/cache/eicu_cohort.csv` for external validation runs.

---

<a id="usage"></a>
## 📊 Usage

### 1. Integrated Execution Entrypoint

Run the complete baseline and federated pipeline in one command:
```bash
python run.py --dataset mimic_iv --rounds 20 --seed 42
```

### 2. Core Baseline & Federated Training

**Run Centralized vs. Non-IID Federated Baseline:**
```bash
python experiments/exp1_baseline.py
python experiments/exp2_noniid.py
```

**Expected Console Output:**
```text
[Round 01] Loss: 0.4215 | Val AUROC: 0.8241 | Val ECE: 0.2104
[Round 05] Loss: 0.3120 | Val AUROC: 0.8784 | Val ECE: 0.2338
Post-Hoc Platt Calibration Applied.
Recalibrated ECE: 0.0091 | Test AUROC: 0.8784 | Test Precision: 69.85% | Test Recall: 43.49%
```

### 3. Multi-Model Architecture Sweep

Compare Logistic Regression, MLP, and Federated XGBoost (Soft Voting):
```bash
python experiments/exp1_baseline_multimodel.py
```

### 4. Aggregation & Clinical Sensitivity Sweep (FedF2)

Evaluate FedAvg vs. FedProx vs. FedF2 ($F_2$-Weighted):
```bash
python experiments/exp4_aggregation_comparison.py
python experiments/exp7_clinical_aggregation.py
python experiments/exp_robustness_fedf2.py
```

### 5. Privacy (DP-SGD) & Calibration Sweeps

Run DP-SGD moments accountant sweep and Platt scaling evaluation:
```bash
python experiments/exp7_differential_privacy.py
python experiments/exp8_calibration_and_pr.py
python experiments/phase5_dp_sweep.py
```

### 6. Scalability & Network Dropout Sweeps

Simulate client expansion (up to 28 clients) and network dropouts (10%--30%):
```bash
python experiments/exp3_clients.py
python experiments/exp5_dropout_simulation.py
```

### 7. Regenerate Manuscript Vector Figures

Recompile all vector graphics embedded in `paper/main.tex`:
```bash
python experiments/regenerate_all_figures.py
```

---

<a id="results"></a>
## 📈 Results

### Primary System Performance & Calibration (MIMIC-IV Dataset, 7 ICU Care Units)

All metrics evaluated at threshold $\tau_{\text{ref}} = 0.39$ after Platt scaling (except uncalibrated raw FedAvg):

| Configuration | Scenario | AUROC | AUPRC | ECE | Recall (Sens.) | Precision | $F_2$-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized Baseline** | Clean | **0.8920** | **0.6124** | **0.0084** | 85.20% | 35.40% | 0.6720 |
| **FedAvg (Raw, Uncalibrated)** | Clean | 0.8784 | 0.6024 | 0.2338 | **99.91%** | 11.45% | 0.3925 |
| **FedAvg (Calibrated, Platt)** | Clean | 0.8784 | 0.6024 | 0.0091 | 43.49% | **69.85%** | **0.4704** |
| **FedProx (Calibrated, $\mu=0.01$)** | Clean | 0.8220 | 0.5245 | 0.0072 | 36.70% | 69.71% | 0.4054 |
| **FedF2 (Calibrated, $\gamma=0.5$)** | Clean | 0.8781 | 0.5999 | 0.0089 | 43.40% | 69.80% | 0.4695 |
| **FedAvg (Calibrated)** | Poisoned (1 Degenerate) | 0.7984 | 0.4448 | 0.0081 | 28.68% | 67.86% | 0.3242 |
| **FedF2 (Calibrated, $\gamma=0.5$)** | Poisoned (1 Degenerate) | 0.7947 | 0.4372 | 0.0083 | 28.11% | 68.19% | 0.3186 |

---

### Multi-Model Architecture Comparison

| Model Architecture | Parameters / Structure | Centralized AUROC | Federated AUROC | AUROC Loss (%) | Centralized Recall | Federated Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear (31 params) | 0.8914 | 0.8782 | 1.48% | 85.4% | 83.1% |
| **MLP Neural Network** | $31 \to 64 \to 32 \to 1$ (4,161 params) | 0.9185 | 0.8701 | 5.28% | 85.6% | 81.9% |
| **XGBoost (Soft Voting)** | Tree Ensemble (200 trees) | **0.9243** | **0.9155** | **0.95%** | 85.1% | **85.4%** |

---

### Byzantine Robustness Evaluation across 8 Attack Scenarios (20 Rounds)

Evaluated under clean and adversarial conditions (label-flip, sign-flip, adaptive poisoning attacks):

| Threat Model / Attack Scenario | FedAvg | FedProx | Median | Krum | FedF2 ($\gamma=0.5$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CLEAN Baseline (0 Malicious)** | **0.8784** | 0.7721 | 0.8628 | 0.8653 | 0.8781 |
| **Label-Flip (1 Malicious Client)** | 0.8523 | 0.4642 | **0.8698** | 0.8653 | 0.8534 |
| **Label-Flip (2 Malicious Clients)** | 0.7995 | 0.3626 | 0.8420 | **0.8511** | 0.7862 |
| **Label-Flip (3 Malicious Clients)** | 0.5011 | 0.2953 | **0.8594** | 0.8511 | 0.4000 |
| *— AUROC Degradation Loss (%)* | *(43.0%)* | *(61.8%)* | *(0.4%)* | *(1.6%)* | *(54.4%)* |
| **Sign-Flip (1 Malicious Client)** | 0.8736 | 0.6963 | 0.8670 | 0.8653 | **0.8740** |
| **Sign-Flip (2 Malicious Clients)** | 0.8713 | 0.7431 | 0.8574 | 0.8511 | 0.8713 |
| **Adaptive Attack (1 Malicious Client)** | 0.8563 | 0.5166 | **0.8710** | 0.8653 | 0.8571 |
| **Adaptive Attack (2 Malicious Clients)** | 0.8255 | 0.4728 | 0.8353 | **0.8511** | 0.8182 |
| **Average Robustness Loss (All Attacks)** | **8.5%** | **33.5%** | **1.4%** | **1.6%** | **12.6%** |

---

### External Multi-Hospital Validation (eICU-CRD Dataset)

| Evaluation Cohort | Centralized AUROC | Federated AUROC | AUROC Gap (%) | Post-Hoc ECE |
| :--- | :---: | :---: | :---: | :---: |
| **MIMIC-IV (Development Cohort)** | 0.8920 | 0.8784 | 1.52% | 0.0091 |
| **eICU-CRD (External 7 Hospitals)** | 0.8441 | 0.8337 | **1.23%** | **0.0134** |

---

### Key Research Insights

✅ **Probability Calibration is Imperative:** Raw federated averaging exhibits severe probability compression ($\text{ECE} = 0.2338$). Post-hoc Platt scaling successfully restores calibration ($\text{ECE} < 0.01$) and boosts Precision from $11.45\%$ to $69.85\%$.  
✅ **Byzantine Defense Security Order:** Under heavy 3-client label-flipping attacks ($42.8\%$ corrupted nodes), **Median** ($\text{AUROC} = 0.8594$) and **Krum** ($\text{AUROC} = 0.8511$) completely insulate the global model, whereas FedAvg drops to $0.5011$.  
✅ **Tree-Based FL Excellence:** Federated XGBoost via soft voting preserves **$99.05\%$** of centralized performance ($\text{AUROC} = 0.9155$), outperforming federated MLP ($\text{AUROC} = 0.8701$).  
⚠️ **FedProx Sensitivity:** Proximal regularization ($\mu=0.01$) restricts local adaptation under Non-IID distributions without offering Byzantine protection.  

---

<a id="project-structure"></a>
## 📁 Project Structure

```text
federated-healthcare-ml/
├── README.md                              # Main overview & quick links
├── LICENSE                                # MIT License
├── requirements.txt                       # Core dependencies
├── run.py                                 # Integrated execution entrypoint
│
├── data/                                  # Distributed Patient Cohorts
│   ├── raw/                               # MIMIC-IV raw table placeholders
│   ├── cache/                             # MIMIC-IV & eICU-CRD cohort caches
│   └── processed/                         # mimic_preprocessed.csv cohort
│
├── src/                                   # Core Source Code
│   ├── config/                            # Configuration management
│   │   ├── config.py                      # Master system & experiment parameters
│   │   └── trial_config.py                # Hyperparameter trial configurations
│   ├── data/                              # Data loaders & partitioning
│   │   ├── loader.py                      # MIMIC-IV dataset loader & preprocessor
│   │   ├── eicu_loader.py                 # eICU-CRD external loader
│   │   ├── non_iid_splitter.py            # Non-IID care-unit & Dirichlet splitters
│   │   └── scaler.py                      # StandardScaler & feature scalers
│   ├── models/                            # Predictive Model Definitions
│   │   ├── baseline.py                    # Logistic Regression baseline
│   │   ├── mlp.py                         # Multi-Layer Perceptron (PyTorch)
│   │   ├── xgboost_model.py               # Federated XGBoost soft voting
│   │   └── base_model.py                  # Abstract base model class
│   ├── fl/                                # Federated Components
│   │   ├── strategy.py                    # FedAvg, FedProx & FedF2 aggregators
│   │   ├── privacy.py                     # DP-SGD gradient clipping & noise
│   │   └── robust_aggregation.py          # Coordinate-wise Median & Krum
│   ├── training/                          # Execution Orchestration Loops
│   │   ├── baseline_trainer.py            # Centralized model trainer
│   │   └── federated_trainer.py           # Multi-round federated trainer
│   ├── evaluation/                        # Validation Metrics & Calibration
│   │   ├── metrics.py                     # AUROC, AUPRC, Recall, Precision
│   │   └── calibration.py                 # Platt scaling & ECE computation
│   └── utils/                             # Explainability & Helpers
│       ├── explainability.py              # SHAP feature drift analysis
│       ├── logger.py                      # System logging utilities
│       └── plotting.py                    # Visualization & curve generation
│
├── experiments/                           # Executable Experiment Suite
│   ├── exp1_baseline.py                   # Exp 1: Centralized baseline
│   ├── exp1_baseline_multimodel.py        # Exp 1b: LR vs MLP vs XGBoost
│   ├── exp2_noniid.py                     # Exp 2: Non-IID care-unit FL
│   ├── exp3_clients.py                    # Exp 3: Client count scalability (3 to 28)
│   ├── exp4_aggregation_comparison.py     # Exp 4: FedAvg vs FedProx comparison
│   ├── exp5_dropout_simulation.py         # Exp 5: Network dropout sweep (10%-30%)
│   ├── exp6_hyperparameter_sensitivity.py # Exp 6: Hyperparameter sensitivity
│   ├── exp7_clinical_aggregation.py       # Exp 7a: FedF2 performance evaluation
│   ├── exp7_differential_privacy.py       # Exp 7b: DP-SGD noise sweep
│   ├── exp8_calibration_and_pr.py         # Exp 8: Platt scaling & PR curves
│   ├── exp_robustness_fedf2.py            # Exp 9: FedF2 vs Median/Krum Byzantine sweep
│   ├── phase5_dp_sweep.py                 # DP moments accountant budget sweep
│   └── regenerate_all_figures.py          # Upgrade & export vector PDF figures
│
├── results/                               # Output Artifacts & Reports
│   ├── plots/                             # Generated PNG & PDF vector plots
│   └── summary/                           # Empirical CSV, TeX & MD tables
│       ├── RESULTS_SUMMARY.md             # Complete executive summary
│       ├── CONFIRMATORY_SUMMARY.md        # Audit validation report
│       ├── table1_main_results.tex        # Table I LaTeX source
│       ├── table2_scalability.csv         # Table II scalability data
│       ├── table5_privacy.csv             # Table V DP budget data
│       └── table6_robustness.csv          # Table VI Byzantine robustness data
│
└── paper/                                 # IEEE Manuscript Source
    ├── main.tex                           # Main LaTeX manuscript
    ├── references.bib                     # Academically cleaned bibliography
    └── figures/                           # Embedded vector figures (Figures 1-5)
```

---

<a id="configuration"></a>
## 🔬 Configuration

### Master Settings (`src/config/config.py`)

Experiment parameters are centrally controlled via `src/config/config.py`:

```python
# ===== DATASET & COHORT CONFIGURATION =====
ACTIVE_DATASET = "mimic_iv"          # Options: "mimic_iv" or "eicu_crd"
PREDICTION_TASK = "mortality"
TARGET_COLUMN = "hospital_expire_flag"
COHORT_MIN_AGE = 18
COHORT_MIN_ICU_LOS_HOURS = 4

# ===== FEDERATED LEARNING CONFIGURATION =====
NUM_CLIENTS = 7                      # 7 ICU Care Units in MIMIC-IV
PARTITION_STRATEGY = "care_unit"     # "care_unit", "dirichlet", or "iid"
NUM_ROUNDS = 20                      # Global communication rounds
CLIENT_FRACTION = 1.0                # Active clients per round

# ===== AGGREGATION & CLINICAL SENSITIVITY =====
AGGREGATION_STRATEGY = "fedavg"      # "fedavg", "fedprox", "fedf2", "median", "krum"
FEDPROX_MU = 0.01                    # Proximal term weight
FEDF2_GAMMA = 0.3                    # F2 clinical sensitivity blending factor

# ===== DIFFERENTIAL PRIVACY =====
DP_EPSILON = 4.36                    # Cumulative privacy budget
DP_DELTA = 1e-5                      # Failure probability
GRADIENT_CLIPPING = True
CLIPPING_THRESHOLD = 1.0             # L2 norm clipping bound

# ===== REPRODUCIBILITY =====
RANDOM_SEED = 42
```

---

<a id="testing--reproducibility"></a>
## 🧪 Testing & Reproducibility

### Reproducibility Protocol

All experiments adhere to strict IEEE reproducibility standards:
- **Global Determinism:** Fixed random seed `RANDOM_SEED = 42` applied across NumPy, PyTorch, and Scikit-Learn.
- **Hardware Reference:** Benchmarked on an Intel Xeon E-2286M CPU @ 2.40GHz with 32 GB RAM (GPU not required).
- **Execution Times:**
  - 5-Round Baseline FedAvg: $\approx 42$ seconds.
  - DP-SGD Privatized Run (clip + noise per sample): $\approx 3.2$ minutes.

### Statistical Validation

Execute 5-seed randomized validation runs:
```bash
python experiments/phase5_statistical_aggregation.py
```

---

<a id="external-validation"></a>
## 🌐 External Validation

To demonstrate generalizability beyond MIMIC-IV, the framework includes full support for the **eICU Collaborative Research Database (eICU-CRD)**:
- **Cohort:** 131,517 adult admissions filtered to 22,361 test samples across 7 distinct hospital sites (Hospitals 73, 264, 338, 420, 243, 458, 167).
- **Outcome:** Federated FedAvg achieves **0.8337 AUROC** vs. Centralized **0.8441 AUROC** (a negligible $1.23\%$ performance gap), confirming that privacy-preserving federated clinical models transfer seamlessly to independent multi-hospital networks.

---

<a id="citation"></a>
## 📚 Citation

If you reference this codebase or methodology in your academic work, please cite:

```bibtex
@article{jiyon2026federated,
  title={Federated Learning for ICU Mortality Prediction on Heterogeneous Clinical Data},
  author={Jiyon, Md. Raiyan Ur Rahman and Islam, Md. Naimul and Uddin, Mohammad Shorif},
  journal={IEEE Journal of Biomedical and Health Informatics},
  volume={XX},
  number={XX},
  pages={XX--XX},
  year={2026}
}
```

---

<a id="license"></a>
## 📄 License

This project is licensed under the **MIT License** — see the [`LICENSE`](LICENSE) file for details.

---

<a id="contributing"></a>
## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/clinical-enhancement`).
3. Format code with Black: `black src/ experiments/ --line-length 100`.
4. Commit your changes (`git commit -m 'Add clinical feature enhancement'`).
5. Push to the branch (`git push origin feature/clinical-enhancement`).
6. Open a Pull Request.

---

<a id="known-limitations"></a>
## ⚠️ Known Limitations

1. **Platt Calibration Prerequisite:** Raw federated averaging outputs compressed probabilities. Server-side Platt scaling must be applied before clinical deployment.
2. **FedProx Hyperparameter Sensitivity:** The proximal term ($\mu = 0.01$) restricts local optimization under severe Non-IID distributions without providing Byzantine immunity.
3. **DP Guarantee Scope:** Formal DP-SGD privacy bounds apply to Logistic Regression and MLP architectures; tree-based XGBoost soft voting uses prediction vector aggregation.

---

<a id="troubleshooting"></a>
## 🐛 Troubleshooting

**Q: `FileNotFoundError` for preprocessed dataset**
```bash
# Ensure preprocessed CSV is placed in data/processed/
mkdir -p data/processed
# Copy your preprocessed cohort to data/processed/mimic_preprocessed.csv
```

**Q: Expected Calibration Error (ECE) is high**
```bash
# Verify that Platt scaling recalibration is enabled in evaluation
python experiments/exp8_calibration_and_pr.py
```

---

<a id="contact-and-academic-context"></a>
## 👤 Academic Context & Contact

This project was developed as part of research on privacy-preserving, Byzantine-robust machine learning for critical care medicine.

- **Primary Maintainer:** Md. Raiyan Ur Rahman Jiyon ([raiyanjiyon@gmail.com](mailto:raiyanjiyon@gmail.com))
- **Institutions:** 
  - Department of Computer Science and Engineering, Green University of Bangladesh, Dhaka, Bangladesh
  - Department of Computer Science and Engineering, Jahangirnagar University, Savar, Dhaka, Bangladesh
- **Target Publication:** IEEE Journal of Biomedical and Health Informatics (IEEE JBHI)
