"""Configuration settings for MIMIC-IV Federated Healthcare ML project"""
import os 
from pathlib import Path
from datetime import datetime

# ===== PROJECT SETUP =====
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_CACHE_DIR = DATA_DIR / "cache"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"

# Create directories if they don't exist
for directory in [DATA_RAW_DIR, DATA_CACHE_DIR, DATA_PROCESSED_DIR, LOGS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ===== GOOGLE BIGQUERY & MIMIC-IV SETUP =====
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'mimic-iv-research-496704')
BQ_BILLING_PROJECT = GCP_PROJECT_ID
MIMIC_VERSION = "3.1"
BQ_PROJECT_PHYSIONET = "physionet-data"
BQ_DATASET_HOSP = f"{BQ_PROJECT_PHYSIONET}.mimiciv_{MIMIC_VERSION.replace('.', '_')}_hosp"
BQ_DATASET_ICU = f"{BQ_PROJECT_PHYSIONET}.mimiciv_{MIMIC_VERSION.replace('.', '_')}_icu"
BQ_DATASET_DERIVED = f"{BQ_PROJECT_PHYSIONET}.mimiciv_{MIMIC_VERSION.replace('.', '_')}_derived"

# Local cache path for cohort
COHORT_CACHE_PATH = DATA_CACHE_DIR / "mimic_iv_cohort.csv"

# ===== DATASET CONFIGURATION =====
PREDICTION_TASK = "mortality"
TARGET_COLUMN = "hospital_expire_flag"

# Cohort filters
COHORT_MIN_AGE = 18
COHORT_MAX_AGE = None  # No upper limit
COHORT_MIN_ICU_LOS_HOURS = 4
COHORT_FIRST_ICU_STAY_ONLY = True

# ICU Care Units (real MIMIC-IV units for non-IID partitioning)
ICU_CARE_UNITS = [
    'MICU',      # Medical ICU
    'SICU',      # Surgical ICU
    'CCU',       # Cardiac Care Unit
    'CVICU',     # Cardiovascular ICU
    'Neuro SICU', # Neurological Surgical ICU
    'TSICU',     # Trauma Surgical ICU
    'MICU/SICU'  # Mixed Medical-Surgical ICU
]

# ===== FEATURE DEFINITIONS (32 raw MIMIC features) =====
# Demographics (4 features)
DEMOGRAPHICS_FEATURES = [
    'age', 'gender_M', 'admission_type_emergency', 'insurance_medicare'
]

# Vital Signs - First 24h (13 features)
VITALS_FEATURES = [
    'heart_rate_mean', 'heart_rate_min', 'heart_rate_max',
    'sbp_mean', 'sbp_min',
    'mbp_mean', 'mbp_min',
    'resp_rate_mean', 'resp_rate_max',
    'temperature_mean',
    'spo2_mean', 'spo2_min',
    'glucose_mean'
]

# Lab Values - First 24h (12 features)
LAB_FEATURES = [
    'creatinine_max', 'bun_max',
    'sodium_min', 'sodium_max',
    'potassium_max',
    'bicarbonate_min',
    'hemoglobin_min',
    'wbc_max',
    'platelet_min',
    'lactate_max',
    'bilirubin_total_max',
    'inr_max'
]

# Clinical Scores (3 features)
SCORES_FEATURES = [
    'sofa_score', 'sapsii_score', 'charlson_index'
]

# All raw features (32 total)
ALL_FEATURES = DEMOGRAPHICS_FEATURES + VITALS_FEATURES + LAB_FEATURES + SCORES_FEATURES

# ===== DATA SPLIT & FEDERATED LEARNING =====
RANDOM_SEED = 42
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15  # Will be computed from remaining after train+val

PARTITION_STRATEGY = "care_unit"  # "care_unit", "dirichlet", or "iid"
DIRICHLET_ALPHA = 0.5  # For Dirichlet partitioning (lower = more non-IID)

# ===== FEDERATED LEARNING CONFIGURATION =====
NUM_CLIENTS = 7  # Number of care units (will adapt to actual data)
NUM_ROUNDS = 20  # Initial FL rounds for Phase 1
CLIENT_FRACTION = 1.0  # Fraction of clients participating per round
MIN_FIT_CLIENTS = 1
MIN_EVAL_CLIENTS = 1
MIN_AVAILABLE_CLIENTS = 1
MIN_PATIENTS_PER_CLIENT = 100  # Minimum patient threshold for a care unit to be a valid client

# ===== MODEL CONFIGURATION =====
MODEL_TYPE = "logistic_regression"  # "logistic_regression", "mlp", "xgboost", "random_forest"
MAX_ITER = 2000  # Maximum iterations for sklearn models
LEARNING_RATE = 0.001  # Learning rate for optimizer
DECISION_THRESHOLD = 0.30  # Recall-optimized threshold for mortality (lower sensitivity is worse)
CLASS_WEIGHT = 'balanced'  # Handle class imbalance

# ===== PRIVACY: DIFFERENTIAL PRIVACY =====
DP_EPSILON = 1.0  # Privacy budget
DP_DELTA = 1e-5  # Failure probability (CRITICAL: use 1e-5 for healthcare, NOT 0.01)
GRADIENT_CLIPPING = True
CLIPPING_THRESHOLD = 1.0  # Explicit L2 norm bound for weight updates

# ===== EXPLAINABILITY: SHAP =====
ENABLE_SHAP = True
SHAP_BACKGROUND_SAMPLES = 200  # Sample count for SHAP explainer background
SHAP_MAX_FEATURES = 20  # Top N features to display in SHAP plots

# ===== AGGREGATION STRATEGY =====
AGGREGATION_STRATEGY = "fedavg"  # "fedavg" or "fedprox"
FEDPROX_MU = 0.01  # Proximal term weight for FedProx

# ===== EXPERIMENT PARAMETERS =====
BASELINE_NUM_CLIENTS = 7
BASELINE_NUM_ROUNDS = 20
BASELINE_MAX_ITER = 2000

# Scalability testing
SCALABILITY_CLIENT_COUNTS = [3, 5, 7, 10, 15]
SCALABILITY_NUM_ROUNDS = 10

# ===== ADVERSARIAL ROBUSTNESS PARAMETERS =====
ENABLE_ADVERSARIAL_TESTING = True
POISON_RATE = 0.1
POISON_STRATEGIES = ["label_flip", "sign_flip", "noise"]
BYZANTINE_ATTACKER_FRACTIONS = [1/7, 2/7]  # 14% and 29% attacker presence

# ===== FEATURE ENGINEERING =====
ENABLE_FEATURE_ENGINEERING = True
INTERACTION_PAIRS = [
    ('sofa_score', 'age'),
    ('sapsii_score', 'age'),
    ('charlson_index', 'age'),
    ('creatinine_max', 'bun_max'),
    ('sapsii_score', 'sofa_score'),
    ('heart_rate_mean', 'sbp_mean')
]
POLYNOMIAL_FEATURES = ['sofa_score', 'sapsii_score', 'age', 'creatinine_max']
RATIO_FEATURES = [
    ('creatinine_max', 'bun_max'),
    ('sapsii_score', 'charlson_index')
]

# ===== CLINICAL BOUNDS FOR OUTLIER CLIPPING =====
# Format: feature_name: (min_val, max_val)
CLINICAL_BOUNDS = {
    'age': (0, 120),
    'heart_rate_mean': (20, 200),
    'heart_rate_min': (20, 200),
    'heart_rate_max': (20, 200),
    'sbp_mean': (40, 250),
    'sbp_min': (40, 250),
    'mbp_mean': (20, 150),
    'mbp_min': (20, 150),
    'resp_rate_mean': (5, 60),
    'resp_rate_max': (5, 60),
    'temperature_mean': (30, 45),  # Celsius
    'spo2_mean': (50, 100),
    'spo2_min': (50, 100),
    'glucose_mean': (30, 600),
    'creatinine_max': (0, 15),
    'bun_max': (0, 200),
    'sodium_min': (100, 160),
    'sodium_max': (100, 160),
    'potassium_max': (1, 10),
    'bicarbonate_min': (5, 40),
    'hemoglobin_min': (4, 20),
    'wbc_max': (0, 100),
    'platelet_min': (0, 1000),
    'lactate_max': (0, 30),
    'bilirubin_total_max': (0, 30),
    'inr_max': (0, 15),
    'sofa_score': (0, 24),
    'sapsii_score': (0, 163),
    'charlson_index': (0, 37)
}

# ===== LOGGING =====
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
EXPERIMENT_LOG_DIR = LOGS_DIR / "experiments"
EXPERIMENT_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ===== REPRODUCIBILITY METADATA =====
CONFIG_CREATION_TIME = datetime.now().isoformat()
MIMIC_COHORT_VERSION = f"mimic_v{MIMIC_VERSION}_phase0"