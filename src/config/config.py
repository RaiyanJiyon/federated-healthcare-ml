"""Configuration settings for the federated healthcare ML project"""
import os 
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"

# Create directories if they don't exist
for directory in [DATA_PROCESSED_DIR, LOGS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_PATH = DATA_RAW_DIR / "diabetes.csv"
TEST_SIZE = 0.2  # 80-20 train-test split
RANDOM_SEED = 42

# Model hyperparameters
LEARNING_RATE = 0.01
BATCH_SIZE = 32
EPOCHS = 10
MODEL_TYPE = "logistic_regression"  # or "neural_network"

# Federated Learning configuration
NUM_CLIENTS = 5  # Number of simulated clients
NUM_ROUNDS = 10  # Communication rounds
CLIENT_FRACTION = 1.0  # Fraction of clients participating per round
MIN_FIT_CLIENTS = 1
MIN_EVAL_CLIENTS = 1
MIN_AVAILABLE_CLIENTS = 1

# Non-IID data distribution
NON_IID = True  # Set to False for IID distribution
DIRICHLET_ALPHA = 0.5  # Lower values = more non-IID (use for realistic setup)

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ===== CRITICAL MODEL TRAINING PARAMETERS =====
# These were hardcoded in experiments - now centralized
MAX_ITER = 2000  # Maximum iterations for model convergence
DECISION_THRESHOLD = 0.30  # Threshold for binary classification (from recall optimization)
CLASS_WEIGHT = 'balanced'  # Handle class imbalance

# ===== EXPERIMENT BASELINE PARAMETERS =====
# Used for consistency across experiments
BASELINE_NUM_CLIENTS = 5
BASELINE_NUM_ROUNDS = 10
BASELINE_MAX_ITER = 2000

# ===== SCALABILITY TESTING PARAMETERS =====
SCALABILITY_CLIENT_COUNTS = [5, 7, 10, 15, 20]  # Client counts to test
SCALABILITY_NUM_ROUNDS = 10  # Rounds per configuration

# ===== PRIVACY PARAMETERS =====
# Differential Privacy
DP_EPSILON = 1.0  # Privacy budget
DP_DELTA = 0.01  # Failure probability (typically 1/n where n=number of records)
GRADIENT_CLIPPING = True  # Enable gradient clipping for privacy
GRADIENT_CLIPPING_NORM = 1.0  # L2 norm threshold for clipping

# ===== ADVERSARIAL ROBUSTNESS PARAMETERS =====
ENABLE_ADVERSARIAL_TESTING = True
POISON_RATE = 0.1  # Fraction of clients to poison in attack
POISON_STRATEGIES = ["scaling", "sign_flip", "label_flip", "random"]

# ===== FEATURE ENGINEERING PARAMETERS =====
ENABLE_FEATURE_ENGINEERING = True
INTERACTION_PAIRS = [  # Feature pairs for interaction terms
    ('Glucose', 'BMI'), 
    ('Glucose', 'Age'), 
    ('BMI', 'Age'),
    ('Glucose', 'Insulin'),
    ('BloodPressure', 'Age')
]
POLYNOMIAL_FEATURES = ['Glucose', 'BMI', 'Age', 'BloodPressure']  # Features for polynomial expansion
RATIO_FEATURES = [  # Pairs of features for ratio creation
    ('Glucose', 'Insulin'),
    ('BloodPressure', 'Age')
]