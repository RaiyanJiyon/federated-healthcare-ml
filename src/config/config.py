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