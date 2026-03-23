# Federated Learning for Healthcare ML

A comprehensive framework for federated learning applied to healthcare machine learning problems.

## Project Structure

```
federated-healthcare-ml/
├── data/
│   ├── raw/              # Raw data
│   └── processed/        # Processed data
├── notebooks/
│   └── exploration.ipynb # Data exploration
├── src/
│   ├── config/           # Configuration
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model definitions
│   ├── fl/               # Federated learning components
│   ├── training/         # Training logic
│   ├── evaluation/       # Evaluation metrics
│   └── utils/            # Utility functions
├── experiments/          # Experimental scripts
├── results/              # Results and logs
├── paper/                # Paper drafts
├── requirements.txt      # Dependencies
├── run.py                # Main entry point
└── README.md             # This file
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation
Place your healthcare data in `data/raw/` directory.

### 2. Configuration
Edit `src/config/config.py` to set your parameters.

### 3. Run Experiments
Execute experiments from the `experiments/` directory:
```bash
python experiments/exp1_baseline.py
python experiments/exp2_noniid.py
python experiments/exp3_clients.py
```

### 4. Main Pipeline
```bash
python run.py
```

## Key Features

- **Centralized Training**: Baseline implementation
- **Federated Learning**: FedAvg and FedProx strategies
- **Non-IID Data Distribution**: Realistic healthcare data scenarios
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Easy Configuration**: Centralized config management

## Modules

### Data Module (`src/data/`)
- `loader.py`: Load data from various sources
- `preprocess.py`: Data preprocessing utilities
- `split.py`: Train/test and client data distribution

### Models (`src/models/`)
- `model.py`: Base model with support for multiple algorithms

### Federated Learning (`src/fl/`)
- `client.py`: FL client implementation
- `server.py`: FL server implementation
- `strategy.py`: Aggregation strategies (FedAvg, FedProx)

### Training (`src/training/`)
- `centralized.py`: Centralized training
- `federated.py`: Federated training

### Evaluation (`src/evaluation/`)
- `metrics.py`: Evaluation metrics
- `visualize.py`: Visualization utilities

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Submit a pull request

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.
