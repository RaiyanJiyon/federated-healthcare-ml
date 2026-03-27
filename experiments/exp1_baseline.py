"""Experiment 1: Baseline Centralized Training

This experiment trains a centralized machine learning model on the complete dataset
without federated learning. This serves as the baseline for comparing against federated
learning approaches.

Key metrics:
- Accuracy: Overall prediction accuracy
- Precision: True positive rate among positive predictions
- Recall: True positive rate among actual positives (important for healthcare)
- F1-Score: Harmonic mean of precision and recall
- Confusion Matrix: Detailed breakdown of predictions

This baseline demonstrates the best-case scenario where all data is available
centrally, providing an upper bound for federated learning performance.
"""

import sys
from pathlib import Path

# Add parent directory to path to import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.centralized import train_centralized_baseline


def main():
    """Run centralized baseline experiment."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: CENTRALIZED BASELINE TRAINING")
    print("=" * 80)
    print("\nObjective: Train a single model on all available data (no federated learning)")
    print("This serves as the baseline for comparing federated approaches.")
    
    try:
        # Run centralized training pipeline
        trainer, results_path = train_centralized_baseline(verbose=True, save_results=True)
        
        print("\n" + "=" * 80)
        print("✅ EXPERIMENT 1 COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nResults saved to: {results_path}")
        print("\nThis baseline will be compared against federated learning in:")
        print("  - Experiment 2: Non-IID Federated Learning")
        print("  - Experiment 3: Multi-Client Analysis")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Experiment 1 failed with error:")
        print(f"{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
