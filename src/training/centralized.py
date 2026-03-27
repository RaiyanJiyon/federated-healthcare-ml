"""Centralized training implementation for baseline comparison"""
import numpy as np
import json
from pathlib import Path
from datetime import datetime

from src.config.config import RESULTS_DIR, LOGS_DIR
from src.data.loader import load_dataset_with_df
from src.data.preprocess import DataPreprocessor
from src.data.split import train_test_split_data
from src.models.model import LogisticRegressionModel


class CentralizedTrainer:
    """
    Centralized trainer for baseline healthcare ML model.
    Trains a single model on all available data without federated learning.
    """
    
    def __init__(self, model=None, preprocessor=None):
        """
        Initialize the centralized trainer.
        
        Args:
            model: LogisticRegressionModel instance (creates new if None)
            preprocessor: DataPreprocessor instance (creates new if None)
        """
        self.model = model or LogisticRegressionModel(learning_rate=0.01, max_iter=1000)
        self.preprocessor = preprocessor or DataPreprocessor()
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.train_metrics = None
        self.test_metrics = None
        self.training_time = None
    
    def load_and_preprocess_data(self, verbose=True):
        """
        Load dataset and apply preprocessing.
        
        Args:
            verbose (bool): Print progress information
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if verbose:
            print("\n" + "=" * 70)
            print("CENTRALIZED BASELINE - DATA LOADING & PREPROCESSING")
            print("=" * 70)
        
        # Load dataset
        if verbose:
            print("\n1. Loading dataset...")
        df, X, y = load_dataset_with_df()
        
        # Preprocess data
        if verbose:
            print("\n2. Preprocessing data...")
        X_processed = self.preprocessor.preprocess(df.iloc[:, :-1], fit=True)
        
        # Train-test split
        if verbose:
            print("\n3. Train-test split...")
        X_train, X_test, y_train, y_test = train_test_split_data(X_processed, y)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        if verbose:
            print(f"\n✓ Data ready:")
            print(f"  - Training samples: {len(X_train)}")
            print(f"  - Testing samples: {len(X_test)}")
            print(f"  - Features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, verbose=True):
        """
        Train the centralized model on all training data.
        
        Args:
            verbose (bool): Print training information
            
        Returns:
            dict: Training metrics
        """
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")
        
        if verbose:
            print("\n" + "=" * 70)
            print("CENTRALIZED MODEL TRAINING")
            print("=" * 70)
            print("\n4. Training centralized model on all data...")
        
        import time
        start_time = time.time()
        
        # Train model
        self.train_metrics = self.model.fit(self.X_train, self.y_train.astype(int), verbose=verbose)
        
        self.training_time = time.time() - start_time
        
        if verbose:
            print(f"  Training time: {self.training_time:.2f} seconds")
        
        return self.train_metrics
    
    def evaluate(self, verbose=True):
        """
        Evaluate the model on test data.
        
        Args:
            verbose (bool): Print evaluation metrics
            
        Returns:
            dict: Test metrics
        """
        if not self.model.is_trained:
            raise ValueError("Model must be trained first. Call train() first.")
        
        if verbose:
            print("\n" + "=" * 70)
            print("CENTRALIZED MODEL EVALUATION")
            print("=" * 70)
            print("\n5. Evaluating on test set...")
        
        self.test_metrics = self.model.evaluate(self.X_test, self.y_test.astype(int), verbose=verbose)
        
        return self.test_metrics
    
    def run(self, verbose=True):
        """
        Run complete centralized training pipeline.
        
        Args:
            verbose (bool): Print progress information
            
        Returns:
            dict: Dictionary with train and test metrics
        """
        # Load and preprocess
        self.load_and_preprocess_data(verbose=verbose)
        
        # Train
        self.train(verbose=verbose)
        
        # Evaluate
        self.evaluate(verbose=verbose)
        
        return {
            'train_metrics': self.train_metrics,
            'test_metrics': self.test_metrics,
            'training_time': self.training_time
        }
    
    def save_results(self, filename=None):
        """
        Save training results to JSON file.
        
        Args:
            filename (str): Output filename (uses timestamp if None)
            
        Returns:
            Path: Path to saved file
        """
        if self.train_metrics is None or self.test_metrics is None:
            raise ValueError("Training not completed. Call run() first.")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"centralized_baseline_{timestamp}.json"
        
        output_path = RESULTS_DIR / filename
        
        # Prepare results dictionary
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'LogisticRegression_Centralized',
            'training_samples': len(self.X_train),
            'testing_samples': len(self.X_test),
            'features': self.X_train.shape[1],
            'training_time_seconds': self.training_time,
            'train_metrics': {
                'accuracy': float(self.train_metrics['accuracy']),
                'loss': float(self.train_metrics['loss'])
            },
            'test_metrics': {
                'accuracy': float(self.test_metrics['accuracy']),
                'precision': float(self.test_metrics['precision']),
                'recall': float(self.test_metrics['recall']),
                'f1_score': float(self.test_metrics['f1_score']),
                'loss': float(self.test_metrics['loss']),
                'confusion_matrix': self.test_metrics['confusion_matrix'].tolist()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
        
        return output_path
    
    def print_comparison_summary(self):
        """Print summary of training vs test metrics."""
        if self.train_metrics is None or self.test_metrics is None:
            raise ValueError("Training not completed. Call run() first.")
        
        print("\n" + "=" * 70)
        print("CENTRALIZED BASELINE - RESULTS SUMMARY")
        print("=" * 70)
        
        print(f"\nTraining Performance:")
        print(f"  Accuracy: {self.train_metrics['accuracy']:.4f}")
        print(f"  Loss:     {self.train_metrics['loss']:.4f}")
        
        print(f"\nTest Performance (Generalization):")
        print(f"  Accuracy:  {self.test_metrics['accuracy']:.4f}")
        print(f"  Precision: {self.test_metrics['precision']:.4f}")
        print(f"  Recall:    {self.test_metrics['recall']:.4f}")
        print(f"  F1-Score:  {self.test_metrics['f1_score']:.4f}")
        print(f"  Loss:      {self.test_metrics['loss']:.4f}")
        
        print(f"\nConfusion Matrix (Test):")
        # [TN, FP]
        # [FN, TP]
        cm = self.test_metrics['confusion_matrix']
        print(f"  True Negatives:  {cm[0, 0]}")
        print(f"  False Positives: {cm[0, 1]}")
        print(f"  False Negatives: {cm[1, 0]}")
        print(f"  True Positives:  {cm[1, 1]}")
        
        print(f"\nTraining Efficiency:")
        print(f"  Training Time: {self.training_time:.2f} seconds")
        print(f"  Samples/Second: {len(self.X_train) / self.training_time:.0f}")
        
        print("\n" + "=" * 70)


def train_centralized_baseline(verbose=True, save_results=True):
    """
    Convenience function to run centralized baseline training.
    
    Args:
        verbose (bool): Print progress information
        save_results (bool): Save results to JSON file
        
    Returns:
        tuple: (trainer, results_path)
    """
    trainer = CentralizedTrainer()
    results = trainer.run(verbose=verbose)
    trainer.print_comparison_summary()
    
    results_path = None
    if save_results:
        results_path = trainer.save_results()
    
    return trainer, results_path
