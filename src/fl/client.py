"""Federated Learning Client Implementation

Client-side logic for federated learning using Flower framework.
Each client trains locally and sends updated weights to server.
"""

import numpy as np
from typing import Tuple, Dict, Any
import flwr as fl
from src.models.model import LogisticRegressionModel


class FLClient(fl.client.NumPyClient):
    """Flower client for federated learning with healthcare models."""
    
    def __init__(self, model: LogisticRegressionModel, X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray):
        """
        Initialize federated client.
        
        Args:
            model: LogisticRegressionModel instance
            X_train: Training features (local client data)
            y_train: Training labels (local client data)
            X_test: Test features (for evaluation)
            y_test: Test labels (for evaluation)
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
    def get_parameters(self, config: Dict[str, Any]) -> Tuple:
        """
        Get current model weights.
        
        Called by server to retrieve model weights before aggregation.
        
        Args:
            config: Server configuration
            
        Returns:
            Tuple of numpy arrays (model weights)
        """
        return self.model.get_weights()
    
    def set_parameters(self, parameters: Tuple) -> None:
        """
        Set model weights from server.
        
        Called by server to distribute aggregated weights.
        
        Args:
            parameters: Tuple of numpy arrays (aggregated weights)
        """
        self.model.set_weights(parameters)
    
    def fit(self, parameters: Tuple, config: Dict[str, Any]) -> Tuple[Tuple, int, Dict]:
        """
        Train model locally and return updated weights.
        
        Called by server each round to perform local training.
        
        Args:
            parameters: Initial model weights from server
            config: Server configuration (epochs, batch size, etc.)
            
        Returns:
            Tuple of:
                - Updated weights
                - Number of training samples
                - Metrics dictionary
        """
        # Set weights from server
        self.set_parameters(parameters)
        
        # Local training
        epochs = config.get('epochs', 1)
        for epoch in range(epochs):
            self.model.fit(self.X_train, self.y_train, verbose=False)
        
        # Return updated weights
        weights = self.get_parameters({})
        
        # Calculate training metrics
        from sklearn.metrics import accuracy_score
        y_pred = self.model.predict(self.X_train)
        train_acc = accuracy_score(self.y_train, y_pred)
        
        metrics = {
            'train_accuracy': float(train_acc),
            'num_samples': len(self.X_train)
        }
        
        return weights, len(self.X_train), metrics
    
    def evaluate(self, parameters: Tuple, config: Dict[str, Any]) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local test set.
        
        Called by server to assess model performance.
        
        Args:
            parameters: Model weights to evaluate
            config: Server configuration
            
        Returns:
            Tuple of:
                - Loss value
                - Number of test samples
                - Metrics dictionary
        """
        # Set weights from server
        self.set_parameters(parameters)
        
        # Evaluate on local test set
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, confusion_matrix
        )
        
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        
        # Loss (using 1 - accuracy as simple loss)
        loss = 1.0 - acc
        
        metrics = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
        }
        
        return loss, len(self.X_test), metrics


def make_flask_client(model: LogisticRegressionModel, 
                     X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray) -> FLClient:
    """
    Factory function to create a Flower client.
    
    Args:
        model: LogisticRegressionModel instance
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        FLClient instance
    """
    return FLClient(model, X_train, y_train, X_test, y_test)
