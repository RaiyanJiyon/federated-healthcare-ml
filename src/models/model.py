"""Model definitions for the federated healthcare ML project"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
from io import BytesIO


class LogisticRegressionModel:
    """
    Logistic Regression model wrapper for federalized healthcare ML.
    Provides interface for training, prediction, and weight management for federated learning.
    """
    
    def __init__(self, learning_rate=0.01, max_iter=100, random_state=42):
        """
        Initialize the logistic regression model.
        
        Args:
            learning_rate (float): Learning rate for optimization (affects solver)
            max_iter (int): Maximum number of iterations for convergence
            random_state (int): Random state for reproducibility
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Initialize scikit-learn logistic regression model
        self.model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs',  # L-BFGS solver works well for small to medium datasets
            C=1.0,  # Inverse of regularization strength
            verbose=0
        )
        
        self.is_trained = False
        self.n_features = None
    
    def fit(self, X_train, y_train, verbose=False):
        """
        Train the logistic regression model.
        
        Args:
            X_train (np.ndarray): Training feature matrix
            y_train (np.ndarray): Training labels
            verbose (bool): Print training information
            
        Returns:
            dict: Training metrics (loss approximated by negative log-likelihood)
        """
        if X_train.shape[0] == 0:
            raise ValueError("Training data cannot be empty")
        
        self.n_features = X_train.shape[1]
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred)
        
        # For logistic regression, approximate loss as -log(likelihood)
        train_loss = self._calculate_loss(X_train, y_train)
        
        if verbose:
            print(f"  Training complete:")
            print(f"    - Accuracy: {train_accuracy:.4f}")
            print(f"    - Loss (neg log-likelihood): {train_loss:.4f}")
        
        return {
            'accuracy': train_accuracy,
            'loss': train_loss
        }
    
    def predict(self, X):
        """
        Make predictions on data.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predicted labels (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Probability predictions for each class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test, verbose=False):
        """
        Evaluate model on test data.
        
        Args:
            X_test (np.ndarray): Test feature matrix
            y_test (np.ndarray): Test labels
            verbose (bool): Print evaluation metrics
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        loss = self._calculate_loss(X_test, y_test)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'loss': loss
        }
        
        if verbose:
            print(f"Evaluation Metrics:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  Loss:      {loss:.4f}")
            print(f"  Confusion Matrix:")
            print(f"    {cm}")
        
        return metrics
    
    def get_weights(self):
        """
        Get model weights (coefficients and bias).
        Used for federated learning weight aggregation.
        
        Returns:
            dict: Dictionary with 'coef', 'intercept', and 'classes' keys
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting weights")
        
        return {
            'coef': self.model.coef_.flatten().copy(),
            'intercept': self.model.intercept_.copy(),
            'classes': self.model.classes_.copy()
        }
    
    def set_weights(self, weights):
        """
        Set model weights (coefficients, bias, and classes).
        Used in federated learning to apply aggregated weights.
        
        Args:
            weights (dict): Dictionary with 'coef', 'intercept', and 'classes' keys
        """
        if isinstance(weights, dict):
            coef = weights['coef']
            intercept = weights['intercept']
            classes = weights.get('classes', np.array([0, 1]))
        else:
            raise ValueError("Weights must be a dictionary with 'coef', 'intercept', and 'classes' keys")
        
        # Reshape coefficients to match sklearn format
        self.model.coef_ = coef.reshape(1, -1)
        self.model.intercept_ = np.array(intercept)
        self.model.classes_ = np.array(classes)
        self.is_trained = True
    
    def get_weights_dict(self):
        """
        Get weights as a dictionary for easier handling.
        
        Returns:
            dict: Dictionary with 'coef', 'intercept', and 'classes' keys
        """
        return self.get_weights()
    
    def set_weights_dict(self, weights_dict):
        """
        Set weights from a dictionary.
        
        Args:
            weights_dict (dict): Dictionary with 'coef', 'intercept', and 'classes' keys
        """
        self.set_weights(weights_dict)
    
    def serialize_weights(self):
        """
        Serialize weights to bytes (for transmission in federated learning).
        
        Returns:
            bytes: Serialized weights
        """
        weights = self.get_weights()
        buffer = BytesIO()
        np.savez(buffer, coef=weights['coef'], intercept=weights['intercept'], classes=weights['classes'])
        buffer.seek(0)
        return buffer.read()
    
    @staticmethod
    def deserialize_weights(weights_bytes):
        """
        Deserialize weights from bytes.
        
        Args:
            weights_bytes (bytes): Serialized weight bytes
            
        Returns:
            dict: Deserialized weights dictionary
        """
        buffer = BytesIO(weights_bytes)
        loaded = np.load(buffer)
        return {
            'coef': loaded['coef'],
            'intercept': loaded['intercept'],
            'classes': loaded['classes']
        }
    
    def get_num_parameters(self):
        """
        Get total number of trainable parameters.
        
        Returns:
            int: Total number of parameters
        """
        if not self.is_trained:
            return None
        
        weights = self.get_weights()
        return len(weights['coef']) + len(weights['intercept'])
    
    def _calculate_loss(self, X, y):
        """
        Calculate loss (negative log-likelihood).
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Labels
            
        Returns:
            float: Loss value
        """
        # Get probability predictions
        y_pred_proba = self.predict_proba(X)
        
        # Calculate log loss (cross-entropy)
        epsilon = 1e-15  # Prevent log(0)
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        
        # Calculate binary cross-entropy
        loss = -np.mean(y * np.log(y_pred_proba[:, 1]) + 
                       (1 - y) * np.log(y_pred_proba[:, 0]))
        
        return loss
    
    def reset(self):
        """Reset the model to untrained state."""
        self.model = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver='lbfgs',
            C=1.0
        )
        self.is_trained = False
        self.n_features = None