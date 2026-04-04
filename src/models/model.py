"""Model definitions for the federated healthcare ML project"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
from io import BytesIO

# Import configuration
from src.config.config import RANDOM_SEED, MAX_ITER, DECISION_THRESHOLD, LEARNING_RATE, CLASS_WEIGHT


class LogisticRegressionModel:
    """
    Logistic Regression model wrapper for federalized healthcare ML.
    Provides interface for training, prediction, and weight management for federated learning.
    """
    
    def __init__(self, learning_rate=None, max_iter=None, random_state=None, class_weight=None):
        """
        Initialize the logistic regression model.
        
        Args:
            learning_rate (float): Learning rate for optimization (affects solver)
            max_iter (int): Maximum number of iterations for convergence
            random_state (int): Random state for reproducibility
            class_weight (str or dict): Class weight balancing. 'balanced' auto-adjusts for class imbalance
        """
        # Use config values as defaults if not provided
        self.learning_rate = learning_rate if learning_rate is not None else LEARNING_RATE
        self.max_iter = max_iter if max_iter is not None else MAX_ITER
        self.random_state = random_state if random_state is not None else RANDOM_SEED
        self.class_weight = class_weight if class_weight is not None else CLASS_WEIGHT
        self.decision_threshold = DECISION_THRESHOLD  # Use config decision threshold
        
        # Initialize scikit-learn logistic regression model
        # class_weight='balanced' automatically adjusts weights inversely proportional to class frequency
        # This helps with class imbalance (more non-diabetic than diabetic patients)
        self.model = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver='lbfgs',  # L-BFGS solver works well for small to medium datasets
            C=1.0,  # Inverse of regularization strength
            class_weight=self.class_weight,  # Handle class imbalance
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
    
    def predict(self, X, use_custom_threshold=True):
        """
        Make predictions on data with custom decision threshold.
        
        Args:
            X (np.ndarray): Feature matrix
            use_custom_threshold (bool): Use custom threshold instead of default 0.5
            
        Returns:
            np.ndarray: Predicted labels (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if use_custom_threshold and self.decision_threshold != 0.5:
            # Use custom threshold for better recall
            probas = self.model.predict_proba(X)[:, 1]  # Probability of class 1 (diabetic)
            return (probas >= self.decision_threshold).astype(int)
        else:
            # Use default threshold (0.5)
            return self.model.predict(X)
    
    def set_decision_threshold(self, threshold):
        """
        Set custom decision threshold for predictions.
        
        Lower threshold increases recall (catches more diabetic cases)
        but may increase false positives.
        
        Example:
            - threshold=0.5 (default): balanced
            - threshold=0.4: higher recall, more false positives
            - threshold=0.6: higher precision, fewer cases caught
        
        Args:
            threshold (float): Classification threshold (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        self.decision_threshold = threshold
        print(f"  Decision threshold set to {threshold:.2f}")
        print(f"  Effect: {'Higher recall' if threshold < 0.5 else 'Higher precision' if threshold > 0.5 else 'Balanced'}")
    
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
            C=1.0,
            class_weight=self.class_weight
        )
        self.is_trained = False
        self.n_features = None


class RandomForestModel:
    """
    Random Forest model for improved healthcare prediction.
    Better handles non-linear relationships and class imbalance.
    """
    
    def __init__(self, n_estimators=100, max_depth=10, random_state=42, class_weight='balanced_subsample'):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators (int): Number of trees in forest
            max_depth (int): Maximum depth of trees
            random_state (int): Random seed
            class_weight (str): How to handle class imbalance
        """
        from sklearn.ensemble import RandomForestClassifier
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.class_weight = class_weight
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1  # Use all processors
        )
        
        self.is_trained = False
        self.n_features = None
    
    def fit(self, X_train, y_train, verbose=False):
        """Train the model."""
        if X_train.shape[0] == 0:
            raise ValueError("Training data cannot be empty")
        
        self.n_features = X_train.shape[1]
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        y_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred)
        
        if verbose:
            print(f"  RandomForest Training:")
            print(f"    - Accuracy: {train_accuracy:.4f}")
        
        return {'accuracy': train_accuracy}
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test, verbose=False):
        """Evaluate model on test data."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
        }
        
        if verbose:
            print(f"RandomForest Evaluation:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  Confusion Matrix: {cm}")
        
        return metrics


class XGBoostModel:
    """
    XGBoost model for optimal healthcare prediction.
    Handles class imbalance and captures complex patterns in data.
    """
    
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42):
        """
        Initialize XGBoost model.
        
        Args:
            n_estimators (int): Number of boosting rounds
            max_depth (int): Maximum depth of trees
            learning_rate (float): Learning rate / eta
            random_state (int): Random seed
        """
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            scale_pos_weight=1,  # Will be computed from data
            eval_metric='logloss',
            verbosity=0
        )
        
        self.is_trained = False
        self.n_features = None
    
    def fit(self, X_train, y_train, verbose=False):
        """Train the model."""
        if X_train.shape[0] == 0:
            raise ValueError("Training data cannot be empty")
        
        self.n_features = X_train.shape[1]
        
        # Compute scale_pos_weight for class imbalance
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        self.model.set_params(scale_pos_weight=scale_pos_weight)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        y_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred)
        
        if verbose:
            print(f"  XGBoost Training:")
            print(f"    - Accuracy: {train_accuracy:.4f}")
            print(f"    - Scale Pos Weight: {scale_pos_weight:.4f}")
        
        return {'accuracy': train_accuracy}
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test, verbose=False):
        """Evaluate model on test data."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
        }
        
        if verbose:
            print(f"XGBoost Evaluation:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  Confusion Matrix: {cm}")
        
        return metrics
