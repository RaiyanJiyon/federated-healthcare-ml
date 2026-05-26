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


class MLPModel:
    """
    Multi-Layer Perceptron (Neural Network) model wrapper for federated healthcare ML.
    Uses PyTorch for training and inference.
    Provides interface compatible with federated learning aggregation.
    
    Architecture: 31 (input) -> 64 (hidden) -> 32 (hidden) -> 1 (output)
    """
    
    def __init__(self, input_dim=31, hidden_layers=None, dropout_rate=0.2, 
                 learning_rate=0.001, batch_size=32, epochs=20, random_state=None):
        """
        Initialize the MLP model.
        
        Args:
            input_dim (int): Number of input features (default 31 for MIMIC-IV)
            hidden_layers (list): Hidden layer dimensions (default [64, 32])
            dropout_rate (float): Dropout rate for regularization (default 0.2)
            learning_rate (float): Learning rate for Adam optimizer (default 0.001)
            batch_size (int): Batch size for training (default 32)
            epochs (int): Number of epochs per local training (default 20)
            random_state (int): Random seed for reproducibility
        """
        import torch
        import torch.nn as nn
        
        self.torch = torch
        self.nn = nn
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers if hidden_layers is not None else [64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state if random_state is not None else RANDOM_SEED
        self.decision_threshold = DECISION_THRESHOLD
        
        # Set random seeds for reproducibility
        self.torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Determine device (CPU or GPU)
        self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        
        # Build the neural network
        self.model = self._build_network()
        self.model.to(self.device)
        
        self.is_trained = False
        self.n_features = None
        self.scaler = None
        self.optimizer = None
        self.criterion = None
    
    def _build_network(self):
        """Build the neural network architecture."""
        layers = []
        prev_dim = self.input_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_layers:
            layers.append(self.nn.Linear(prev_dim, hidden_dim))
            layers.append(self.nn.ReLU())
            layers.append(self.nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (sigmoid for binary classification)
        layers.append(self.nn.Linear(prev_dim, 1))
        layers.append(self.nn.Sigmoid())
        
        return self.nn.Sequential(*layers)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=False):
        """
        Train the MLP model.
        
        Args:
            X_train (np.ndarray): Training feature matrix
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation feature matrix (optional)
            y_val (np.ndarray): Validation labels (optional)
            verbose (bool): Print training information
            
        Returns:
            dict: Training metrics
        """
        from sklearn.preprocessing import StandardScaler
        
        if X_train.shape[0] == 0:
            raise ValueError("Training data cannot be empty")
        
        self.n_features = X_train.shape[1]
        if self.n_features != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {self.n_features}")
        
        # Standardize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Convert to PyTorch tensors
        X_train_tensor = self.torch.from_numpy(X_train_scaled).float().to(self.device)
        y_train_tensor = self.torch.from_numpy(y_train.reshape(-1, 1)).float().to(self.device)
        
        # Set up optimizer and loss
        self.optimizer = self.torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = self.nn.BCELoss()
        
        # Training loop
        train_losses = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, len(X_train_tensor), self.batch_size):
                batch_X = X_train_tensor[i:i+self.batch_size]
                batch_y = y_train_tensor[i:i+self.batch_size]
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        
        # Calculate training metrics
        y_train_pred_proba = self.predict_proba(X_train)
        train_auroc = self._calculate_auroc(y_train, y_train_pred_proba)
        
        if verbose:
            print(f"  Training complete:")
            print(f"    - Final Loss: {train_losses[-1]:.4f}")
            print(f"    - Train AUROC: {train_auroc:.4f}")
        
        return {
            'loss': train_losses[-1],
            'auroc': train_auroc
        }
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Probability predictions for positive class (shape: [n_samples])
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Standardize using the same scaler
        X_scaled = self.scaler.transform(X)
        
        # Convert to tensor
        X_tensor = self.torch.from_numpy(X_scaled).float().to(self.device)
        
        # Get predictions
        self.model.eval()
        with self.torch.no_grad():
            proba = self.model(X_tensor).cpu().numpy().flatten()
        
        return proba
    
    def predict(self, X, use_custom_threshold=True):
        """
        Make binary predictions using threshold.
        
        Args:
            X (np.ndarray): Feature matrix
            use_custom_threshold (bool): Use custom threshold instead of 0.5
            
        Returns:
            np.ndarray: Predicted labels (0 or 1)
        """
        proba = self.predict_proba(X)
        threshold = self.decision_threshold if use_custom_threshold else 0.5
        return (proba >= threshold).astype(int)
    
    def set_decision_threshold(self, threshold):
        """Set custom decision threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.decision_threshold = threshold
    
    def get_weights(self):
        """
        Get model weights as a flattened numpy array (for federated averaging).
        
        Returns:
            np.ndarray: Flattened weight vector
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting weights")
        
        weights_list = []
        for param in self.model.parameters():
            weights_list.append(param.data.cpu().numpy().flatten())
        
        return np.concatenate(weights_list)
    
    def set_weights(self, weights):
        """
        Set model weights from a flattened numpy array (federated learning).
        
        Args:
            weights (np.ndarray): Flattened weight vector
        """
        if not isinstance(weights, np.ndarray):
            raise ValueError("Weights must be a numpy array")
        
        offset = 0
        for param in self.model.parameters():
            param_size = param.data.cpu().numpy().size
            param_data = weights[offset:offset+param_size]
            param_data = param_data.reshape(param.data.shape)
            param.data = self.torch.from_numpy(param_data).float().to(self.device)
            offset += param_size
        
        self.is_trained = True
    
    def get_num_parameters(self):
        """Get total number of trainable parameters."""
        if not self.is_trained:
            return sum(p.numel() for p in self.model.parameters())
        return sum(p.numel() for p in self.model.parameters())
    
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
        
        y_pred_proba = self.predict_proba(X_test)
        y_pred = self.predict(X_test)
        
        auroc = self._calculate_auroc(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'auroc': auroc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
        
        if verbose:
            print(f"MLP Evaluation:")
            print(f"  AUROC:     {auroc:.4f}")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  Confusion Matrix:\n{cm}")
        
        return metrics
    
    def _calculate_auroc(self, y_true, y_scores):
        """Calculate AUROC metric."""
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_scores)
    
    def reset(self):
        """Reset the model to untrained state."""
        self.torch.manual_seed(self.random_state)
        self.model = self._build_network()
        self.model.to(self.device)
        self.is_trained = False
        self.n_features = None
        self.scaler = None


# ===== MODEL REGISTRY & FACTORY FUNCTION =====
MODEL_REGISTRY = {
    'logistic_regression': LogisticRegressionModel,
    'mlp': MLPModel,
    'random_forest': RandomForestModel,
    'xgboost': XGBoostModel,
}


def create_model(model_type: str, **kwargs):
    """
    Factory function to create models by type.
    
    Args:
        model_type (str): Type of model ('logistic_regression', 'mlp', 'random_forest', 'xgboost')
        **kwargs: Arguments to pass to the model constructor
        
    Returns:
        Model instance (LogisticRegressionModel, MLPModel, RandomForestModel, or XGBoostModel)
        
    Raises:
        ValueError: If model_type is not in registry
    """
    if model_type not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
    
    return MODEL_REGISTRY[model_type](**kwargs)
