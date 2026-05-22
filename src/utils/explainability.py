"""
Explainability utilities with SHAP integration.

Provides SHAP-based explanations for logistic regression models 
and federated aggregation of feature importance across clients.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not installed. Install with: pip install shap")

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based explainability for logistic regression models.
    
    Supports both global (feature importance) and local (instance-level) explanations.
    """
    
    def __init__(self, model, X_background: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained logistic regression model
            X_background: Background data for SHAP (e.g., training set sample)
            feature_names: Optional list of feature names
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        self.model = model
        self.X_background = X_background
        self.n_features = X_background.shape[1]
        
        # Set feature names
        if feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(self.n_features)]
        else:
            self.feature_names = feature_names
        
        # Initialize SHAP explainer
        # Using masker parameter for newer SHAP API
        try:
            self.explainer = shap.LinearExplainer(
                model=model,
                masker=X_background,
                feature_names=self.feature_names,
                check_additivity=False
            )
        except TypeError:
            # Fallback for older SHAP API
            self.explainer = shap.LinearExplainer(
                model=model,
                data=X_background,
                feature_names=self.feature_names,
                check_additivity=False
            )
        
        logger.info(f"SHAP explainer initialized with {self.n_features} features")
    
    def explain_global(self, X: np.ndarray) -> Dict:
        """
        Compute global feature importance (mean absolute SHAP values across dataset).
        
        Args:
            X: Data to explain
            
        Returns:
            Dictionary with:
            - shap_values: Full SHAP values (n_samples, n_features)
            - feature_importance: Mean absolute SHAP per feature
            - feature_importance_df: DataFrame for easy analysis/plotting
        """
        shap_values = self.explainer.shap_values(X)
        
        # Handle binary classification (returns list of 2 arrays)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use class 1 (positive class)
        
        # Compute mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame for easy analysis
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap,
            'rank': len(mean_abs_shap) - np.argsort(mean_abs_shap).argsort()  # 1-indexed rank
        }).sort_values('mean_abs_shap', ascending=False)
        
        return {
            'shap_values': shap_values,
            'feature_importance': mean_abs_shap,
            'feature_importance_df': importance_df
        }
    
    def explain_instance(self, x: np.ndarray, top_k: int = 10) -> Dict:
        """
        Explain a single instance using SHAP.
        
        Args:
            x: Single data point (1D array of shape (n_features,))
            top_k: Number of top features to return
            
        Returns:
            Dictionary with instance-level explanations
        """
        x_reshaped = x.reshape(1, -1)
        shap_values = self.explainer.shap_values(x_reshaped)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        shap_values = shap_values[0]
        
        # Get top contributing features
        top_indices = np.argsort(np.abs(shap_values))[-top_k:][::-1]
        
        explanation = {
            'shap_values': shap_values,
            'top_features': [self.feature_names[i] for i in top_indices],
            'top_shap_values': [shap_values[i] for i in top_indices],
            'top_feature_values': [x[i] for i in top_indices]
        }
        
        return explanation
    
    def feature_importance_df(self, X: np.ndarray, top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance as DataFrame.
        
        Args:
            X: Data to explain
            top_k: If provided, only return top k features
            
        Returns:
            DataFrame with feature importance rankings
        """
        result = self.explain_global(X)
        df = result['feature_importance_df']
        
        if top_k is not None:
            df = df.head(top_k)
        
        return df


class FederatedSHAPAggregator:
    """
    Aggregates SHAP-based explanations across federated clients.
    
    Computes per-client feature importance and federated aggregation
    to understand which features drive decisions in different care units.
    """
    
    def __init__(self, feature_names: List[str]):
        """
        Initialize federated SHAP aggregator.
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.client_importances = {}
    
    def explain_client(
        self,
        client_name: str,
        model,
        X_client: np.ndarray,
        X_background: np.ndarray
    ) -> Dict:
        """
        Compute SHAP explanations for a single client.
        
        Args:
            client_name: Name/ID of client (e.g., "MICU")
            model: Client's trained model
            X_client: Client's data to explain
            X_background: Background data for SHAP
            
        Returns:
            Dictionary with client-level importance
        """
        explainer = SHAPExplainer(
            model=model,
            X_background=X_background,
            feature_names=self.feature_names
        )
        
        result = explainer.explain_global(X_client)
        self.client_importances[client_name] = result['feature_importance']
        
        return {
            'client': client_name,
            'feature_importance': result['feature_importance'],
            'importance_df': result['feature_importance_df'],
            'shap_values': result['shap_values']
        }
    
    def aggregate_importance(self) -> pd.DataFrame:
        """
        Aggregate feature importance across all clients.
        
        Returns:
            DataFrame with per-client and mean importance rankings
        """
        if not self.client_importances:
            raise ValueError("No client importances computed yet. Run explain_client() first.")
        
        # Create DataFrame
        importance_data = pd.DataFrame({
            client: importances
            for client, importances in self.client_importances.items()
        }, index=self.feature_names)
        
        # Add mean and std
        importance_data['mean'] = importance_data.mean(axis=1)
        importance_data['std'] = importance_data.std(axis=1)
        importance_data['cv'] = importance_data['std'] / importance_data['mean']  # Coefficient of variation
        
        return importance_data.sort_values('mean', ascending=False)
    
    def feature_drift(self, reference_client: str) -> pd.DataFrame:
        """
        Compute feature importance drift relative to a reference client.
        
        Useful for identifying which features have heterogeneous importance across care units.
        
        Args:
            reference_client: Name of reference client for comparison
            
        Returns:
            DataFrame with absolute and relative drift per feature
        """
        if reference_client not in self.client_importances:
            raise ValueError(f"Reference client '{reference_client}' not found in importances.")
        
        ref_importance = self.client_importances[reference_client]
        
        drift_data = {}
        for client, importance in self.client_importances.items():
            abs_drift = np.abs(importance - ref_importance)
            rel_drift = abs_drift / (ref_importance + 1e-10)  # Avoid division by zero
            
            drift_data[f'{client}_abs'] = abs_drift
            drift_data[f'{client}_rel'] = rel_drift
        
        drift_df = pd.DataFrame(drift_data, index=self.feature_names)
        return drift_df.sort_values(f'{reference_client}_abs', ascending=False)
    
    def summary_plot_data(self, top_k: int = 15) -> pd.DataFrame:
        """
        Prepare data for SHAP summary plot.
        
        Args:
            top_k: Number of top features to include
            
        Returns:
            DataFrame with top features and their importance across clients
        """
        agg = self.aggregate_importance()
        top_features = agg.head(top_k)
        
        return top_features[[col for col in top_features.columns 
                           if col not in ['mean', 'std', 'cv']]]


def compute_federated_feature_importance(
    client_models: Dict[str, any],
    client_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    feature_names: List[str],
    background_data: np.ndarray
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute and aggregate feature importance across federated clients.
    
    Convenience function for end-to-end federated SHAP analysis.
    
    Args:
        client_models: Dict[client_name -> model]
        client_data: Dict[client_name -> (X, y)]
        feature_names: List of feature names
        background_data: Background data for SHAP explainer
        
    Returns:
        Tuple of (aggregated_importance_df, all_client_results)
    """
    aggregator = FederatedSHAPAggregator(feature_names)
    client_results = {}
    
    for client_name, model in client_models.items():
        X_client, _ = client_data[client_name]
        
        result = aggregator.explain_client(
            client_name=client_name,
            model=model,
            X_client=X_client,
            X_background=background_data
        )
        client_results[client_name] = result
        
        logger.info(f"Computed SHAP explanations for {client_name}")
    
    aggregated_df = aggregator.aggregate_importance()
    
    return aggregated_df, client_results


# ============================================================================
# Utility functions for visualization and analysis
# ============================================================================

def get_top_features(importance_df: pd.DataFrame, top_k: int = 10) -> List[str]:
    """
    Get list of top k most important features.
    
    Args:
        importance_df: Output from FederatedSHAPAggregator.aggregate_importance()
        top_k: Number of top features
        
    Returns:
        List of feature names (sorted by importance)
    """
    return importance_df['mean'].nlargest(top_k).index.tolist()


def summarize_client_differences(
    aggregator: FederatedSHAPAggregator,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Identify which features show the most variation across clients.
    
    This helps identify care-unit specific factors.
    
    Args:
        aggregator: FederatedSHAPAggregator with computed importances
        top_k: Number of most variable features to return
        
    Returns:
        DataFrame with top variable features (sorted by std of importance)
    """
    agg = aggregator.aggregate_importance()
    top_variable = agg.nlargest(top_k, 'std')
    
    return top_variable[['mean', 'std', 'cv']]
