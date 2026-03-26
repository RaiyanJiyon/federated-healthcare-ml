"""Data preprocessing module for feature normalization and handling missing values"""
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

class DataPreprocessor:
    """Handles data preprocessing including normalization and missing value imputation."""
    
    # Columns where 0 values are physiologically impossible (for PIMA Indians dataset)
    INVALID_ZERO_COLUMNS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.scaler = StandardScaler()
        self.scaler_fitted = False
    
    def handle_invalid_zeros(self, X):
        """
        Handle invalid zero values in healthcare data.
        Replaces physically impossible zeros with NaN in specific columns.
        
        For PIMA Indians dataset:
        - Glucose cannot be 0
        - BloodPressure cannot be 0
        - SkinThickness cannot be 0
        - Insulin cannot be 0
        - BMI cannot be 0
        
        Args:
            X (np.ndarray or pd.DataFrame): Feature matrix
            
        Returns:
            np.ndarray or pd.DataFrame: Matrix with invalid zeros replaced by NaN
        """
        X_processed = X.copy()
        
        if isinstance(X_processed, pd.DataFrame):
            # Work with column names
            cols_to_check = [col for col in self.INVALID_ZERO_COLUMNS if col in X_processed.columns]
            
            zero_counts = {}
            for col in cols_to_check:
                zero_count = (X_processed[col] == 0).sum()
                if zero_count > 0:
                    zero_counts[col] = zero_count
                    X_processed[col] = X_processed[col].replace(0, np.nan)
            
            if zero_counts:
                print(f"  Replaced invalid zeros with NaN:")
                for col, count in zero_counts.items():
                    print(f"    - {col}: {count} zeros → NaN")
        else:
            # For numpy arrays, we can't identify columns by name
            print("  Note: Pass DataFrame if you want to handle invalid zeros in specific columns")
        
        return X_processed
    
    def handle_missing_values(self, X, use_median=True):
        """
        Handle missing values by replacing with median (or mean) of each column.
        Median is more robust for healthcare data with outliers.
        
        Args:
            X (np.ndarray or pd.DataFrame): Feature matrix
            use_median (bool): Use median for imputation (recommended), otherwise use mean
            
        Returns:
            np.ndarray or pd.DataFrame: Feature matrix with missing values handled
        """
        X_processed = X.copy()
        
        # Check for missing values
        if isinstance(X_processed, pd.DataFrame):
            missing_mask = X_processed.isna()
            if missing_mask.any().any():
                missing_count = missing_mask.sum().sum()
                print(f"  Found {missing_count} missing values")
                
                if use_median:
                    X_processed = X_processed.fillna(X_processed.median())
                    print(f"  Imputed with median values (more robust for healthcare data)")
                else:
                    X_processed = X_processed.fillna(X_processed.mean())
                    print(f"  Imputed with mean values")
        else:
            missing_mask = np.isnan(X_processed)
            if np.any(missing_mask):
                for col in range(X_processed.shape[1]):
                    if use_median:
                        col_stat = np.nanmedian(X_processed[:, col])
                    else:
                        col_stat = np.nanmean(X_processed[:, col])
                    X_processed[missing_mask[:, col], col] = col_stat
                stat_type = "median" if use_median else "mean"
                print(f"  Found and imputed missing values using {stat_type}")
        
        return X_processed
    
    def normalize_features(self, X, fit=True):
        """
        Normalize features using StandardScaler (zero mean, unit variance).
        
        Args:
            X (np.ndarray): Feature matrix
            fit (bool): Whether to fit the scaler or use existing fit
            
        Returns:
            np.ndarray: Normalized feature matrix
        """
        if fit:
            X_normalized = self.scaler.fit_transform(X)
            self.scaler_fitted = True
            print(f"  Scaler fitted on training data")
        else:
            if not self.scaler_fitted:
                raise ValueError("Scaler not fitted. Call normalize_features with fit=True first.")
            X_normalized = self.scaler.transform(X)
            print(f"  Features normalized using fitted scaler")
        
        return X_normalized
    
    def preprocess(self, X, fit=True):
        """
        Apply full preprocessing pipeline:
        1. Replace invalid zeros with NaN
        2. Impute missing values with median
        3. Normalize features
        
        Args:
            X (np.ndarray or pd.DataFrame): Feature matrix
            fit (bool): Whether to fit the scaler
            
        Returns:
            np.ndarray or pd.DataFrame: Preprocessed feature matrix
        """
        print("Preprocessing data...")
        
        # Step 1: Handle invalid zeros (physiologically impossible values)
        X_processed = self.handle_invalid_zeros(X)
        
        # Step 2: Handle missing values
        X_processed = self.handle_missing_values(X_processed, use_median=True)
        
        # Step 3: Normalize features
        X_processed = self.normalize_features(X_processed, fit=fit)
        
        print("  ✓ Preprocessing complete!")
        
        return X_processed
    
    def get_feature_stats(self, X):
        """
        Get statistics of features (mean, std).
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            dict: Dictionary with mean and std for each feature
        """
        return {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0)
        }
