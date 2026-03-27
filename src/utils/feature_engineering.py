"""Feature Engineering for Healthcare ML

Create meaningful features from raw patient data to improve model performance.
Interaction terms help capture complex disease patterns.
"""

import numpy as np
import pandas as pd


class HealthcareFeatureEngineer:
    """Engineer features from healthcare data to improve predictive power."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_names = None
        self.interaction_pairs = [
            ('Glucose', 'BMI'),           # Glucose metabolism and weight
            ('Glucose', 'Age'),           # Age-related glucose control
            ('BloodPressure', 'BMI'),     # Cardiovascular + weight
            ('Insulin', 'Glucose'),       # Insulin resistance indicator
            ('Age', 'BMI'),               # Age + weight effects
        ]
    
    def create_interaction_features(self, X, feature_names):
        """
        Create interaction features between relevant pairs.
        
        Interaction features help capture non-linear relationships.
        E.g., Glucose×BMI captures how glucose levels interact with weight.
        
        Args:
            X (np.ndarray or pd.DataFrame): Feature matrix
            feature_names (list): Names of features
            
        Returns:
            tuple: (X_engineered, new_feature_names)
        """
        if isinstance(X, pd.DataFrame):
            X_new = X.copy()
            feature_dict = {name: i for i, name in enumerate(feature_names)}
        else:
            X_new = pd.DataFrame(X, columns=feature_names)
            feature_dict = {name: i for i, name in enumerate(feature_names)}
        
        self.feature_names = list(feature_names)
        new_features = {}
        
        print("  Creating interaction features:")
        for feat1, feat2 in self.interaction_pairs:
            if feat1 in feature_dict and feat2 in feature_dict:
                idx1 = feature_dict[feat1]
                idx2 = feature_dict[feat2]
                
                if isinstance(X, pd.DataFrame):
                    interaction = X[feat1] * X[feat2]
                else:
                    interaction = X[:, idx1] * X[:, idx2]
                
                interaction_name = f"{feat1}_x_{feat2}"
                new_features[interaction_name] = interaction
                self.feature_names.append(interaction_name)
                print(f"    ✓ {interaction_name}")
        
        # Combine original and new features
        for col_name, col_data in new_features.items():
            X_new[col_name] = col_data
        
        return X_new.values, self.feature_names
    
    def create_polynomial_features(self, X, feature_names, degree=2):
        """
        Create polynomial features (e.g., Glucose^2, BMI^2).
        
        Polynomial features help capture non-linear relationships.
        
        Args:
            X (np.ndarray or pd.DataFrame): Feature matrix
            feature_names (list): Names of features
            degree (int): Degree of polynomial (2 = quadratic)
            
        Returns:
            tuple: (X_engineered, new_feature_names)
        """
        if isinstance(X, pd.DataFrame):
            X_new = X.copy()
        else:
            X_new = pd.DataFrame(X, columns=feature_names)
        
        self.feature_names = list(feature_names)
        
        # Create polynomial features for important columns
        important_features = ['Glucose', 'BMI', 'Age', 'BloodPressure']
        
        print("  Creating polynomial features:")
        for feat in important_features:
            if feat in feature_names:
                if degree >= 2:
                    X_new[f"{feat}_squared"] = X_new[feat] ** 2
                    self.feature_names.append(f"{feat}_squared")
                    print(f"    ✓ {feat}_squared")
                
                if degree >= 3:
                    X_new[f"{feat}_cubed"] = X_new[feat] ** 3
                    self.feature_names.append(f"{feat}_cubed")
                    print(f"    ✓ {feat}_cubed")
        
        return X_new.values, self.feature_names
    
    def create_ratio_features(self, X, feature_names):
        """
        Create ratio features (e.g., Glucose/BMI).
        
        Ratios can capture meaningful medical relationships.
        
        Args:
            X (np.ndarray or pd.DataFrame): Feature matrix
            feature_names (list): Names of features
            
        Returns:
            tuple: (X_engineered, new_feature_names)
        """
        if isinstance(X, pd.DataFrame):
            X_new = X.copy()
        else:
            X_new = pd.DataFrame(X, columns=feature_names)
        
        self.feature_names = list(feature_names)
        
        ratio_pairs = [
            ('Glucose', 'Insulin'),
            ('BloodPressure', 'Age'),
        ]
        
        print("  Creating ratio features:")
        for feat1, feat2 in ratio_pairs:
            if feat1 in feature_names and feat2 in feature_names:
                # Avoid division by zero
                denominator = X_new[feat2] + 1e-6
                ratio = X_new[feat1] / denominator
                
                ratio_name = f"{feat1}_per_{feat2}"
                X_new[ratio_name] = ratio
                self.feature_names.append(ratio_name)
                print(f"    ✓ {ratio_name}")
        
        return X_new.values, self.feature_names
    
    def engineer_all_features(self, X, feature_names):
        """
        Apply all feature engineering techniques.
        
        Args:
            X (np.ndarray or pd.DataFrame): Feature matrix
            feature_names (list): Names of features
            
        Returns:
            tuple: (X_engineered, new_feature_names)
        """
        print("\n📊 FEATURE ENGINEERING")
        print("=" * 70)
        
        if isinstance(X, pd.DataFrame):
            X_work = X.copy()
        else:
            X_work = pd.DataFrame(X, columns=feature_names)
        
        # Apply all techniques
        X_work, self.feature_names = self.create_interaction_features(X_work, feature_names)
        X_work = pd.DataFrame(X_work, columns=self.feature_names)
        
        X_work, self.feature_names = self.create_polynomial_features(X_work, self.feature_names, degree=2)
        X_work = pd.DataFrame(X_work, columns=self.feature_names)
        
        X_work, self.feature_names = self.create_ratio_features(X_work, self.feature_names)
        X_work = pd.DataFrame(X_work, columns=self.feature_names)
        
        print(f"\n✓ Features: {len(feature_names)} → {len(self.feature_names)}")
        print(f"  Original: {len(feature_names)}")
        print(f"  Engineered: {len(self.feature_names) - len(feature_names)}")
        
        return X_work.values, self.feature_names
