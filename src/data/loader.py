"""Data loading module for the diabetes dataset"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.config.config import DATASET_PATH


def load_dataset():
    """
    Load the diabetes dataset from CSV file.
    
    Returns:
        tuple: (X, y) where X is features and y is target labels
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    
    # Separate features and target
    # Assuming last column is the target variable
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    print(f"Dataset loaded successfully!")
    print(f"  Shape: {X.shape}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Target classes: {np.unique(y)}")
    
    return X, y


def get_feature_names():
    """
    Get feature names from the dataset.
    
    Returns:
        list: Feature column names
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    return df.columns[:-1].tolist()


def load_dataset_with_df():
    """
    Load the dataset and return both dataframe and numpy arrays.
    
    Returns:
        tuple: (df, X, y) where df is the full dataframe
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    return df, X, y
