"""Data splitting and distribution module for federated learning"""
import numpy as np
from sklearn.model_selection import train_test_split
from src.config.config import TEST_SIZE, RANDOM_SEED, NUM_CLIENTS, DIRICHLET_ALPHA, NON_IID


def train_test_split_data(X, y):
    """
    Split data into training and testing sets.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target labels
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"Train-test split completed (80-20):")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def distribute_iid(X, y, num_clients):
    """
    Distribute data to clients with IID (Independent and Identically Distributed) assumption.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target labels
        num_clients (int): Number of clients to distribute data to
        
    Returns:
        dict: Dictionary mapping client_id to (X_client, y_client)
    """
    num_samples = len(X)
    samples_per_client = num_samples // num_clients
    
    client_data = {}
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for client_id in range(num_clients):
        start_idx = client_id * samples_per_client
        end_idx = start_idx + samples_per_client if client_id < num_clients - 1 else num_samples
        
        client_indices = indices[start_idx:end_idx]
        client_data[client_id] = (X[client_indices], y[client_indices])
    
    print(f"IID distribution to {num_clients} clients:")
    for client_id, (X_c, y_c) in client_data.items():
        print(f"  Client {client_id}: {len(X_c)} samples, "
              f"class distribution: {np.unique(y_c, return_counts=True)}")
    
    return client_data


def distribute_non_iid(X, y, num_clients, alpha=DIRICHLET_ALPHA):
    """
    Distribute data to clients with Non-IID (non-uniform) distribution using Dirichlet.
    Simulates realistic healthcare scenarios where different clients have different patient populations.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target labels
        num_clients (int): Number of clients
        alpha (float): Dirichlet concentration parameter (lower = more non-IID)
        
    Returns:
        dict: Dictionary mapping client_id to (X_client, y_client)
    """
    num_samples = len(X)
    num_classes = len(np.unique(y))
    
    # Create partitions for each class using Dirichlet distribution
    client_data = {client_id: [] for client_id in range(num_clients)}
    
    # For each class, distribute samples to clients using Dirichlet
    for class_id in range(num_classes):
        class_indices = np.where(y == class_id)[0]
        np.random.shuffle(class_indices)
        
        # Generate client weights for this class using Dirichlet
        client_weights = np.random.dirichlet(np.ones(num_clients) * alpha)
        
        # Distribute class samples to clients
        start_idx = 0
        for client_id in range(num_clients):
            num_samples_client = int(len(class_indices) * client_weights[client_id])
            end_idx = start_idx + num_samples_client
            
            if client_id == num_clients - 1:
                end_idx = len(class_indices)
            
            client_data[client_id].extend(class_indices[start_idx:end_idx])
            start_idx = end_idx
    
    # Convert to numpy arrays and prepare final data
    final_client_data = {}
    for client_id in range(num_clients):
        if len(client_data[client_id]) > 0:
            indices = np.array(client_data[client_id])
            final_client_data[client_id] = (X[indices], y[indices])
        else:
            # If client has no samples, give random sample (edge case)
            random_idx = np.random.choice(len(X), size=max(1, len(X)//num_clients))
            final_client_data[client_id] = (X[random_idx], y[random_idx])
    
    print(f"Non-IID distribution to {num_clients} clients (alpha={alpha}):")
    for client_id, (X_c, y_c) in final_client_data.items():
        unique_classes, counts = np.unique(y_c, return_counts=True)
        class_dist = {f"class_{c}": cnt for c, cnt in zip(unique_classes, counts)}
        print(f"  Client {client_id}: {len(X_c)} samples, distribution: {class_dist}")
    
    return final_client_data


def distribute_data(X, y, num_clients=NUM_CLIENTS, non_iid=NON_IID):
    """
    Main function to distribute data to clients.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target labels
        num_clients (int): Number of clients
        non_iid (bool): Whether to use Non-IID distribution
        
    Returns:
        dict: Dictionary mapping client_id to (X_client, y_client)
    """
    print(f"\nDistributing data to {num_clients} clients ({'Non-IID' if non_iid else 'IID'})...")
    
    if non_iid:
        return distribute_non_iid(X, y, num_clients)
    else:
        return distribute_iid(X, y, num_clients)


def get_client_data(client_data, client_id):
    """
    Retrieve data for a specific client.
    
    Args:
        client_data (dict): Client data dictionary
        client_id (int): Client identifier
        
    Returns:
        tuple: (X_client, y_client)
    """
    if client_id not in client_data:
        raise ValueError(f"Client {client_id} not found in client data")
    
    return client_data[client_id]
