#!/usr/bin/env python
"""
Experiment 5: SHAP Feature Importance Drift Analysis

Analyzes how feature importance varies across federated clients (care units).
Identifies which clinical factors drive mortality predictions in each ICU.

Key questions:
- Which features are universally important vs care-unit specific?
- How much feature importance drifts across care units?
- What does this reveal about clinical heterogeneity?

Results: Feature importance DataFrame, drift analysis, visualization data
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_dataset_with_df
from src.data.split import distribute_by_care_unit
from src.training.federated import FederatedTrainer
from src.config.config import RANDOM_SEED
from src.utils.explainability import FederatedSHAPAggregator, get_top_features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_exp5():
    """
    Analyze SHAP-based feature importance drift across care units.
    """
    logger.info("="*70)
    logger.info("EXPERIMENT 5: SHAP Feature Importance Drift Analysis")
    logger.info("="*70)
    
    # Load data
    logger.info("\n[1/5] Loading MIMIC-IV cohort...")
    df, X, y = load_dataset_with_df(use_cache=True)
    logger.info(f"Cohort loaded: {X.shape[0]} patients, {X.shape[1]} features")
    
    # Split
    logger.info("\n[2/5] Splitting data...")
    n_train = int(0.70 * len(X))
    n_val = int(0.15 * len(X))
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    
    # Get care units
    if 'first_careunit' in df.columns:
        care_units_train = df.iloc[:n_train]['first_careunit']
    else:
        care_units_train = pd.Series(['Unit_' + str(i % 7) for i in range(n_train)])
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Distribute to care units
    logger.info("\n[3/5] Distributing training data to care units...")
    clients = distribute_by_care_unit(X_train, y_train, care_units_train, min_patients_per_unit=100)
    logger.info(f"Created {len(clients)} federated clients:")
    for unit_name in sorted(clients.keys()):
        X_c, y_c = clients[unit_name]
        logger.info(f"  {unit_name}: {len(X_c)} samples")
    
    # Scale data globally
    logger.info("\n[4/5] Training per-client models and computing SHAP explanations...")
    scaler = StandardScaler()
    all_X_train = np.vstack([X for X, _ in clients.values()])
    scaler.fit(all_X_train)
    
    # Create feature names
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    # Initialize SHAP aggregator
    aggregator = FederatedSHAPAggregator(feature_names)
    
    # Train model per client and compute SHAP values
    client_models = {}
    for client_name in sorted(clients.keys()):
        X_client, y_client = clients[client_name]
        X_scaled = scaler.transform(X_client)
        
        # Train local model
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        model.fit(X_scaled, y_client)
        client_models[client_name] = model
        
        # Compute SHAP on background set (sample for speed)
        X_background = X_scaled[:min(200, len(X_scaled))]
        
        try:
            result = aggregator.explain_client(
                client_name=client_name,
                model=model,
                X_client=X_scaled,
                X_background=X_background
            )
            
            top_features = get_top_features(result['importance_df'], top_k=5)
            logger.info(f"  {client_name}: Top features = {top_features}")
        except Exception as e:
            logger.warning(f"  {client_name}: SHAP computation failed ({e})")
    
    # Aggregate across clients
    logger.info("\n[5/5] Analyzing feature importance drift...")
    
    aggregated_importance = aggregator.aggregate_importance()
    logger.info("\nAggregated Feature Importance (all clients):")
    logger.info(aggregated_importance[['mean', 'std', 'cv']].head(10).to_string())
    
    # Identify high-drift features
    high_drift = aggregated_importance.nlargest(10, 'std')
    logger.info("\nHigh-Drift Features (most variable across care units):")
    logger.info(high_drift[['mean', 'std', 'cv']].to_string())
    
    # Save results
    output_dir = Path('results/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save aggregated importance
    agg_file = output_dir / 'exp5_feature_importance_aggregated.csv'
    aggregated_importance.to_csv(agg_file)
    logger.info(f"\n✓ Aggregated importance saved to {agg_file}")
    
    # Save high-drift features
    drift_file = output_dir / 'exp5_high_drift_features.csv'
    high_drift[['mean', 'std', 'cv']].to_csv(drift_file)
    logger.info(f"✓ High-drift features saved to {drift_file}")
    
    # Analysis summary
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS SUMMARY")
    logger.info("="*70)
    
    # Universal importance (low CV)
    universal = aggregated_importance[aggregated_importance['cv'] < 0.2]
    logger.info(f"\nUniversal Features (CV < 0.2): {len(universal)}")
    if len(universal) > 0:
        logger.info(f"  {list(universal.index[:5])}")
    
    # Variable importance (high CV)
    variable = aggregated_importance[aggregated_importance['cv'] > 0.5]
    logger.info(f"\nVariable Features (CV > 0.5): {len(variable)}")
    if len(variable) > 0:
        logger.info(f"  {list(variable.index[:5])}")
    
    # Heterogeneity insight
    mean_cv = aggregated_importance['cv'].mean()
    logger.info(f"\nMean Coefficient of Variation: {mean_cv:.3f}")
    if mean_cv > 0.3:
        logger.info("  → High clinical heterogeneity across care units")
    elif mean_cv > 0.2:
        logger.info("  → Moderate clinical heterogeneity")
    else:
        logger.info("  → Low clinical heterogeneity (universal predictors)")
    
    # Per-client summary
    logger.info("\n" + "="*70)
    logger.info("PER-CLIENT SUMMARIES")
    logger.info("="*70)
    
    client_cols = [col for col in aggregated_importance.columns if col not in ['mean', 'std', 'cv']]
    for client_name in sorted(client_cols):
        top5 = aggregated_importance.nlargest(5, client_name)
        logger.info(f"\n{client_name}:")
        for feat in top5.index:
            logger.info(f"  {feat}: {aggregated_importance.loc[feat, client_name]:.4f}")
    
    return aggregated_importance


if __name__ == '__main__':
    importance_df = run_exp5()
    logger.info("\n✅ EXPERIMENT 5 COMPLETE")
