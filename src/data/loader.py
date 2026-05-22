"""Data loading module for MIMIC-IV from Google BigQuery"""
import pandas as pd
import numpy as np
from pathlib import Path
from google.cloud import bigquery
import logging

from src.config.config import (
    GCP_PROJECT_ID, BQ_BILLING_PROJECT, BQ_PROJECT_PHYSIONET,
    BQ_DATASET_HOSP, BQ_DATASET_ICU, BQ_DATASET_DERIVED,
    COHORT_CACHE_PATH, COHORT_MIN_AGE, COHORT_MIN_ICU_LOS_HOURS,
    COHORT_FIRST_ICU_STAY_ONLY, TARGET_COLUMN, ALL_FEATURES
)

logger = logging.getLogger(__name__)

# ===== MIMIC-IV Cohort SQL Query - Simplified =====
MIMIC_IV_COHORT_SQL = f"""
WITH first_icu AS (
  SELECT 
    subject_id, hadm_id, stay_id, first_careunit,
    intime, outtime
  FROM `{BQ_DATASET_ICU}.icustays`
  WHERE first_careunit IS NOT NULL
    AND DATETIME_DIFF(outtime, intime, HOUR) >= {COHORT_MIN_ICU_LOS_HOURS}
  QUALIFY ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime ASC) = 1
)
SELECT 
  f.subject_id, 
  f.hadm_id, 
  f.stay_id, 
  f.first_careunit,
  EXTRACT(YEAR FROM a.admittime) - p.anchor_year + p.anchor_age AS age,
  CASE WHEN p.gender = 'M' THEN 1.0 ELSE 0.0 END AS gender_M,
  CASE WHEN a.admission_type IN ('EMERGENCY', 'URGENT') THEN 1.0 ELSE 0.0 END AS admission_emergency,
  CASE WHEN a.insurance LIKE '%Medicare%' THEN 1.0 ELSE 0.0 END AS insurance_medicare,
  a.hospital_expire_flag AS hospital_expire_flag,
  -- Placeholder clinical features (to be backfilled from chartevents)
  NULL AS heart_rate_mean,
  NULL AS sbp_mean,
  NULL AS mbp_mean,
  NULL AS resp_rate_mean,
  NULL AS temperature_mean,
  NULL AS spo2_mean,
  NULL AS glucose_mean,
  NULL AS creatinine_max,
  NULL AS bun_max,
  NULL AS sodium_min,
  NULL AS sodium_max,
  NULL AS potassium_max,
  NULL AS bicarbonate_min,
  NULL AS hemoglobin_min,
  NULL AS wbc_max,
  NULL AS platelet_min,
  NULL AS lactate_max,
  NULL AS bilirubin_total_max,
  NULL AS inr_max,
  NULL AS sofa_score,
  NULL AS sapsii_score,
  NULL AS charlson_index
FROM first_icu f
JOIN `{BQ_DATASET_HOSP}.patients` p USING(subject_id)
JOIN `{BQ_DATASET_HOSP}.admissions` a USING(hadm_id)
WHERE EXTRACT(YEAR FROM a.admittime) - p.anchor_year + p.anchor_age >= {COHORT_MIN_AGE}
  AND a.hospital_expire_flag IN (0, 1)
ORDER BY f.subject_id
"""

def load_from_bigquery(billing_project=None):
    """
    Load MIMIC-IV cohort from Google BigQuery.
    
    Args:
        billing_project (str): GCP billing project ID. Defaults to GCP_PROJECT_ID.
    
    Returns:
        pd.DataFrame: MIMIC-IV cohort DataFrame
    """
    if billing_project is None:
        billing_project = GCP_PROJECT_ID
    
    logger.info(f"Connecting to BigQuery (project: {billing_project})")
    client = bigquery.Client(project=billing_project)
    
    logger.info("Running MIMIC-IV cohort query...")
    job_config = bigquery.QueryJobConfig(use_query_cache=False)
    query_job = client.query(MIMIC_IV_COHORT_SQL, job_config=job_config)
    
    df = query_job.to_dataframe()
    logger.info(f"Cohort loaded: {len(df)} rows, {len(df.columns)} columns")
    
    return df

def load_dataset_with_df(use_cache=True, billing_project=None):
    """
    Load MIMIC-IV dataset with cache-first strategy.
    
    Args:
        use_cache (bool): If True, load from cache if available. Defaults to True.
        billing_project (str): GCP billing project for BigQuery. Defaults to GCP_PROJECT_ID.
    
    Returns:
        tuple: (df, X, y) where df is full DataFrame, X is features, y is target
    """
    # Try cache first
    if use_cache and COHORT_CACHE_PATH.exists():
        logger.info(f"Loading cohort from cache: {COHORT_CACHE_PATH}")
        df = pd.read_csv(COHORT_CACHE_PATH)
    else:
        # Load from BigQuery
        df = load_from_bigquery(billing_project=billing_project)
        # Save to cache
        COHORT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(COHORT_CACHE_PATH, index=False)
        logger.info(f"Cohort cached to: {COHORT_CACHE_PATH}")
    
    # Extract features and target
    feature_cols = [col for col in ALL_FEATURES if col in df.columns]
    missing_features = set(ALL_FEATURES) - set(feature_cols)
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    
    X = df[feature_cols].values
    y = df[TARGET_COLUMN].values if TARGET_COLUMN in df.columns else None
    
    logger.info(f"Dataset: {X.shape[0]} patients, {X.shape[1]} features")
    if y is not None:
        logger.info(f"Target distribution: {np.bincount(y.astype(int))}")
    
    return df, X, y

def load_dataset(use_cache=True, billing_project=None):
    """
    Convenience wrapper to load features and target only.
    
    Args:
        use_cache (bool): Use cached cohort if available
        billing_project (str): GCP billing project
    
    Returns:
        tuple: (X, y) numpy arrays
    """
    _, X, y = load_dataset_with_df(use_cache=use_cache, billing_project=billing_project)
    return X, y

def get_feature_names(df=None):
    """
    Get feature names from dataset or config.
    
    Args:
        df (pd.DataFrame, optional): DataFrame to extract features from.
    
    Returns:
        list: Feature column names
    """
    if df is not None:
        return [col for col in ALL_FEATURES if col in df.columns]
    return ALL_FEATURES
