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

# ===== MIMIC-IV Cohort SQL Query with Clinical Features =====
MIMIC_IV_COHORT_SQL = f"""
WITH first_icu AS (
  SELECT 
    subject_id, hadm_id, stay_id, first_careunit,
    intime, outtime
  FROM `{BQ_DATASET_ICU}.icustays`
  WHERE first_careunit IS NOT NULL
    AND DATETIME_DIFF(outtime, intime, HOUR) >= {COHORT_MIN_ICU_LOS_HOURS}
  QUALIFY ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime ASC) = 1
),
cohort_base AS (
  SELECT 
    f.subject_id, f.hadm_id, f.stay_id, f.first_careunit,
    f.intime, f.outtime,
    EXTRACT(YEAR FROM a.admittime) - p.anchor_year + p.anchor_age AS age,
    CASE WHEN p.gender = 'M' THEN 1.0 ELSE 0.0 END AS gender_M,
    CASE WHEN a.admission_type IN ('EMERGENCY', 'URGENT') THEN 1.0 ELSE 0.0 END AS admission_emergency,
    CASE WHEN a.insurance LIKE '%Medicare%' THEN 1.0 ELSE 0.0 END AS insurance_medicare,
    a.hospital_expire_flag
  FROM first_icu f
  JOIN `{BQ_DATASET_HOSP}.patients` p USING(subject_id)
  JOIN `{BQ_DATASET_HOSP}.admissions` a USING(hadm_id)
  WHERE EXTRACT(YEAR FROM a.admittime) - p.anchor_year + p.anchor_age >= {COHORT_MIN_AGE}
    AND a.hospital_expire_flag IN (0, 1)
),
-- Heart Rate (chartevents itemid 220045)
hr_24h AS (
  SELECT stay_id, 
    ROUND(AVG(CAST(valuenum AS FLOAT64)), 2) as heart_rate_mean,
    ROUND(MIN(CAST(valuenum AS FLOAT64)), 2) as heart_rate_min,
    ROUND(MAX(CAST(valuenum AS FLOAT64)), 2) as heart_rate_max
  FROM `{BQ_DATASET_ICU}.chartevents`
  WHERE itemid = 220045 AND valuenum IS NOT NULL AND valuenum BETWEEN 0 AND 300
    AND stay_id IN (SELECT stay_id FROM cohort_base)
  GROUP BY stay_id
),
-- Systolic BP (chartevents itemid 220050)
sbp_24h AS (
  SELECT stay_id, 
    ROUND(AVG(CAST(valuenum AS FLOAT64)), 2) as sbp_mean,
    ROUND(MIN(CAST(valuenum AS FLOAT64)), 2) as sbp_min
  FROM `{BQ_DATASET_ICU}.chartevents`
  WHERE itemid = 220050 AND valuenum IS NOT NULL AND valuenum BETWEEN 0 AND 400
    AND stay_id IN (SELECT stay_id FROM cohort_base)
  GROUP BY stay_id
),
-- Diastolic BP (chartevents itemid 220051, use as proxy for MBP)
mbp_24h AS (
  SELECT stay_id, 
    ROUND(AVG(CAST(valuenum AS FLOAT64)), 2) as mbp_mean,
    ROUND(MIN(CAST(valuenum AS FLOAT64)), 2) as mbp_min
  FROM `{BQ_DATASET_ICU}.chartevents`
  WHERE itemid = 220051 AND valuenum IS NOT NULL AND valuenum BETWEEN 0 AND 300
    AND stay_id IN (SELECT stay_id FROM cohort_base)
  GROUP BY stay_id
),
-- Respiratory Rate (chartevents itemid 220210)
rr_24h AS (
  SELECT stay_id, 
    ROUND(AVG(CAST(valuenum AS FLOAT64)), 2) as resp_rate_mean,
    ROUND(MAX(CAST(valuenum AS FLOAT64)), 2) as resp_rate_max
  FROM `{BQ_DATASET_ICU}.chartevents`
  WHERE itemid = 220210 AND valuenum IS NOT NULL AND valuenum BETWEEN 0 AND 60
    AND stay_id IN (SELECT stay_id FROM cohort_base)
  GROUP BY stay_id
),
-- Temperature (chartevents itemid 223762 = Celsius)
temp_24h AS (
  SELECT stay_id, ROUND(AVG(CAST(valuenum AS FLOAT64)), 2) as temperature_mean
  FROM `{BQ_DATASET_ICU}.chartevents`
  WHERE itemid = 223762 AND valuenum IS NOT NULL AND valuenum BETWEEN 25 AND 45
    AND stay_id IN (SELECT stay_id FROM cohort_base)
  GROUP BY stay_id
),
-- SpO2 (chartevents itemid 220277)
spo2_24h AS (
  SELECT stay_id, 
    ROUND(AVG(CAST(valuenum AS FLOAT64)), 2) as spo2_mean,
    ROUND(MIN(CAST(valuenum AS FLOAT64)), 2) as spo2_min
  FROM `{BQ_DATASET_ICU}.chartevents`
  WHERE itemid = 220277 AND valuenum IS NOT NULL AND valuenum BETWEEN 0 AND 100
    AND stay_id IN (SELECT stay_id FROM cohort_base)
  GROUP BY stay_id
),
-- Glucose (labevents itemid 50809, from hosp dataset, use hadm_id!)
glucose_24h AS (
  SELECT cb.stay_id, ROUND(AVG(CAST(l.valuenum AS FLOAT64)), 2) as glucose_mean
  FROM cohort_base cb
  JOIN `{BQ_DATASET_HOSP}.labevents` l USING(hadm_id)
  WHERE l.itemid = 50809 AND l.valuenum IS NOT NULL AND l.valuenum BETWEEN 0 AND 1000
    AND l.charttime >= cb.intime AND l.charttime < DATETIME_ADD(cb.intime, INTERVAL 24 HOUR)
  GROUP BY cb.stay_id
),
-- Creatinine (labevents itemid 50912)
creatinine_24h AS (
  SELECT cb.stay_id, ROUND(MAX(CAST(l.valuenum AS FLOAT64)), 2) as creatinine_max
  FROM cohort_base cb
  JOIN `{BQ_DATASET_HOSP}.labevents` l USING(hadm_id)
  WHERE l.itemid = 50912 AND l.valuenum IS NOT NULL AND l.valuenum BETWEEN 0 AND 10
    AND l.charttime >= cb.intime AND l.charttime < DATETIME_ADD(cb.intime, INTERVAL 24 HOUR)
  GROUP BY cb.stay_id
),
-- BUN (labevents itemid 51006)
bun_24h AS (
  SELECT cb.stay_id, ROUND(MAX(CAST(l.valuenum AS FLOAT64)), 2) as bun_max
  FROM cohort_base cb
  JOIN `{BQ_DATASET_HOSP}.labevents` l USING(hadm_id)
  WHERE l.itemid = 51006 AND l.valuenum IS NOT NULL AND l.valuenum BETWEEN 0 AND 200
    AND l.charttime >= cb.intime AND l.charttime < DATETIME_ADD(cb.intime, INTERVAL 24 HOUR)
  GROUP BY cb.stay_id
),
-- Sodium (labevents itemid 50983)
sodium_24h AS (
  SELECT cb.stay_id, 
    ROUND(MIN(CAST(l.valuenum AS FLOAT64)), 2) as sodium_min,
    ROUND(MAX(CAST(l.valuenum AS FLOAT64)), 2) as sodium_max
  FROM cohort_base cb
  JOIN `{BQ_DATASET_HOSP}.labevents` l USING(hadm_id)
  WHERE l.itemid = 50983 AND l.valuenum IS NOT NULL AND l.valuenum BETWEEN 100 AND 160
    AND l.charttime >= cb.intime AND l.charttime < DATETIME_ADD(cb.intime, INTERVAL 24 HOUR)
  GROUP BY cb.stay_id
),
-- Potassium (labevents itemid 50971)
potassium_24h AS (
  SELECT cb.stay_id, ROUND(MAX(CAST(l.valuenum AS FLOAT64)), 2) as potassium_max
  FROM cohort_base cb
  JOIN `{BQ_DATASET_HOSP}.labevents` l USING(hadm_id)
  WHERE l.itemid = 50971 AND l.valuenum IS NOT NULL AND l.valuenum BETWEEN 0 AND 10
    AND l.charttime >= cb.intime AND l.charttime < DATETIME_ADD(cb.intime, INTERVAL 24 HOUR)
  GROUP BY cb.stay_id
),
-- Hemoglobin (labevents itemid 50811)
hemoglobin_24h AS (
  SELECT cb.stay_id, ROUND(MIN(CAST(l.valuenum AS FLOAT64)), 2) as hemoglobin_min
  FROM cohort_base cb
  JOIN `{BQ_DATASET_HOSP}.labevents` l USING(hadm_id)
  WHERE l.itemid = 50811 AND l.valuenum IS NOT NULL AND l.valuenum BETWEEN 0 AND 20
    AND l.charttime >= cb.intime AND l.charttime < DATETIME_ADD(cb.intime, INTERVAL 24 HOUR)
  GROUP BY cb.stay_id
),
-- WBC (labevents itemid 51301)
wbc_24h AS (
  SELECT cb.stay_id, ROUND(MAX(CAST(l.valuenum AS FLOAT64)), 2) as wbc_max
  FROM cohort_base cb
  JOIN `{BQ_DATASET_HOSP}.labevents` l USING(hadm_id)
  WHERE l.itemid = 51301 AND l.valuenum IS NOT NULL AND l.valuenum BETWEEN 0 AND 500
    AND l.charttime >= cb.intime AND l.charttime < DATETIME_ADD(cb.intime, INTERVAL 24 HOUR)
  GROUP BY cb.stay_id
),
-- Bicarbonate (labevents itemid 50882)
bicarbonate_24h AS (
  SELECT cb.stay_id, ROUND(MIN(CAST(l.valuenum AS FLOAT64)), 2) as bicarbonate_min
  FROM cohort_base cb
  JOIN `{BQ_DATASET_HOSP}.labevents` l USING(hadm_id)
  WHERE l.itemid = 50882 AND l.valuenum IS NOT NULL AND l.valuenum BETWEEN 0 AND 60
    AND l.charttime >= cb.intime AND l.charttime < DATETIME_ADD(cb.intime, INTERVAL 24 HOUR)
  GROUP BY cb.stay_id
),
-- Lactate (labevents itemid 50954)
lactate_24h AS (
  SELECT cb.stay_id, ROUND(MAX(CAST(l.valuenum AS FLOAT64)), 2) as lactate_max
  FROM cohort_base cb
  JOIN `{BQ_DATASET_HOSP}.labevents` l USING(hadm_id)
  WHERE l.itemid = 50954 AND l.valuenum IS NOT NULL AND l.valuenum BETWEEN 0 AND 50
    AND l.charttime >= cb.intime AND l.charttime < DATETIME_ADD(cb.intime, INTERVAL 24 HOUR)
  GROUP BY cb.stay_id
),
-- Total Bilirubin (labevents itemid 50885)
bilirubin_24h AS (
  SELECT cb.stay_id, ROUND(MAX(CAST(l.valuenum AS FLOAT64)), 2) as bilirubin_total_max
  FROM cohort_base cb
  JOIN `{BQ_DATASET_HOSP}.labevents` l USING(hadm_id)
  WHERE l.itemid = 50885 AND l.valuenum IS NOT NULL AND l.valuenum BETWEEN 0 AND 50
    AND l.charttime >= cb.intime AND l.charttime < DATETIME_ADD(cb.intime, INTERVAL 24 HOUR)
  GROUP BY cb.stay_id
)
SELECT 
  cb.subject_id, cb.hadm_id, cb.stay_id, cb.first_careunit,
  cb.age, cb.gender_M, cb.admission_emergency, cb.insurance_medicare,
  cb.hospital_expire_flag,
  COALESCE(hr.heart_rate_mean, 0) as heart_rate_mean,
  COALESCE(hr.heart_rate_min, 0) as heart_rate_min,
  COALESCE(hr.heart_rate_max, 0) as heart_rate_max,
  COALESCE(sbp.sbp_mean, 0) as sbp_mean,
  COALESCE(sbp.sbp_min, 0) as sbp_min,
  COALESCE(mbp.mbp_mean, 0) as mbp_mean,
  COALESCE(mbp.mbp_min, 0) as mbp_min,
  COALESCE(rr.resp_rate_mean, 0) as resp_rate_mean,
  COALESCE(rr.resp_rate_max, 0) as resp_rate_max,
  COALESCE(temp.temperature_mean, 0) as temperature_mean,
  COALESCE(spo2.spo2_mean, 0) as spo2_mean,
  COALESCE(spo2.spo2_min, 0) as spo2_min,
  COALESCE(glucose.glucose_mean, 0) as glucose_mean,
  COALESCE(creatinine.creatinine_max, 0) as creatinine_max,
  COALESCE(bun.bun_max, 0) as bun_max,
  COALESCE(sodium.sodium_min, 0) as sodium_min,
  COALESCE(sodium.sodium_max, 0) as sodium_max,
  COALESCE(potassium.potassium_max, 0) as potassium_max,
  COALESCE(bicarbonate.bicarbonate_min, 0) as bicarbonate_min,
  COALESCE(hemoglobin.hemoglobin_min, 0) as hemoglobin_min,
  COALESCE(wbc.wbc_max, 0) as wbc_max,
  0 as platelet_min,
  COALESCE(lactate.lactate_max, 0) as lactate_max,
  COALESCE(bilirubin.bilirubin_total_max, 0) as bilirubin_total_max,
  0 as inr_max,
  0 as sofa_score,
  0 as sapsii_score,
  0 as charlson_index
FROM cohort_base cb
LEFT JOIN hr_24h hr ON cb.stay_id = hr.stay_id
LEFT JOIN sbp_24h sbp ON cb.stay_id = sbp.stay_id
LEFT JOIN mbp_24h mbp ON cb.stay_id = mbp.stay_id
LEFT JOIN rr_24h rr ON cb.stay_id = rr.stay_id
LEFT JOIN temp_24h temp ON cb.stay_id = temp.stay_id
LEFT JOIN spo2_24h spo2 ON cb.stay_id = spo2.stay_id
LEFT JOIN glucose_24h glucose ON cb.stay_id = glucose.stay_id
LEFT JOIN creatinine_24h creatinine ON cb.stay_id = creatinine.stay_id
LEFT JOIN bun_24h bun ON cb.stay_id = bun.stay_id
LEFT JOIN sodium_24h sodium ON cb.stay_id = sodium.stay_id
LEFT JOIN potassium_24h potassium ON cb.stay_id = potassium.stay_id
LEFT JOIN hemoglobin_24h hemoglobin ON cb.stay_id = hemoglobin.stay_id
LEFT JOIN wbc_24h wbc ON cb.stay_id = wbc.stay_id
LEFT JOIN bicarbonate_24h bicarbonate ON cb.stay_id = bicarbonate.stay_id
LEFT JOIN lactate_24h lactate ON cb.stay_id = lactate.stay_id
LEFT JOIN bilirubin_24h bilirubin ON cb.stay_id = bilirubin.stay_id
ORDER BY cb.subject_id
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
