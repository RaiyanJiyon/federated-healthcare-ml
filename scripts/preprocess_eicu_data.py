#!/usr/bin/env python3
"""
Phase 1.2: Preprocess eICU-CRD cohort and validate data quality

This script:
1. Loads eICU cohort from cache
2. Applies clinical clipping (physiologically valid ranges)
3. Handles data quality issues (gender field all zeros, missing vitals)
4. Compares distributions with MIMIC-IV
5. Saves processed cohort for partitioning (Phase 1.3)

Run after: Phase 1.1 (extract_eicu_cohort.py)
Output: data/cache/eicu_cohort_processed.csv

Usage:
    python scripts/preprocess_eicu_data.py [--compare-mimic]
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import eicu_loader, loader
from src.config.config import GCP_PROJECT_ID

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== CLINICAL CLIPPING RANGES =====
# These represent physiologically valid ranges for ICU data
CLINICAL_CLIPPING_RANGES = {
    # Demographics (no clipping needed)
    'age': (18, 150),  # age should already be filtered
    'gender_M': (0, 1),  # binary
    'admission_emergency': (0, 1),  # binary
    'insurance_medicare': (0, 1),  # binary
    
    # Vitals
    'heart_rate_mean': (20, 250),
    'heart_rate_min': (20, 250),
    'heart_rate_max': (20, 250),
    'sbp_mean': (40, 300),
    'sbp_min': (40, 300),
    'mbp_mean': (20, 200),
    'mbp_min': (20, 200),
    'resp_rate_mean': (4, 60),
    'resp_rate_max': (4, 60),
    'temperature_mean': (25, 45),  # Celsius
    'spo2_mean': (40, 100),  # Percent
    'spo2_min': (40, 100),
    'glucose_mean': (20, 800),  # mg/dL
    
    # Labs
    'creatinine_max': (0.1, 15),  # mg/dL
    'bun_max': (0, 200),  # mg/dL
    'sodium_min': (100, 160),  # mEq/L
    'sodium_max': (100, 160),
    'potassium_max': (0.5, 10),  # mEq/L
    'bicarbonate_min': (5, 60),  # mEq/L
    'hemoglobin_min': (3, 20),  # g/dL
    'wbc_max': (0.5, 500),  # K/uL
    'platelet_min': (5, 1000),  # K/uL
    'lactate_max': (0.5, 50),  # mmol/L
    'bilirubin_total_max': (0.1, 50),  # mg/dL
    'inr_max': (0.5, 20),  # ratio
    'albumin_min': (1, 10),  # g/dL
    
    # Severity scores (should be 0-based)
    'sofa_score': (0, 24),
    'sapsii_score': (0, 163),
    'charlson_index': (0, 50),
}

# Features to drop due to data quality issues
FEATURES_TO_DROP = {
    'gender_M': 'All values are 0 (data quality issue in eICU)',
    'sofa_score': 'All values are 0 (not computed in eICU raw data)',
    'sapsii_score': 'All values are 0 (not computed in eICU raw data)',
    'charlson_index': 'All values are 0 (not computed in eICU raw data)',
}

# Features with known high missingness to potentially drop
FEATURES_HIGH_MISSINGNESS = {
    'sbp_mean': 0.775,  # 77.5% missing
    'sbp_min': 0.777,
    'mbp_mean': 0.774,
    'mbp_min': 0.780,
    'temperature_mean': 0.910,  # 91% missing
    'hemoglobin_min': 0.928,  # 92.8% missing
    'lactate_max': 0.789,  # 78.9% missing
    'bilirubin_total_max': 0.635,  # 63.5% missing
    'inr_max': 0.681,  # 68.1% missing
    'albumin_min': 0.596,  # 59.6% missing
}


def apply_clinical_clipping(df, feature_name, valid_range):
    """Apply clinical range clipping to a feature"""
    min_val, max_val = valid_range
    invalid_before = ((df[feature_name] < min_val) | (df[feature_name] > max_val)).sum()
    
    if invalid_before > 0:
        df[feature_name] = df[feature_name].clip(min_val, max_val)
        logger.debug(f"  {feature_name}: clipped {invalid_before} out-of-range values")
    
    return df


def preprocess_eicu_cohort(df_raw):
    """
    Preprocess eICU cohort with clinical clipping and feature selection
    
    Args:
        df_raw: Raw eICU dataframe from cache
        
    Returns:
        df_processed: Preprocessed dataframe ready for experiments
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1.2: Preprocessing eICU-CRD Cohort")
    logger.info("=" * 70)
    
    df = df_raw.copy()
    
    # Step 1: Identify features
    feature_cols = [col for col in df.columns if col not in 
                   ['patientunitstayid', 'patient_id', 'hospitalid', 'unittype', 
                    'hospital_expire_flag']]
    
    logger.info(f"\nInitial dataset:")
    logger.info(f"  Admissions: {len(df):,}")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Mortality: {df['hospital_expire_flag'].sum():,} ({df['hospital_expire_flag'].mean()*100:.1f}%)")
    
    # Step 2: Apply clinical clipping to numeric features
    logger.info(f"\nApplying clinical clipping ranges...")
    for feature in feature_cols:
        if feature in CLINICAL_CLIPPING_RANGES and feature in df.columns:
            df = apply_clinical_clipping(df, feature, CLINICAL_CLIPPING_RANGES[feature])
    
    # Step 3: Drop features with known data quality issues
    logger.info(f"\nDropping features with data quality issues:")
    features_to_drop = []
    for feature, reason in FEATURES_TO_DROP.items():
        if feature in df.columns:
            features_to_drop.append(feature)
            logger.info(f"  ✗ {feature}: {reason}")
    
    df = df.drop(columns=features_to_drop)
    
    # Step 4: Handle missing values
    logger.info(f"\nHandling missing values:")
    missing_per_feature = {}
    
    for feature in df.columns:
        if feature not in ['patientunitstayid', 'patient_id', 'hospitalid', 'unittype', 'hospital_expire_flag']:
            missing_count = (df[feature] == 0).sum()  # In eICU, missing vitals/labs are 0
            missing_pct = missing_count / len(df)
            missing_per_feature[feature] = missing_pct
            
            if missing_pct > 0.80:
                logger.warning(f"  {feature}: {missing_pct*100:.1f}% missing → dropping")
                df = df.drop(columns=[feature])
            elif missing_pct > 0.50:
                logger.warning(f"  {feature}: {missing_pct*100:.1f}% missing (keeping)")
            elif missing_pct > 0.10:
                logger.info(f"  {feature}: {missing_pct*100:.1f}% missing (acceptable)")
    
    # Step 5: Summary statistics
    feature_cols_final = [col for col in df.columns if col not in 
                         ['patientunitstayid', 'patient_id', 'hospitalid', 'unittype', 
                          'hospital_expire_flag']]
    
    logger.info(f"\nFinal dataset:")
    logger.info(f"  Admissions: {len(df):,}")
    logger.info(f"  Features: {len(feature_cols_final)}")
    logger.info(f"  Mortality: {df['hospital_expire_flag'].sum():,} ({df['hospital_expire_flag'].mean()*100:.1f}%)")
    logger.info(f"  Dropped: {len(feature_cols) - len(feature_cols_final)} features")
    
    return df


def compare_with_mimic(df_eicu):
    """Compare eICU feature distributions with MIMIC-IV"""
    logger.info("\n" + "=" * 70)
    logger.info("Comparing eICU with MIMIC-IV Distributions")
    logger.info("=" * 70)
    
    try:
        logger.info("Loading MIMIC-IV cohort...")
        df_mimic = loader.load_dataset_with_df(use_cache=True, billing_project=GCP_PROJECT_ID)[0]
        
        # Get common features
        eicu_features = set(df_eicu.columns)
        mimic_features = set(df_mimic.columns)
        common_features = eicu_features & mimic_features
        common_features = [f for f in common_features if f not in 
                          ['stay_id', 'patientunitstayid', 'patient_id', 'hospital_expire_flag',
                           'first_careunit', 'hospitalid', 'unittype']]
        
        logger.info(f"\nCommon features: {len(common_features)}")
        logger.info(f"MIMIC-IV only: {len(mimic_features - eicu_features)}")
        logger.info(f"eICU-CRD only: {len(eicu_features - mimic_features)}")
        
        # Compare distributions
        logger.info(f"\nFeature Distribution Comparison (first 24 hours):")
        logger.info(f"{'Feature':<25} {'eICU Mean':<15} {'MIMIC Mean':<15} {'Diff %':<12}")
        logger.info("-" * 70)
        
        diffs = {}
        for feature in sorted(common_features)[:10]:  # Show first 10
            if feature in df_eicu.columns and feature in df_mimic.columns:
                eicu_mean = df_eicu[feature].mean()
                mimic_mean = df_mimic[feature].mean()
                
                if mimic_mean != 0:
                    pct_diff = abs(eicu_mean - mimic_mean) / abs(mimic_mean) * 100
                else:
                    pct_diff = 0
                
                diffs[feature] = pct_diff
                logger.info(f"{feature:<25} {eicu_mean:<15.2f} {mimic_mean:<15.2f} {pct_diff:<12.1f}%")
        
        avg_diff = np.mean(list(diffs.values()))
        logger.info("-" * 70)
        logger.info(f"Average absolute % difference: {avg_diff:.1f}%")
        
        if avg_diff < 20:
            logger.info("✅ Distributions are similar (< 20% avg difference)")
        elif avg_diff < 50:
            logger.warning("⚠️  Distributions differ moderately (20-50% avg difference)")
        else:
            logger.warning("❌ Distributions differ significantly (> 50% avg difference)")
        
        return df_mimic
        
    except Exception as e:
        logger.warning(f"Could not load MIMIC-IV for comparison: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Preprocess eICU-CRD cohort')
    parser.add_argument('--compare-mimic', action='store_true',
                        help='Compare eICU distributions with MIMIC-IV')
    
    args = parser.parse_args()
    
    # Load eICU cohort
    logger.info("Loading eICU-CRD cohort from cache...")
    df_eicu_raw, X_eicu, y_eicu = eicu_loader.load_dataset_with_df(
        use_cache=True, 
        billing_project=GCP_PROJECT_ID
    )
    
    # Preprocess
    df_eicu_processed = preprocess_eicu_cohort(df_eicu_raw)
    
    # Save processed cohort
    output_path = Path(__file__).parent.parent / "data" / "cache" / "eicu_cohort_processed.csv"
    df_eicu_processed.to_csv(output_path, index=False)
    logger.info(f"\n✅ Processed cohort saved to: {output_path}")
    logger.info(f"   Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Optional: Compare with MIMIC-IV
    if args.compare_mimic:
        compare_with_mimic(df_eicu_processed)
    
    # Print final statistics
    logger.info(f"\n" + "=" * 70)
    logger.info("Phase 1.2 Complete: Data Preprocessed and Ready for Phase 1.3")
    logger.info("=" * 70)
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Phase 1.3 - Hospital Partitioning:")
    logger.info(f"     python scripts/partition_eicu_hospitals.py")
    logger.info(f"  2. Phase 2.1 - Unified Experiment Framework:")
    logger.info(f"     Update experiments to accept --dataset eicu_crd flag")
    logger.info(f"  3. Phase 2.2 - Core Experiments on eICU:")
    logger.info(f"     python experiments/exp1_baseline_eicu.py")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
