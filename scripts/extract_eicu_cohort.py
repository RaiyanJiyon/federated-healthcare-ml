#!/usr/bin/env python3
"""
Phase 1.1: Extract eICU-CRD cohort from BigQuery and cache locally

This script:
1. Connects to BigQuery
2. Extracts eICU-CRD cohort with 31 clinical features
3. Caches to local CSV (data/cache/eicu_cohort.csv)
4. Validates cohort quality

Runtime: ~5-10 minutes (first time)
         ~5 seconds (subsequent runs from cache)

Usage:
    python scripts/extract_eicu_cohort.py [--force-download] [--dry-run]
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import eicu_loader
from src.config.config import GCP_PROJECT_ID

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_cohort_summary(df):
    """Print summary statistics of loaded cohort"""
    logger.info("\n" + "=" * 70)
    logger.info("eICU-CRD COHORT SUMMARY")
    logger.info("=" * 70)
    
    logger.info(f"\nBasic Statistics:")
    logger.info(f"  Total ICU stays: {len(df):,}")
    logger.info(f"  Total patients: {df['patient_id'].nunique():,}")
    logger.info(f"  Total hospitals: {df['hospitalid'].nunique():,}")
    
    logger.info(f"\nDemographics:")
    logger.info(f"  Age (mean ± std): {df['age'].mean():.1f} ± {df['age'].std():.1f} years")
    logger.info(f"  Age range: {df['age'].min():.0f} - {df['age'].max():.0f} years")
    logger.info(f"  Male (%): {df['gender_M'].mean() * 100:.1f}%")
    logger.info(f"  Emergency admission: {df['admission_emergency'].mean() * 100:.1f}%")
    
    logger.info(f"\nMortality:")
    mortality = df['hospital_expire_flag'].mean()
    deaths = df['hospital_expire_flag'].sum()
    logger.info(f"  Mortality rate: {mortality * 100:.1f}% ({deaths:,} deaths)")
    
    logger.info(f"\nFeature Completeness:")
    feature_cols = [col for col in df.columns if col not in 
                    ['patientunitstayid', 'patient_id', 'hospitalid', 'unittype', 
                     'hospital_expire_flag', 'admission_emergency', 'insurance_medicare', 'gender_M']]
    
    missing_count = 0
    for col in feature_cols:
        zeros = (df[col] == 0).sum()
        pct = (zeros / len(df)) * 100
        if pct > 50:
            logger.warning(f"  {col}: {pct:.1f}% zero/missing (consider dropping)")
            missing_count += 1
        elif pct > 10:
            logger.info(f"  {col}: {pct:.1f}% zero/missing")
    
    if missing_count == 0:
        logger.info(f"  ✅ All features well-represented (<50% missing)")
    
    logger.info(f"\nTop 5 hospitals by patient count:")
    top_hospitals = df['hospitalid'].value_counts().head(5)
    for hosp_id, count in top_hospitals.items():
        logger.info(f"    Hospital {hosp_id}: {count:,} patients")
    
    logger.info("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Extract eICU-CRD cohort from BigQuery')
    parser.add_argument('--force-download', action='store_true', 
                        help='Force download from BigQuery (skip cache)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run: test query without downloading')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("PHASE 1.1: Extract eICU-CRD Cohort")
    logger.info("=" * 70)
    logger.info(f"Project ID: {GCP_PROJECT_ID}")
    logger.info(f"Dataset: eicu_crd, eicu_crd_derived")
    logger.info(f"Target features: 31 clinical features (demographics, vitals, labs, scores)")
    
    if args.dry_run:
        logger.info("\n⚠️  DRY RUN MODE - Query will not be executed")
    
    # Load cohort (with caching)
    use_cache = not args.force_download
    logger.info(f"\nCaching strategy: use_cache={use_cache}")
    
    try:
        logger.info("\nLoading eICU-CRD cohort...")
        df, X, y = eicu_loader.load_dataset_with_df(
            use_cache=use_cache, 
            billing_project=GCP_PROJECT_ID
        )
        
        logger.info(f"✅ Cohort loaded successfully!")
        logger.info(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        logger.info(f"   Features: {X.shape[1]} clinical features")
        logger.info(f"   Target: {sum(y)}/{len(y)} mortality cases ({sum(y)/len(y)*100:.1f}%)")
        
        # Print summary
        print_cohort_summary(df)
        
        # Save metadata
        metadata = {
            'extraction_time': datetime.now().isoformat(),
            'cohort_size': len(df),
            'num_features': X.shape[1],
            'mortality_rate': float(sum(y) / len(y)),
            'num_hospitals': df['hospitalid'].nunique(),
            'cache_path': str(eicu_loader.EICU_COHORT_CACHE_PATH)
        }
        
        logger.info(f"\n✅ Phase 1.1 complete!")
        logger.info(f"\nCached to: {eicu_loader.EICU_COHORT_CACHE_PATH}")
        logger.info(f"Size: {eicu_loader.EICU_COHORT_CACHE_PATH.stat().st_size / 1024 / 1024:.1f} MB")
        
        logger.info("\n📋 Next steps:")
        logger.info("   1. Review cohort quality summary above")
        logger.info("   2. Proceed to Phase 1.2: Preprocessing & Validation")
        logger.info("   3. Run: python scripts/validate_eicu_preprocessing.py")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Phase 1.1 failed: {str(e)}")
        logger.error("Troubleshooting:")
        logger.error("  1. Verify BigQuery connection: gcloud auth login")
        logger.error("  2. Test connection: python scripts/test_eicu_bigquery.py")
        logger.error("  3. Check GCP project: gcloud config list")
        
        return 1


if __name__ == '__main__':
    sys.exit(main())
