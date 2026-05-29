#!/usr/bin/env python3
"""
Test script to verify BigQuery connection and eICU cohort extraction
Run this before starting full Phase 1.1 experiments

Usage:
    python scripts/test_eicu_bigquery.py [--test-query] [--download-cohort] [--dry-run]
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from google.cloud import bigquery
from src.config.config import GCP_PROJECT_ID, BQ_DATASET_EICU, BQ_DATASET_EICU_DERIVED

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_bigquery_connection():
    """Test connection to BigQuery"""
    try:
        logger.info("Testing BigQuery connection...")
        client = bigquery.Client(project=GCP_PROJECT_ID)
        
        # List datasets
        datasets = list(client.list_datasets())
        logger.info(f"✅ BigQuery connection successful!")
        logger.info(f"   Project: {GCP_PROJECT_ID}")
        logger.info(f"   Found {len(datasets)} datasets")
        
        return client
    except Exception as e:
        logger.error(f"❌ BigQuery connection failed: {str(e)}")
        return None


def test_eicu_tables(client):
    """Test availability of eICU tables"""
    try:
        logger.info(f"\nChecking eICU-CRD tables in {BQ_DATASET_EICU}...")
        
        # List tables in eicu_crd
        tables = list(client.list_tables(BQ_DATASET_EICU))
        logger.info(f"✅ eICU-CRD dataset found! Tables available:")
        for table in tables[:10]:  # Show first 10
            logger.info(f"   - {table.table_id}")
        if len(tables) > 10:
            logger.info(f"   ... and {len(tables) - 10} more")
        
        # Check key tables
        required_tables = ['patient', 'vital', 'lab']
        for table_name in required_tables:
            table_ref = f"{BQ_DATASET_EICU}.{table_name}"
            try:
                table = client.get_table(table_ref)
                logger.info(f"✅ Table '{table_name}': {table.num_rows:,} rows")
            except:
                logger.warning(f"⚠️  Table '{table_name}' not found")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error checking tables: {str(e)}")
        return False


def test_cohort_query_dry_run(client, dry_run=True):
    """Test eICU cohort query with dry run"""
    try:
        logger.info(f"\nTesting eICU cohort query (dry_run={dry_run})...")
        
        # Simple test query to get cohort size
        test_query = f"""
        SELECT 
          COUNT(DISTINCT patientunitstayid) as total_icu_stays,
          COUNT(DISTINCT patient_id) as total_patients,
          COUNT(DISTINCT hospitalid) as total_hospitals,
          MIN(age) as min_age,
          MAX(age) as max_age
        FROM `{BQ_DATASET_EICU}.patient`
        WHERE age >= 18
          AND hospitaldischargestatus IN ('Expired', 'Alive')
        """
        
        job_config = bigquery.QueryJobConfig(dry_run=dry_run)
        query_job = client.query(test_query, job_config=job_config)
        
        if dry_run:
            logger.info(f"✅ Query dry run successful")
            logger.info(f"   Bytes to be scanned: {query_job.total_bytes_processed:,}")
            logger.info(f"   Estimated cost: ${query_job.total_bytes_processed / (1024**10) * 7.5:.2f} (at $7.50/TB)")
        else:
            df = query_job.to_dataframe()
            logger.info(f"✅ Query executed successfully!")
            logger.info(f"   Results:")
            for col in df.columns:
                logger.info(f"     {col}: {df[col].values[0]}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Query test failed: {str(e)}")
        return False


def quick_hospital_count(client):
    """Get quick count of hospitals and patients"""
    try:
        logger.info(f"\nQuick statistics...")
        
        query = f"""
        SELECT 
          hospitalid,
          COUNT(DISTINCT patientunitstayid) as patient_count,
          COUNT(CASE WHEN hospitaldischargestatus = 'Expired' THEN 1 END) as mortality_count
        FROM `{BQ_DATASET_EICU}.patient`
        WHERE age >= 18
          AND DATETIME_DIFF(unitdischargetime, unitadmittime, HOUR) >= 4
        GROUP BY hospitalid
        ORDER BY patient_count DESC
        LIMIT 10
        """
        
        job_config = bigquery.QueryJobConfig()
        query_job = client.query(query, job_config=job_config)
        df = query_job.to_dataframe()
        
        logger.info(f"✅ Top 10 hospitals by patient count:")
        logger.info(f"{'Hospital ID':<12} {'Patients':<10} {'Deaths':<10} {'Mortality %':<12}")
        logger.info("=" * 50)
        
        for _, row in df.iterrows():
            hosp_id = row['hospitalid']
            count = row['patient_count']
            deaths = row['mortality_count']
            mortality_pct = (deaths / count * 100) if count > 0 else 0
            logger.info(f"{hosp_id:<12} {count:<10} {deaths:<10} {mortality_pct:<12.1f}%")
        
        return True
    except Exception as e:
        logger.error(f"❌ Statistics query failed: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test eICU BigQuery connection')
    parser.add_argument('--test-query', action='store_true', help='Run test query')
    parser.add_argument('--hospital-stats', action='store_true', help='Show hospital statistics')
    parser.add_argument('--no-dry-run', action='store_true', help='Execute query without dry run')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("eICU-CRD BigQuery Connection Test")
    logger.info("=" * 60)
    
    # Step 1: Test connection
    client = test_bigquery_connection()
    if not client:
        logger.error("\n❌ Cannot proceed without BigQuery connection")
        sys.exit(1)
    
    # Step 2: Test tables
    if not test_eicu_tables(client):
        logger.warning("⚠️  Some tables may be missing")
    
    # Step 3: Test query
    if args.test_query:
        dry_run = not args.no_dry_run
        test_cohort_query_dry_run(client, dry_run=dry_run)
    
    # Step 4: Hospital statistics
    if args.hospital_stats:
        quick_hospital_count(client)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Connection tests complete!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Run: python scripts/test_eicu_bigquery.py --test-query --hospital-stats")
    logger.info("2. Review hospital statistics to confirm data quality")
    logger.info("3. When ready, run Phase 1.1: python src/data/eicu_loader.py")


if __name__ == '__main__':
    main()
