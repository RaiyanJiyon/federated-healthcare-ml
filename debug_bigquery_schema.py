#!/usr/bin/env python3
"""
Debug BigQuery MIMIC-IV schema to find correct table paths and item IDs
"""

from google.cloud import bigquery
import os

GCP_PROJECT = os.getenv('GCP_PROJECT_ID', 'mimic-iv-research-496704')
BQ_PROJECT_PHYSIONET = "physionet-data"

client = bigquery.Client(project=GCP_PROJECT)

print("=" * 80)
print("MIMIC-IV v3.1 BigQuery Schema Inspector")
print("=" * 80)

# Check what datasets exist in physionet-data
print("\n1. CHECKING DATASETS IN physionet-data:")
query = f"""
SELECT schema_name 
FROM `{BQ_PROJECT_PHYSIONET}.INFORMATION_SCHEMA.SCHEMATA`
WHERE schema_name LIKE '%mimic%'
ORDER BY schema_name
"""
try:
    results = client.query(query).result()
    for row in results:
        print(f"  - {row.schema_name}")
except Exception as e:
    print(f"  Error: {e}")

# Check tables in mimiciv_3_1_icu
print("\n2. TABLES IN mimiciv_3_1_icu:")
query = f"""
SELECT table_name 
FROM `{BQ_PROJECT_PHYSIONET}.mimiciv_3_1_icu.INFORMATION_SCHEMA.TABLES`
ORDER BY table_name
"""
try:
    results = client.query(query).result()
    for row in results:
        print(f"  - {row.table_name}")
except Exception as e:
    print(f"  Error: {e}")

# Check tables in mimiciv_3_1_hosp
print("\n3. TABLES IN mimiciv_3_1_hosp:")
query = f"""
SELECT table_name 
FROM `{BQ_PROJECT_PHYSIONET}.mimiciv_3_1_hosp.INFORMATION_SCHEMA.TABLES`
ORDER BY table_name
"""
try:
    results = client.query(query).result()
    for row in results:
        print(f"  - {row.table_name}")
except Exception as e:
    print(f"  Error: {e}")

# Check if derived dataset exists
print("\n4. CHECKING mimiciv_3_1_derived:")
query = f"""
SELECT schema_name 
FROM `{BQ_PROJECT_PHYSIONET}.INFORMATION_SCHEMA.SCHEMATA`
WHERE schema_name = 'mimiciv_3_1_derived'
"""
try:
    results = client.query(query).result()
    count = sum(1 for _ in results)
    if count > 0:
        print("  ✓ Dataset exists")
        # List tables
        query2 = f"""
        SELECT table_name 
        FROM `{BQ_PROJECT_PHYSIONET}.mimiciv_3_1_derived.INFORMATION_SCHEMA.TABLES`
        ORDER BY table_name
        """
        results2 = client.query(query2).result()
        for row in results2:
            print(f"    - {row.table_name}")
    else:
        print("  ✗ Dataset does not exist")
except Exception as e:
    print(f"  Error: {e}")

# Sample a few records from chartevents to see structure
print("\n5. SAMPLE FROM chartevents (first 5 itemids):")
query = f"""
SELECT DISTINCT itemid, COUNT(*) as count
FROM `{BQ_PROJECT_PHYSIONET}.mimiciv_3_1_icu.chartevents`
GROUP BY itemid
ORDER BY count DESC
LIMIT 10
"""
try:
    results = client.query(query).result()
    for row in results:
        print(f"  itemid {row.itemid}: {row.count} records")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 80)
