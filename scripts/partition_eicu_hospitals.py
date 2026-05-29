#!/usr/bin/env python3
"""
Phase 1.3: Create hospital-based federated partitioning for eICU-CRD

This script:
1. Loads preprocessed eICU cohort
2. Identifies top 7 hospitals by patient count
3. Creates federated client assignments
4. Validates Non-IID distribution across clients
5. Saves partitioned data for Phase 2.2 experiments

Run after: Phase 1.2 (preprocess_eicu_data.py)
Output: data/cache/eicu_partitioned/hospital_<id>_clients.csv

Usage:
    python scripts/partition_eicu_hospitals.py [--num-hospitals 7] [--visualize]
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config import GCP_PROJECT_ID, DATASET_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_iid_metrics(df, partition_key, num_classes=2):
    """
    Compute Non-IID metrics across clients
    
    Args:
        df: Dataframe with partitions
        partition_key: Column name for partition (e.g., 'hospitalid')
        num_classes: Number of target classes (2 for binary mortality)
        
    Returns:
        metrics: Dictionary with Non-IID statistics
    """
    metrics = {
        'total_clients': df[partition_key].nunique(),
        'total_samples': len(df),
        'target_distribution': {},
        'class_distribution_by_client': defaultdict(dict),
        'kld': [],  # Kullback-Leibler divergence for each client
    }
    
    # Global class distribution
    if 'hospital_expire_flag' in df.columns:
        global_dist = df['hospital_expire_flag'].value_counts(normalize=True).sort_index().values
        metrics['target_distribution']['global'] = {
            f'class_{i}': float(global_dist[i]) for i in range(len(global_dist))
        }
    
    # Per-client distribution
    for client_id in sorted(df[partition_key].unique()):
        client_data = df[df[partition_key] == client_id]
        metrics['class_distribution_by_client'][client_id] = {
            'samples': len(client_data),
            'mortality_rate': client_data['hospital_expire_flag'].mean() if 'hospital_expire_flag' in df.columns else 0,
        }
        
        if 'hospital_expire_flag' in df.columns:
            # Compute KLD from global distribution
            local_dist = client_data['hospital_expire_flag'].value_counts(normalize=True).sort_index().values
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            local_dist = np.clip(local_dist, eps, 1)
            global_dist_clipped = np.clip(global_dist, eps, 1)
            kld = np.sum(local_dist * np.log(local_dist / global_dist_clipped))
            metrics['kld'].append(kld)
    
    if metrics['kld']:
        metrics['avg_kld'] = np.mean(metrics['kld'])
        metrics['max_kld'] = np.max(metrics['kld'])
    
    return metrics


def create_hospital_partitions(df_processed, num_hospitals=7):
    """
    Create federated partitions based on top hospitals
    
    Args:
        df_processed: Preprocessed eICU dataframe
        num_hospitals: Number of hospitals to select as clients
        
    Returns:
        partitions: Dictionary mapping hospital_id -> client_data
        metadata: Partitioning metadata
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1.3: Hospital-Based Federated Partitioning")
    logger.info("=" * 70)
    
    # Get top N hospitals by patient count
    hospital_counts = df_processed['hospitalid'].value_counts()
    top_hospitals = hospital_counts.head(num_hospitals).index.tolist()
    
    logger.info(f"\nSelecting top {num_hospitals} hospitals by patient count:")
    logger.info(f"{'Hospital ID':<12} {'Patients':<12} {'% of Total':<12} {'Mortality %':<12}")
    logger.info("-" * 50)
    
    partitions = {}
    total_samples = len(df_processed)
    
    for idx, hospital_id in enumerate(top_hospitals, 1):
        hospital_data = df_processed[df_processed['hospitalid'] == hospital_id].copy()
        partitions[hospital_id] = hospital_data
        
        pct_total = len(hospital_data) / total_samples * 100
        mortality_rate = hospital_data['hospital_expire_flag'].mean() * 100
        
        logger.info(f"{hospital_id:<12} {len(hospital_data):<12,} {pct_total:<12.1f} {mortality_rate:<12.1f}")
    
    total_partitioned = sum(len(p) for p in partitions.values())
    logger.info("-" * 50)
    logger.info(f"{'TOTAL':<12} {total_partitioned:<12,} {total_partitioned/total_samples*100:<12.1f}")
    
    # Create partitioned dataframe
    df_partitioned = pd.concat(partitions.values(), ignore_index=False)
    
    # Compute Non-IID metrics
    logger.info(f"\nNon-IID Analysis (Federated Data Heterogeneity):")
    metrics = compute_iid_metrics(df_partitioned, 'hospitalid')
    
    logger.info(f"  Total clients: {metrics['total_clients']}")
    logger.info(f"  Total samples: {metrics['total_samples']:,}")
    logger.info(f"  Average KL divergence: {metrics['avg_kld']:.4f}" if 'avg_kld' in metrics else "  (N/A)")
    
    logger.info(f"\n  Client Statistics:")
    logger.info(f"  {'Client':<12} {'Samples':<12} {'% Total':<12} {'Mortality %':<12}")
    logger.info("-" * 50)
    
    for client_id in sorted(metrics['class_distribution_by_client'].keys()):
        stats = metrics['class_distribution_by_client'][client_id]
        pct = stats['samples'] / metrics['total_samples'] * 100
        mortality = stats['mortality_rate'] * 100
        logger.info(f"{client_id:<12} {stats['samples']:<12,} {pct:<12.1f} {mortality:<12.1f}")
    
    # Data heterogeneity summary
    if 'avg_kld' in metrics:
        heterogeneity = "high (KLD > 0.1)" if metrics['avg_kld'] > 0.1 else "moderate (0.01 < KLD < 0.1)" if metrics['avg_kld'] > 0.01 else "low (KLD < 0.01)"
        logger.info(f"\nData Heterogeneity: {heterogeneity}")
        logger.info("  → Realistic federated scenario with local data drift across hospitals")
    
    metadata = {
        'num_hospitals': num_hospitals,
        'top_hospitals': top_hospitals,
        'total_samples': total_partitioned,
        'total_admissions': total_samples,
        'coverage_pct': total_partitioned / total_samples * 100,
        'metrics': metrics,
    }
    
    return df_partitioned, partitions, metadata


def save_partitions(partitions, output_dir):
    """Save each hospital's data to separate CSV"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving partitioned data to: {output_dir}")
    
    for hospital_id, data in partitions.items():
        output_path = output_dir / f"hospital_{hospital_id}_clients.csv"
        data.to_csv(output_path, index=False)
        logger.info(f"  ✓ Hospital {hospital_id}: {len(data):,} admissions → {output_path.name}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Create hospital-based federated partitioning')
    parser.add_argument('--num-hospitals', type=int, default=7,
                        help='Number of hospitals to use as federated clients (default: 7)')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization of data distribution (requires matplotlib)')
    
    args = parser.parse_args()
    
    # Load preprocessed cohort
    logger.info("Loading preprocessed eICU cohort...")
    cache_path = Path(__file__).parent.parent / "data" / "cache" / "eicu_cohort_processed.csv"
    
    if not cache_path.exists():
        logger.error(f"❌ Preprocessed cohort not found: {cache_path}")
        logger.error("   Run Phase 1.2 first: python scripts/preprocess_eicu_data.py")
        return 1
    
    df_processed = pd.read_csv(cache_path)
    logger.info(f"Loaded: {len(df_processed):,} admissions, {len(df_processed.columns)} columns")
    
    # Create partitions
    df_partitioned, partitions, metadata = create_hospital_partitions(
        df_processed, 
        num_hospitals=args.num_hospitals
    )
    
    # Save partitions
    output_dir = Path(__file__).parent.parent / "data" / "cache" / "eicu_partitioned"
    save_partitions(partitions, output_dir)
    
    # Save metadata
    metadata_path = output_dir / "partitioning_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("eICU-CRD Hospital-Based Federated Partitioning\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Number of hospitals (clients): {metadata['num_hospitals']}\n")
        f.write(f"Hospital IDs: {metadata['top_hospitals']}\n")
        f.write(f"Total samples in partition: {metadata['total_samples']:,}\n")
        f.write(f"Coverage: {metadata['coverage_pct']:.1f}%\n")
        
        if 'metrics' in metadata and 'avg_kld' in metadata['metrics']:
            f.write(f"\nNon-IID Metrics:\n")
            f.write(f"  Average KL divergence: {metadata['metrics']['avg_kld']:.4f}\n")
            f.write(f"  Max KL divergence: {metadata['metrics']['max_kld']:.4f}\n")
    
    logger.info(f"\n✅ Partitioning complete!")
    logger.info(f"   Metadata saved to: {metadata_path}")
    
    # Optional visualization
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Samples per hospital
            hospitals = list(metadata['metrics']['class_distribution_by_client'].keys())
            samples = [metadata['metrics']['class_distribution_by_client'][h]['samples'] for h in hospitals]
            
            axes[0].bar(hospitals, samples)
            axes[0].set_xlabel('Hospital ID')
            axes[0].set_ylabel('Number of Admissions')
            axes[0].set_title('Federated Clients: Sample Distribution')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Mortality rate per hospital
            mortality = [metadata['metrics']['class_distribution_by_client'][h]['mortality_rate'] * 100 
                        for h in hospitals]
            
            axes[1].bar(hospitals, mortality, color='coral')
            axes[1].set_xlabel('Hospital ID')
            axes[1].set_ylabel('Mortality Rate (%)')
            axes[1].set_title('Non-IID Distribution: Mortality Rate by Client')
            axes[1].axhline(y=df_partitioned['hospital_expire_flag'].mean() * 100,
                          color='r', linestyle='--', label='Global avg')
            axes[1].legend()
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            viz_path = output_dir / "federated_distribution.png"
            plt.savefig(viz_path, dpi=150)
            logger.info(f"   Visualization saved to: {viz_path}")
            
        except ImportError:
            logger.warning("   Matplotlib not available for visualization")
    
    # Print next steps
    logger.info(f"\n" + "=" * 70)
    logger.info("Next: Phase 2 - Federated Learning Experiments")
    logger.info("=" * 70)
    logger.info(f"\nPhase 2.1 - Create unified experiment framework")
    logger.info(f"  Update experiments to support: --dataset mimic_iv | eicu_crd")
    logger.info(f"\nPhase 2.2 - Run core experiments on eICU")
    logger.info(f"  1. Centralized baseline: python experiments/exp1_baseline.py --dataset eicu_crd")
    logger.info(f"  2. Federated FedAvg: python experiments/exp2_federated.py --dataset eicu_crd")
    logger.info(f"  3. Calibration: python experiments/exp8_calibration.py --dataset eicu_crd")
    logger.info(f"\nPhase 2.3 - Compare results")
    logger.info(f"  Generate multi-dataset comparison table and figures")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
