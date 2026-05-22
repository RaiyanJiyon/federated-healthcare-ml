#!/usr/bin/env python
"""
Reproducibility Metadata Logging

Records all information needed to reproduce experiments:
- Package versions (dependencies)
- Random seed
- Dataset version and MIMIC SQL hash
- Feature set version
- Experiment run metadata
- Timestamp and system info
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import hashlib
import platform
import subprocess

sys.path.insert(0, str(Path(__file__).parent))

from src.config.config import RANDOM_SEED, DP_EPSILON, DP_DELTA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_package_versions():
    """Get versions of key dependencies."""
    packages = [
        'sklearn', 'numpy', 'pandas', 'scipy',
        'shap', 'matplotlib', 'seaborn'
    ]
    
    versions = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            versions[pkg] = 'not installed'
    
    return versions


def get_git_info():
    """Get current git commit hash if in a git repo."""
    try:
        repo_root = Path(__file__).parent
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return {
                'commit_hash': result.stdout.strip(),
                'branch': subprocess.run(
                    ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    timeout=5
                ).stdout.strip()
            }
    except Exception:
        pass
    
    return {'commit_hash': 'unknown', 'branch': 'unknown'}


def compute_sql_hash():
    """Compute hash of SQL query for cohort extraction."""
    # Read the BigQuery SQL from config or data loader
    try:
        from src.data.loader import COHORT_QUERY
        sql_hash = hashlib.sha256(COHORT_QUERY.encode()).hexdigest()[:16]
        return sql_hash
    except Exception:
        return 'unknown'


def get_feature_set_info():
    """Get feature set metadata."""
    return {
        'n_features': 31,
        'feature_types': {
            'demographics': 4,
            'vitals': 14,
            'labs': 12,
            'clinical_scores': 3
        },
        'version': '1.0'
    }


def create_reproducibility_metadata(experiment_name: str) -> dict:
    """
    Create comprehensive reproducibility metadata for an experiment.
    
    Args:
        experiment_name: Name of experiment (e.g., 'exp1_baseline', 'exp7_dp')
    
    Returns:
        dict: Complete metadata for reproducibility
    """
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'experiment': experiment_name,
        'random_seed': RANDOM_SEED,
        'dp_config': {
            'epsilon': DP_EPSILON,
            'delta': DP_DELTA
        },
        'system': {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'hostname': platform.node()
        },
        'git': get_git_info(),
        'packages': get_package_versions(),
        'dataset': {
            'name': 'MIMIC-IV',
            'version': '3.1',
            'cohort_sql_hash': compute_sql_hash(),
            'n_patients': 65273,
            'train_test_split': '70-15-15'
        },
        'features': get_feature_set_info(),
        'reproducibility_instructions': (
            "To reproduce this experiment:\n"
            "1. Install packages: pip install -r requirements.txt\n"
            f"2. Set RANDOM_SEED={RANDOM_SEED} in src/config/config.py\n"
            "3. Download MIMIC-IV from PhysioNet (requires access)\n"
            f"4. Run: python experiments/{experiment_name}.py"
        )
    }
    
    return metadata


def save_reproducibility_log(experiment_name: str, metadata: dict = None) -> Path:
    """
    Save reproducibility metadata to JSON file.
    
    Args:
        experiment_name: Name of experiment
        metadata: Metadata dict (created if None)
    
    Returns:
        Path to saved metadata file
    """
    
    if metadata is None:
        metadata = create_reproducibility_metadata(experiment_name)
    
    output_dir = Path('results/reproducibility')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_file = output_dir / f'reproducibility_{experiment_name}_{timestamp}.json'
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Reproducibility metadata saved to {metadata_file}")
    
    return metadata_file


def create_reproducibility_summary() -> dict:
    """Create summary of all reproducibility information."""
    
    summary = {
        'project': 'Federated Healthcare ML (MIMIC-IV)',
        'phases': {
            'phase_1': {
                'status': 'complete',
                'experiments': ['exp1_baseline', 'exp2_noniid', 'exp3_clients'],
                'key_result': 'FedAvg AUROC 0.8850'
            },
            'phase_2': {
                'status': 'complete',
                'experiments': ['exp4_aggregation', 'exp5_shap', 'exp6_statistical'],
                'key_results': [
                    'FedProx underperforms (-2.6% AUROC)',
                    'SHAP drift analysis (mean CV 0.684)',
                    '95% CI validation across 5 seeds'
                ]
            },
            'phase_3': {
                'status': 'complete',
                'experiments': ['exp7_privacy', 'exp8_robustness', 'exp9_scalability'],
                'key_results': [
                    'DP cost too high (-49% AUROC)',
                    'Byzantine resilient to <2/7 attackers',
                    'Perfect scaling to 28+ clients'
                ]
            },
            'phase_4': {
                'status': 'in progress',
                'tasks': [
                    'Automated figure generation',
                    'Reproducibility metadata logging',
                    'Publication tables and summaries'
                ]
            }
        },
        'random_seed': RANDOM_SEED,
        'dataset_version': 'MIMIC-IV 3.1',
        'reproducibility_files': [
            'results/reproducibility/reproducibility_*.json',
            'src/config/config.py (fixed seed)',
            'requirements.txt (fixed versions)'
        ],
        'paper_artifacts': [
            'results/plots/paper_*.csv',
            'results/plots/paper_*.png',
            'results/plots/paper_*.tex'
        ]
    }
    
    return summary


def save_reproducibility_summary() -> Path:
    """Save reproducibility summary to JSON."""
    summary = create_reproducibility_summary()
    
    output_dir = Path('results/reproducibility')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_dir / 'reproducibility_summary.json'
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✓ Reproducibility summary saved to {summary_file}")
    
    return summary_file


def main():
    """Generate all reproducibility documentation."""
    logger.info("="*70)
    logger.info("REPRODUCIBILITY METADATA GENERATION")
    logger.info("="*70)
    
    # Create metadata for each phase
    experiments = [
        'exp1_baseline',
        'exp4_aggregation_comparison',
        'exp5_shap_drift_analysis',
        'exp6_statistical_validation',
        'exp7_differential_privacy',
        'exp8_adversarial_robustness',
        'exp9_scalability_analysis'
    ]
    
    logger.info("\nGenerating reproducibility metadata for each experiment...")
    for exp in experiments:
        metadata = create_reproducibility_metadata(exp)
        save_reproducibility_log(exp, metadata)
    
    # Create summary
    logger.info("\nGenerating reproducibility summary...")
    save_reproducibility_summary()
    
    logger.info("\n" + "="*70)
    logger.info("REPRODUCIBILITY DOCUMENTATION COMPLETE")
    logger.info("="*70)
    logger.info("\nKey reproducibility information:")
    logger.info(f"  - Random Seed: {RANDOM_SEED}")
    logger.info(f"  - Dataset: MIMIC-IV 3.1")
    logger.info(f"  - DP Config: ε={DP_EPSILON}, δ={DP_DELTA}")
    logger.info(f"  - Metadata Location: results/reproducibility/")
    logger.info("\n✓ All reproducibility artifacts saved")


if __name__ == '__main__':
    main()
    logger.info("\n✅ REPRODUCIBILITY LOGGING COMPLETE")
