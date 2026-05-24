"""Trial Configuration for Phase 5: Statistical Rigor

Provides 30 independent trial seeds (42-71) and trial management utilities
for expanded statistical validation of federated learning experiments.

This module ensures reproducibility while enabling Monte Carlo estimation
of mean, standard deviation, confidence intervals, and significance tests.

Trial Seeds: 42-71 (30 trials, selected for reproducibility and diversity)
Expected Runtime: 12-16 hours for all trials across key experiments
Output: CSV results, statistical summaries, hypothesis test results
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import json

# ===== TRIAL CONFIGURATION =====
# 30 independent trial seeds spanning reproducible range
TRIAL_SEEDS = list(range(42, 72))  # Seeds 42-71 inclusive
NUM_TRIALS = len(TRIAL_SEEDS)

# Critical experiments to run across all 30 trials
CRITICAL_EXPERIMENTS = {
    'baseline': {
        'name': 'exp1_baseline.py',
        'description': 'Federated vs Centralized baseline (FedAvg, No DP)',
        'key_metrics': ['federated_auroc', 'federated_recall', 'centralized_auroc', 'centralized_recall'],
        'hypothesis': 'Federated Recall ≈ Centralized Recall (within 1.85%)',
        'test': 'paired_t_test',  # Paired t-test: FedAvg recall vs Centralized recall
    },
    'differential_privacy': {
        'name': 'exp7_differential_privacy.py',
        'description': 'DP-SGD epsilon tradeoff (epsilon in {0.5, 1.0, 2.0, 5.0, 10.0})',
        'key_metrics': ['epsilon', 'auroc', 'recall'],
        'hypothesis': 'Recall degrades monotonically with tighter privacy',
        'test': 'anova',  # ANOVA across epsilon values
    },
    'aggregation': {
        'name': 'exp4_aggregation_comparison.py',
        'description': 'Byzantine robustness (FedAvg vs Median vs Krum)',
        'key_metrics': ['method', 'auroc', 'recall'],
        'hypothesis': 'Byzantine methods (Median, Krum) trade recall for robustness',
        'test': 'paired_t_tests',  # Pairwise t-tests: Median vs FedAvg, Krum vs FedAvg
    },
    'scalability': {
        'name': 'exp9_scalability_analysis.py',
        'description': 'Network growth (7, 14, 21, 28 clients)',
        'key_metrics': ['num_clients', 'auroc', 'recall', 'communication_rounds'],
        'hypothesis': 'Recall remains stable (>84%) across client counts',
        'test': 'anova',  # ANOVA across client counts
    },
}

# ===== TRIAL MANAGEMENT CLASS =====
class TrialManager:
    """Manages multi-trial experiment execution, result aggregation, and statistics."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize trial manager.
        
        Args:
            output_dir: Directory for trial results (default: results/trials/)
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "results" / "trials"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.trial_results = {}  # {experiment: {seed: results}}
        self.statistical_summaries = {}  # {experiment: summary_stats}
        self.hypothesis_tests = {}  # {experiment: test_results}
    
    def get_trial_seeds(self) -> List[int]:
        """Return list of trial seeds."""
        return TRIAL_SEEDS.copy()
    
    def get_experiment_config(self, experiment_key: str) -> Dict:
        """Get configuration for a specific critical experiment.
        
        Args:
            experiment_key: Key from CRITICAL_EXPERIMENTS
            
        Returns:
            Configuration dictionary with experiment details
        """
        return CRITICAL_EXPERIMENTS.get(experiment_key, {})
    
    def save_trial_result(self, experiment: str, seed: int, results: Dict) -> None:
        """Save individual trial results.
        
        Args:
            experiment: Experiment name (e.g., 'baseline', 'differential_privacy')
            seed: Random seed for this trial
            results: Dictionary of results (metrics)
        """
        if experiment not in self.trial_results:
            self.trial_results[experiment] = {}
        
        self.trial_results[experiment][seed] = results
        
        # Save to JSON file immediately
        trial_file = self.output_dir / f"trial_{experiment}_seed{seed}.json"
        with open(trial_file, 'w') as f:
            json.dump({
                'experiment': experiment,
                'seed': seed,
                'timestamp': datetime.now().isoformat(),
                'results': results
            }, f, indent=2)
    
    def load_trial_result(self, experiment: str, seed: int) -> Dict:
        """Load previously saved trial results.
        
        Args:
            experiment: Experiment name
            seed: Random seed
            
        Returns:
            Results dictionary or empty dict if not found
        """
        trial_file = self.output_dir / f"trial_{experiment}_seed{seed}.json"
        if trial_file.exists():
            with open(trial_file, 'r') as f:
                data = json.load(f)
                return data.get('results', {})
        return {}
    
    def compute_statistics(self, experiment: str, metric: str) -> Dict:
        """Compute statistics for a metric across all trials.
        
        Args:
            experiment: Experiment name
            metric: Metric name (e.g., 'recall', 'auroc')
            
        Returns:
            Statistics dict with mean, std, sem, 95% CI, min, max
        """
        if experiment not in self.trial_results:
            return {}
        
        values = []
        for seed, results in self.trial_results[experiment].items():
            if metric in results:
                val = results[metric]
                # Handle nested structures (e.g., {'value': 0.85})
                if isinstance(val, dict) and 'value' in val:
                    val = val['value']
                if isinstance(val, (int, float)):
                    values.append(float(val))
        
        if not values:
            return {}
        
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0
        sem = std / np.sqrt(len(values))
        ci_95 = 1.96 * sem  # 95% confidence interval half-width
        
        return {
            'metric': metric,
            'num_trials': len(values),
            'mean': float(mean),
            'std': float(std),
            'sem': float(sem),
            'ci_95_lower': float(mean - ci_95),
            'ci_95_upper': float(mean + ci_95),
            'ci_95_halfwidth': float(ci_95),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
        }
    
    def generate_statistical_summary(self, experiment: str) -> pd.DataFrame:
        """Generate summary statistics for all metrics in experiment.
        
        Args:
            experiment: Experiment name
            
        Returns:
            DataFrame with statistics for each metric
        """
        if experiment not in self.trial_results:
            return pd.DataFrame()
        
        config = self.get_experiment_config(experiment)
        key_metrics = config.get('key_metrics', [])
        
        summaries = []
        for metric in key_metrics:
            stats = self.compute_statistics(experiment, metric)
            if stats:
                summaries.append(stats)
        
        return pd.DataFrame(summaries)
    
    def save_summary(self, experiment: str) -> Path:
        """Save statistical summary to CSV.
        
        Args:
            experiment: Experiment name
            
        Returns:
            Path to saved CSV file
        """
        df = self.generate_statistical_summary(experiment)
        if df.empty:
            return None
        
        summary_file = self.output_dir / f"summary_{experiment}_statistics.csv"
        df.to_csv(summary_file, index=False)
        return summary_file
    
    def create_trial_report(self) -> str:
        """Create comprehensive trial report as text.
        
        Returns:
            Formatted report string
        """
        report = [
            "=" * 80,
            "PHASE 5: STATISTICAL RIGOR - TRIAL REPORT",
            "=" * 80,
            f"Report Generated: {datetime.now().isoformat()}",
            f"Total Trials: {NUM_TRIALS}",
            f"Trial Seeds: {min(TRIAL_SEEDS)}-{max(TRIAL_SEEDS)}",
            "",
        ]
        
        for exp_key in CRITICAL_EXPERIMENTS.keys():
            report.append(f"\n{'─' * 80}")
            report.append(f"EXPERIMENT: {exp_key.upper()}")
            report.append(f"{'─' * 80}")
            
            config = self.get_experiment_config(exp_key)
            report.append(f"Description: {config.get('description', 'N/A')}")
            report.append(f"Hypothesis: {config.get('hypothesis', 'N/A')}")
            report.append(f"Test: {config.get('test', 'N/A')}")
            report.append("")
            
            df = self.generate_statistical_summary(exp_key)
            if not df.empty:
                report.append(df.to_string(index=False))
            else:
                report.append("No results available yet")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)
    
    def save_report(self) -> Path:
        """Save trial report to file.
        
        Returns:
            Path to saved report
        """
        report = self.create_trial_report()
        report_file = self.output_dir / f"TRIAL_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        return report_file


# ===== CONVENIENCE FUNCTIONS =====
def get_trial_seeds() -> List[int]:
    """Get list of all trial seeds."""
    return TRIAL_SEEDS.copy()

def create_trial_manager(output_dir: Path = None) -> TrialManager:
    """Create and return a trial manager instance."""
    return TrialManager(output_dir)

def print_trial_plan() -> None:
    """Print overview of trial plan and hypothesis tests."""
    print("\n" + "=" * 80)
    print("PHASE 5: STATISTICAL RIGOR - TRIAL PLAN")
    print("=" * 80)
    print(f"Number of Trials: {NUM_TRIALS}")
    print(f"Trial Seeds: {min(TRIAL_SEEDS)}-{max(TRIAL_SEEDS)}")
    print(f"Estimated Runtime: 12-16 hours")
    print("\nCritical Experiments:")
    
    for i, (exp_key, exp_config) in enumerate(CRITICAL_EXPERIMENTS.items(), 1):
        print(f"\n{i}. {exp_key.upper()}")
        print(f"   Script: {exp_config['name']}")
        print(f"   Metrics: {', '.join(exp_config['key_metrics'])}")
        print(f"   Hypothesis: {exp_config['hypothesis']}")
        print(f"   Test: {exp_config['test']}")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # Print trial plan
    print_trial_plan()
    
    # Create trial manager and show capabilities
    tm = TrialManager()
    print(f"Trial Manager initialized")
    print(f"Output directory: {tm.output_dir}")
    print(f"Available trial seeds: {len(tm.get_trial_seeds())}")
    print(f"Available experiments: {list(CRITICAL_EXPERIMENTS.keys())}")
