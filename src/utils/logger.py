"""Logging utilities for experiments"""

import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


class ExperimentLogger:
    """Structured logging for FL experiments"""
    
    def __init__(self, log_dir: str = "results/logs", name: str = "experiment"):
        """Initialize logger with optional file and console handlers"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler (INFO level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (DEBUG level)
        log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{name}_{log_timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        self.log_file = log_file
        self.logger.info(f"Logger initialized: {log_file}")
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> Path:
        """Save experiment results to JSON file"""
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"
        
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {filepath}")
        return filepath


class MetricsCollector:
    """Collect and aggregate metrics across experiments"""
    
    def __init__(self):
        self.experiments: Dict[str, Dict[str, Any]] = {}
    
    def add_experiment(self, exp_name: str, metrics: Dict[str, Any], 
                       duration: float = None, status: str = "completed"):
        """Add experiment results to collector"""
        self.experiments[exp_name] = {
            "metrics": metrics,
            "duration": duration,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all collected experiments"""
        return {
            "total_experiments": len(self.experiments),
            "experiments": self.experiments,
            "timestamp": datetime.now().isoformat()
        }
    
    def save_summary(self, filepath: str = "results/summary.json"):
        """Save summary to JSON"""
        summary = self.get_summary()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return filepath
    
    def print_summary(self):
        """Print summary of all experiments"""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        for exp_name, data in self.experiments.items():
            print(f"\n📊 {exp_name}")
            print(f"   Status: {data['status']}")
            if data['duration']:
                print(f"   Duration: {data['duration']:.2f}s")
            print(f"   Metrics:")
            for key, value in data['metrics'].items():
                if isinstance(value, float):
                    print(f"     - {key}: {value:.4f}")
                else:
                    print(f"     - {key}: {value}")
        
        print("\n" + "="*80)


def create_result_summary(experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create comprehensive summary of all experiment results"""
    summary = {
        "total_experiments": len(experiments),
        "timestamp": datetime.now().isoformat(),
        "experiments": {}
    }
    
    for exp in experiments:
        exp_name = exp.get("name", "unknown")
        summary["experiments"][exp_name] = {
            "status": exp.get("status", "unknown"),
            "metrics": exp.get("metrics", {}),
            "duration": exp.get("duration"),
            "output_file": exp.get("output_file")
        }
    
    return summary
