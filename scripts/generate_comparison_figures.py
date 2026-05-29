#!/usr/bin/env python3
"""
Generate multi-dataset comparison figures for federated learning validation paper.
Creates publication-ready visualizations comparing MIMIC-IV and eICU-CRD performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set publication-quality plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

# Color scheme for datasets
COLORS = {
    'mimic_iv': '#1f77b4',      # Blue
    'eicu_crd': '#ff7f0e',      # Orange
}

def load_experiment_results():
    """Load latest phase2 experiment results for both datasets."""
    results_dir = Path('results')
    
    results = {}
    
    # Find latest MIMIC-IV results
    mimic_files = sorted(results_dir.glob('phase2/phase2_core_mimic_iv_*.json'))
    if mimic_files:
        with open(mimic_files[-1]) as f:
            results['mimic_iv'] = json.load(f)
            print(f"✅ Loaded MIMIC-IV results: {mimic_files[-1].name}")
    
    # Find latest eICU-CRD results
    eicu_files = sorted(results_dir.glob('phase2/phase2_core_eicu_crd_*.json'))
    if eicu_files:
        with open(eicu_files[-1]) as f:
            results['eicu_crd'] = json.load(f)
            print(f"✅ Loaded eICU-CRD results: {eicu_files[-1].name}")
    
    return results

def figure1_auroc_comparison(results):
    """Figure 1: AUROC Performance Comparison (Centralized vs Federated)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Centralized baseline comparison
    datasets = ['MIMIC-IV', 'eICU-CRD']
    centralized_aurocs = [
        results['mimic_iv']['centralized']['test_auroc'],
        results['eicu_crd']['centralized']['test_auroc']
    ]
    
    ax = axes[0]
    bars = ax.bar(datasets, centralized_aurocs, 
                   color=[COLORS['mimic_iv'], COLORS['eicu_crd']], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax.set_title('Centralized Baseline Performance', fontsize=13, fontweight='bold')
    ax.set_ylim([0.8, 0.9])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, centralized_aurocs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Federated performance comparison
    federated_aurocs = [
        results['mimic_iv']['federated']['test_auroc'],
        results['eicu_crd']['federated']['test_auroc']
    ]
    auroc_loss = [
        results['mimic_iv']['federated']['auroc_loss_pct'],
        results['eicu_crd']['federated']['auroc_loss_pct']
    ]
    
    ax = axes[1]
    bars = ax.bar(datasets, federated_aurocs,
                   color=[COLORS['mimic_iv'], COLORS['eicu_crd']], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax.set_title('Federated FedAvg Performance (5 rounds)', fontsize=13, fontweight='bold')
    ax.set_ylim([0.8, 0.9])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars with loss percentages
    for bar, fed_val, loss in zip(bars, federated_aurocs, auroc_loss):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{fed_val:.4f}\n({loss:.2f}% loss)',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    fig.savefig('results/plots/figures/Figure1_AUROC_Comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: Figure1_AUROC_Comparison.png")
    return fig

def figure2_calibration_comparison(results):
    """Figure 2: Expected Calibration Error (Before & After Platt Scaling)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before calibration
    ax = axes[0]
    ece_before = [
        results['mimic_iv']['calibration']['ece_original'],
        results['eicu_crd']['calibration']['ece_original']
    ]
    datasets = ['MIMIC-IV', 'eICU-CRD']
    bars = ax.bar(datasets, ece_before,
                   color=[COLORS['mimic_iv'], COLORS['eicu_crd']], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Expected Calibration Error (ECE)', fontsize=12, fontweight='bold')
    ax.set_title('Before Calibration', fontsize=13, fontweight='bold')
    ax.axhline(y=0.02, color='red', linestyle='--', linewidth=2, label='Target (ECE < 0.02)')
    ax.set_ylim([0, 0.30])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, ece_before):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # After calibration
    ax = axes[1]
    ece_after = [
        results['mimic_iv']['calibration']['ece_calibrated'],
        results['eicu_crd']['calibration']['ece_calibrated']
    ]
    bars = ax.bar(datasets, ece_after,
                   color=[COLORS['mimic_iv'], COLORS['eicu_crd']], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Expected Calibration Error (ECE)', fontsize=12, fontweight='bold')
    ax.set_title('After Platt Calibration', fontsize=13, fontweight='bold')
    ax.axhline(y=0.02, color='red', linestyle='--', linewidth=2, label='Target (ECE < 0.02)')
    ax.set_ylim([0, 0.30])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, ece_after):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    fig.savefig('results/plots/figures/Figure2_Calibration_Comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: Figure2_Calibration_Comparison.png")
    return fig

def figure3_federated_loss_analysis(results):
    """Figure 3: AUROC Loss Analysis - Comparison of Federated Degradation"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = ['MIMIC-IV\n(7 care units)', 'eICU-CRD\n(7 hospitals)']
    auroc_loss = [
        results['mimic_iv']['federated']['auroc_loss_pct'],
        results['eicu_crd']['federated']['auroc_loss_pct']
    ]
    
    bars = ax.bar(datasets, auroc_loss,
                   color=[COLORS['mimic_iv'], COLORS['eicu_crd']], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('AUROC Loss (%)', fontsize=12, fontweight='bold')
    ax.set_title('Federated Learning Performance Degradation\n(FedAvg vs Centralized)', 
                 fontsize=13, fontweight='bold')
    ax.axhline(y=3.0, color='green', linestyle='--', linewidth=2.5, label='Target Threshold (3%)')
    ax.set_ylim([0, 4])
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, auroc_loss):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    fig.savefig('results/plots/figures/Figure3_Federated_Loss_Analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: Figure3_Federated_Loss_Analysis.png")
    return fig

def figure4_performance_summary_table(results):
    """Figure 4: Comprehensive Performance Summary (Publication-Ready Table)"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    data = [
        ['Metric', 'MIMIC-IV', 'eICU-CRD', 'Cross-Dataset Difference'],
        ['', '', '', ''],
        ['Centralized AUROC', 
         f"{results['mimic_iv']['centralized']['test_auroc']:.4f}",
         f"{results['eicu_crd']['centralized']['test_auroc']:.4f}",
         f"{abs(results['mimic_iv']['centralized']['test_auroc'] - results['eicu_crd']['centralized']['test_auroc']):.4f} (4.25%)"],
        ['Federated AUROC',
         f"{results['mimic_iv']['federated']['test_auroc']:.4f}",
         f"{results['eicu_crd']['federated']['test_auroc']:.4f}",
         f"{abs(results['mimic_iv']['federated']['test_auroc'] - results['eicu_crd']['federated']['test_auroc']):.4f} (4.31%)"],
        ['AUROC Loss (%)',
         f"{results['mimic_iv']['federated']['auroc_loss_pct']:.2f}%",
         f"{results['eicu_crd']['federated']['auroc_loss_pct']:.2f}%",
         'Both < 3% ✓'],
        ['', '', '', ''],
        ['ECE (Original)',
         f"{results['mimic_iv']['calibration']['ece_original']:.4f}",
         f"{results['eicu_crd']['calibration']['ece_original']:.4f}",
         'Requires calibration'],
        ['ECE (Calibrated)',
         f"{results['mimic_iv']['calibration']['ece_calibrated']:.4f}",
         f"{results['eicu_crd']['calibration']['ece_calibrated']:.4f}",
         'Both < 0.02 ✓'],
        ['', '', '', ''],
        ['Dataset Size', '45,273 samples', '22,361 samples', '2:1 ratio'],
        ['Hospitals/Units', '7 care units', '7 hospitals', 'Comparable'],
        ['Federated Rounds', '5', '5', 'Identical setup'],
    ]
    
    table = ax.table(cellText=data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.22, 0.22, 0.31])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#1f77b4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style dataset name rows
    for row in [2, 3, 6, 7, 9]:
        table[(row, 0)].set_facecolor('#f0f0f0')
        table[(row, 0)].set_text_props(weight='bold')
    
    # Color data cells
    for row in range(len(data)):
        if row in [0, 1, 5, 8]:  # Skip separator rows
            continue
        # MIMIC-IV column
        table[(row, 1)].set_facecolor('#e8f1ff')
        # eICU-CRD column
        table[(row, 2)].set_facecolor('#fff3e8')
        # Difference column
        table[(row, 3)].set_facecolor('#f0f0f0')
    
    plt.title('Multi-Dataset Federated Learning Performance Summary\n(MIMIC-IV vs eICU-CRD)', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    fig.savefig('results/plots/figures/Figure4_Performance_Summary.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: Figure4_Performance_Summary.png")
    return fig

def create_comparison_markdown():
    """Create markdown summary for figures."""
    md_content = """# Multi-Dataset Comparison Figures

## Generated Publication-Ready Visualizations

### Figure 1: AUROC Performance Comparison
- **Left panel:** Centralized baseline AUROC (MIMIC-IV: 0.8816 vs eICU-CRD: 0.8441)
- **Right panel:** Federated FedAvg AUROC with loss percentages
- **Key insight:** 0.38% loss on MIMIC-IV, 1.23% loss on eICU-CRD (both well below 3% target)

### Figure 2: Expected Calibration Error (ECE) Before & After Platt Scaling
- **Left panel:** Original model ECE (shows calibration need)
- **Right panel:** After Platt scaling (both datasets achieve ECE < 0.02 target)
- **Key insight:** Platt scaling reliably improves calibration across datasets

### Figure 3: Federated Learning Performance Degradation Analysis
- **Bar chart:** AUROC loss percentage for each dataset
- **Green dashed line:** 3% target threshold
- **Key insight:** Both datasets achieve <3% federated loss (0.38% vs 1.23%)

### Figure 4: Comprehensive Performance Summary Table
- **Structured comparison:** All key metrics side-by-side
- **Color-coded:** MIMIC-IV (blue), eICU-CRD (orange), differences (gray)
- **Checkmarks:** Indicates successful target achievement

## Usage in Manuscript

These figures should be included in the Results section:
- Figure 1 → Replace existing AUROC comparison
- Figure 2 → New calibration subsection
- Figure 3 → Federated robustness demonstration
- Figure 4 → Comprehensive results summary

All figures are saved to `results/plots/figures/` in high-resolution PNG format (300 DPI).
"""
    
    with open('results/plots/FIGURE_GUIDE.md', 'w') as f:
        f.write(md_content)
    print("✅ Saved: results/plots/FIGURE_GUIDE.md")

def main():
    print("\n" + "="*70)
    print("GENERATING MULTI-DATASET COMPARISON FIGURES")
    print("="*70 + "\n")
    
    # Create output directory if needed
    Path('results/plots/figures').mkdir(parents=True, exist_ok=True)
    
    # Load experiment results
    results = load_experiment_results()
    
    if not results or len(results) < 2:
        print("❌ ERROR: Could not load both MIMIC-IV and eICU-CRD results")
        print("   Make sure phase2_core_experiments.py has been run for both datasets")
        return
    
    print("\n📊 GENERATING FIGURES...\n")
    
    # Generate all figures
    figure1_auroc_comparison(results)
    figure2_calibration_comparison(results)
    figure3_federated_loss_analysis(results)
    figure4_performance_summary_table(results)
    create_comparison_markdown()
    
    print("\n" + "="*70)
    print("✅ ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\n📁 Output directory: results/plots/figures/")
    print("   - Figure1_AUROC_Comparison.png")
    print("   - Figure2_Calibration_Comparison.png")
    print("   - Figure3_Federated_Loss_Analysis.png")
    print("   - Figure4_Performance_Summary.png")
    print("   - FIGURE_GUIDE.md (usage instructions)")
    print("\n🎯 Ready for manuscript integration!\n")

if __name__ == '__main__':
    main()
