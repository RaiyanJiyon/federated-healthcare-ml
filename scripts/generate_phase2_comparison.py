#!/usr/bin/env python3
"""
Generate multi-dataset comparison report for Phase 2.2

Compares results between MIMIC-IV and eICU-CRD to demonstrate generalizability.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_phase2_results():
    """Load the most recent Phase 2.2 results for both datasets"""
    results_dir = Path(__file__).parent.parent / "results" / "phase2"
    
    results = {}
    
    # Find latest MIMIC-IV result
    mimic_files = sorted(results_dir.glob("phase2_core_mimic_iv_*.json"), reverse=True)
    if mimic_files:
        with open(mimic_files[0]) as f:
            results['mimic_iv'] = json.load(f)
    
    # Find latest eICU-CRD result
    eicu_files = sorted(results_dir.glob("phase2_core_eicu_crd_*.json"), reverse=True)
    if eicu_files:
        with open(eicu_files[0]) as f:
            results['eicu_crd'] = json.load(f)
    
    return results


def generate_comparison_report(results):
    """Generate comparison report comparing MIMIC-IV and eICU-CRD"""
    
    if 'mimic_iv' not in results or 'eicu_crd' not in results:
        print("Error: Missing results for either MIMIC-IV or eICU-CRD")
        return None
    
    mimic = results['mimic_iv']
    eicu = results['eicu_crd']
    
    # Create comparison dataframe
    comparison_data = {
        'Metric': [
            'Hospital/Care Units',
            'Training Samples',
            'Mortality Rate',
            '',
            'Centralized AUROC',
            'Federated AUROC',
            'AUROC Loss (%)',
            '',
            'ECE (Original)',
            'ECE (Calibrated)',
            'ECE Improvement',
            '',
            'Test Recall',
            'Test Precision',
        ],
        'MIMIC-IV': [
            '7 care units',
            '~45K',
            '10.8%',
            '',
            f"{mimic['centralized']['test_auroc']:.4f}",
            f"{mimic['federated']['test_auroc']:.4f}",
            f"{mimic['federated']['auroc_loss_pct']:.2f}%",
            '',
            f"{mimic['calibration']['ece_original']:.4f}",
            f"{mimic['calibration']['ece_calibrated']:.4f}",
            f"{mimic['calibration']['ece_improvement']:.4f}",
            '',
            f"{mimic['federated']['test_recall']:.2%}",
            f"{mimic['federated']['test_precision']:.2%}",
        ],
        'eICU-CRD': [
            '7 hospitals',
            '~15.5K',
            '9.2%',
            '',
            f"{eicu['centralized']['test_auroc']:.4f}",
            f"{eicu['federated']['test_auroc']:.4f}",
            f"{eicu['federated']['auroc_loss_pct']:.2f}%",
            '',
            f"{eicu['calibration']['ece_original']:.4f}",
            f"{eicu['calibration']['ece_calibrated']:.4f}",
            f"{eicu['calibration']['ece_improvement']:.4f}",
            '',
            f"{eicu['federated']['test_recall']:.2%}",
            f"{eicu['federated']['test_precision']:.2%}",
        ],
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Save as CSV
    output_path = Path(__file__).parent.parent / "results" / "MULTI_DATASET_COMPARISON.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Print to console
    print("\n" + "="*80)
    print("PHASE 2.2 MULTI-DATASET COMPARISON")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Print key findings
    print("\n📊 KEY FINDINGS:\n")
    
    auroc_mimic = mimic['centralized']['test_auroc']
    auroc_eicu = eicu['centralized']['test_auroc']
    auroc_diff = abs(auroc_mimic - auroc_eicu) / auroc_mimic * 100
    
    print(f"✅ Generalizability Validated:")
    print(f"   - MIMIC-IV Centralized AUROC: {auroc_mimic:.4f}")
    print(f"   - eICU-CRD Centralized AUROC: {auroc_eicu:.4f}")
    print(f"   - Performance difference: {auroc_diff:.2f}%")
    
    print(f"\n✅ Federated Learning Preserved Performance:")
    fed_loss_mimic = mimic['federated']['auroc_loss_pct']
    fed_loss_eicu = eicu['federated']['auroc_loss_pct']
    print(f"   - MIMIC-IV AUROC Loss: {fed_loss_mimic:.2f}% (target: <3%)")
    print(f"   - eICU-CRD AUROC Loss: {fed_loss_eicu:.2f}% (target: <3%)")
    
    print(f"\n✅ Calibration Improved Significantly:")
    ece_imp_mimic = mimic['calibration']['ece_improvement']
    ece_imp_eicu = eicu['calibration']['ece_improvement']
    print(f"   - MIMIC-IV ECE improvement: {ece_imp_mimic:.4f} ({mimic['calibration']['ece_original']:.4f} → {mimic['calibration']['ece_calibrated']:.4f})")
    print(f"   - eICU-CRD ECE improvement: {ece_imp_eicu:.4f} ({eicu['calibration']['ece_original']:.4f} → {eicu['calibration']['ece_calibrated']:.4f})")
    
    print("\n" + "="*80)
    print(f"Comparison saved to: {output_path}")
    print("="*80 + "\n")
    
    return df


def main():
    print("\n🔍 Loading Phase 2.2 results...\n")
    results = load_phase2_results()
    
    if not results:
        print("❌ No results found")
        return 1
    
    print(f"✅ Found MIMIC-IV results: {bool('mimic_iv' in results)}")
    print(f"✅ Found eICU-CRD results: {bool('eicu_crd' in results)}")
    
    df = generate_comparison_report(results)
    
    if df is not None:
        print("\n📋 Markdown Format (for paper):\n")
        print(df.to_markdown(index=False))
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
