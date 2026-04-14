#!/usr/bin/env python3
"""
Visualization utilities for scalability analysis results

Generates plots for:
- Time vs client count
- Recall vs client count
- Communication overhead vs client count
- Memory usage vs client count
- Scaling law fitting
- Bottleneck analysis
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime


class ScalabilityVisualizer:
    """Create publication-ready visualizations for scalability analysis"""
    
    def __init__(self, result_file: str):
        """Load and parse scalability analysis results"""
        with open(result_file, 'r') as f:
            self.data = json.load(f)
        
        # Extract data
        self.client_counts = []
        self.times = []
        self.recalls = []
        self.communications = []
        self.memories = []
        self.accuracies = []
        self.per_client_costs = []
        
        for num_clients_str in sorted(self.data['scalability_by_client_count'].keys(), 
                                     key=lambda x: int(x)):
            num_clients = int(num_clients_str)
            result = self.data['scalability_by_client_count'][num_clients_str]
            
            self.client_counts.append(num_clients)
            self.times.append(result['fl_training_time_s'])
            self.recalls.append(result['metrics']['recall'] * 100)
            self.communications.append(result['communication']['total_overall_mb'])
            self.memories.append(result['peak_memory_mb'])
            self.accuracies.append(result['metrics']['accuracy'] * 100)
            self.per_client_costs.append(result['per_client_avg_time_s'] * 1000)  # ms
        
        # Setup figure
        self.set_style()
    
    @staticmethod
    def set_style():
        """Set matplotlib style for publication-quality plots"""
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['figure.figsize'] = (14, 10)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 13
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['lines.markersize'] = 8
    
    def plot_comprehensive_analysis(self, output_file: str = None):
        """Create comprehensive 6-panel scalability analysis figure"""
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Total Training Time vs Client Count
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(self.client_counts, self.times, 'o-', color='#1f77b4', linewidth=2.5, markersize=8)
        ax1.set_xlabel('Number of Clients')
        ax1.set_ylabel('Total Training Time (seconds)')
        ax1.set_title('A) Training Time Scaling')
        ax1.grid(True, alpha=0.3)
        ax1.fill_between(self.client_counts, self.times, alpha=0.2, color='#1f77b4')
        
        # Add scaling law annotation
        scaling_law = self.data['scaling_laws']['best_fit']
        ax1.text(0.98, 0.05, f'Best fit: {scaling_law}\nR² = {self.data["scaling_laws"]["best_fit_r2"]:.4f}',
                transform=ax1.transAxes, fontsize=9, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. Recall vs Client Count
        ax2 = plt.subplot(2, 3, 2)
        safety_line = [80] * len(self.client_counts)
        ax2.plot(self.client_counts, self.recalls, 'o-', color='#2ca02c', linewidth=2.5, markersize=8, label='Actual Recall')
        ax2.axhline(y=80, color='red', linestyle='--', linewidth=2, label='Safety Threshold (80%)')
        ax2.set_xlabel('Number of Clients')
        ax2.set_ylabel('Recall (%)')
        ax2.set_title('B) Clinical Safety (Recall) vs Clients')
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        ax2.fill_between(self.client_counts, self.recalls, 80, alpha=0.2, 
                        color=['red' if r < 80 else 'green' for r in self.recalls])
        
        # 3. Communication Overhead
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(self.client_counts, self.communications, 's-', color='#ff7f0e', linewidth=2.5, markersize=8)
        ax3.set_xlabel('Number of Clients')
        ax3.set_ylabel('Total Communication (MB)')
        ax3.set_title('C) Communication Overhead Scaling')
        ax3.grid(True, alpha=0.3)
        ax3.fill_between(self.client_counts, self.communications, alpha=0.2, color='#ff7f0e')
        
        # 4. Memory Usage
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(self.client_counts, self.memories, '^-', color='#d62728', linewidth=2.5, markersize=8)
        ax4.set_xlabel('Number of Clients')
        ax4.set_ylabel('Peak Memory (MB)')
        ax4.set_title('D) Memory Usage vs Clients')
        ax4.grid(True, alpha=0.3)
        ax4.fill_between(self.client_counts, self.memories, alpha=0.2, color='#d62728')
        
        # 5. Per-Client Computation Cost
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(self.client_counts, self.per_client_costs, 'D-', color='#9467bd', linewidth=2.5, markersize=8)
        ax5.set_xlabel('Number of Clients')
        ax5.set_ylabel('Per-Client Cost (ms per round)')
        ax5.set_title('E) Per-Client Computational Cost')
        ax5.grid(True, alpha=0.3)
        ax5.fill_between(self.client_counts, self.per_client_costs, alpha=0.2, color='#9467bd')
        
        # 6. Throughput (Rounds per Second)
        ax6 = plt.subplot(2, 3, 6)
        throughputs = [10 / t for t in self.times]  # 10 rounds per exper
        ax6.plot(self.client_counts, throughputs, 'v-', color='#8c564b', linewidth=2.5, markersize=8)
        ax6.set_xlabel('Number of Clients')
        ax6.set_ylabel('Throughput (rounds/second)')
        ax6.set_title('F) System Throughput')
        ax6.grid(True, alpha=0.3)
        ax6.fill_between(self.client_counts, throughputs, alpha=0.2, color='#8c564b')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved comprehensive plot to: {output_file}")
        
        return fig
    
    def plot_scaling_laws(self, output_file: str = None):
        """Plot scaling law fits (linear, polynomial, exponential)"""
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        
        client_counts_arr = np.array(self.client_counts, dtype=float)
        times_arr = np.array(self.times, dtype=float)
        
        # Linear fit
        linear_fit = self.data['scaling_laws']['linear']['fit']
        linear_r2 = self.data['scaling_laws']['linear']['r2']
        linear_pred = linear_fit[0] * client_counts_arr + linear_fit[1]
        
        axes[0].scatter(self.client_counts, self.times, s=100, color='#1f77b4', alpha=0.7, label='Observed')
        axes[0].plot(client_counts_arr, linear_pred, '--', color='red', linewidth=2.5, label='Linear Fit')
        axes[0].set_xlabel('Number of Clients')
        axes[0].set_ylabel('Training Time (seconds)')
        axes[0].set_title(f'Linear Model (R² = {linear_r2:.4f})\nT = {linear_fit[0]:.4f}·C + {linear_fit[1]:.4f}')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Polynomial fit
        poly_fit = self.data['scaling_laws']['polynomial']['fit']
        poly_r2 = self.data['scaling_laws']['polynomial']['r2']
        poly_pred = poly_fit[0] * client_counts_arr**2 + poly_fit[1] * client_counts_arr + poly_fit[2]
        
        axes[1].scatter(self.client_counts, self.times, s=100, color='#2ca02c', alpha=0.7, label='Observed')
        axes[1].plot(client_counts_arr, poly_pred, '--', color='red', linewidth=2.5, label='Polynomial Fit')
        axes[1].set_xlabel('Number of Clients')
        axes[1].set_ylabel('Training Time (seconds)')
        axes[1].set_title(f'Polynomial Model (R² = {poly_r2:.4f})\nT = {poly_fit[0]:.4f}·C² + {poly_fit[1]:.4f}·C + {poly_fit[2]:.4f}')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Exponential fit
        exp_fit = self.data['scaling_laws']['exponential']['fit']
        exp_r2 = self.data['scaling_laws']['exponential']['r2']
        exp_pred = np.exp(exp_fit[0] * client_counts_arr + exp_fit[1])
        
        axes[2].scatter(self.client_counts, self.times, s=100, color='#ff7f0e', alpha=0.7, label='Observed')
        axes[2].plot(client_counts_arr, exp_pred, '--', color='red', linewidth=2.5, label='Exponential Fit')
        axes[2].set_xlabel('Number of Clients')
        axes[2].set_ylabel('Training Time (seconds)')
        axes[2].set_title(f'Exponential Model (R² = {exp_r2:.4f})\nT = exp({exp_fit[0]:.4f}·C + {exp_fit[1]:.4f})')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved scaling laws plot to: {output_file}")
        
        return fig
    
    def plot_bottleneck_analysis(self, output_file: str = None):
        """Visualize bottleneck breakdown (local training vs aggregation)"""
        
        # Extract bottleneck data
        bottlenecks = self.data['bottleneck_analysis']
        local_pcts = []
        agg_pcts = []
        
        for num_clients_str in sorted(bottlenecks.keys(), key=lambda x: int(x)):
            local_pcts.append(bottlenecks[num_clients_str]['local_training_pct'])
            agg_pcts.append(100 - bottlenecks[num_clients_str]['local_training_pct'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Stacked bar chart
        x_pos = np.arange(len(self.client_counts))
        ax1.bar(x_pos, local_pcts, label='Local Training', color='#1f77b4', alpha=0.8)
        ax1.bar(x_pos, agg_pcts, bottom=local_pcts, label='Aggregation', color='#ff7f0e', alpha=0.8)
        ax1.set_xlabel('Number of Clients')
        ax1.set_ylabel('Time Percentage (%)')
        ax1.set_title('Per-Round Time Breakdown')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(self.client_counts)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 100])
        
        # Lines showing dominant bottleneck
        ax2.plot(self.client_counts, local_pcts, 'o-', label='Local Training %', 
                linewidth=2.5, markersize=8, color='#1f77b4')
        ax2.axhline(y=60, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                   label='Bottleneck Threshold (60%)')
        ax2.fill_between(self.client_counts, 60, 100, alpha=0.1, color='red')
        ax2.fill_between(self.client_counts, 0, 60, alpha=0.1, color='green')
        ax2.set_xlabel('Number of Clients')
        ax2.set_ylabel('Local Training Time (%)')
        ax2.set_title('Bottleneck Identification')
        ax2.set_ylim([0, 105])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved bottleneck analysis plot to: {output_file}")
        
        return fig
    
    def create_summary_table(self, output_file: str = None):
        """Create comprehensive summary table"""
        
        summary_data = {
            'Clients': self.client_counts,
            'Time (s)': [f"{t:.2f}" for t in self.times],
            'Recall (%)': [f"{r:.1f}" if r >= 80 else f"❌ {r:.1f}" for r in self.recalls],
            'Accuracy (%)': [f"{a:.1f}" for a in self.accuracies],
            'Comm (MB)': [f"{c:.3f}" for c in self.communications],
            'Memory (MB)': [f"{m:.1f}" for m in self.memories],
            'Per-Client Cost (ms)': [f"{p:.2f}" for p in self.per_client_costs],
            'Throughput (rounds/s)': [f"{10/t:.2f}" for t in self.times],
        }
        
        df = pd.DataFrame(summary_data)
        
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"✓ Saved summary table to: {output_file}")
        
        print("\n" + "=" * 120)
        print("SCALABILITY ANALYSIS SUMMARY TABLE")
        print("=" * 120)
        print(df.to_string(index=False))
        print("=" * 120 + "\n")
        
        return df


def visualize_scalability_results(result_file: str):
    """Main visualization function"""
    
    print("\n" + "=" * 120)
    print("GENERATING SCALABILITY VISUALIZATIONS")
    print("=" * 120 + "\n")
    
    # Load and create visualizer
    visualizer = ScalabilityVisualizer(result_file)
    
    # Create output directory
    output_dir = Path(result_file).parent / 'plots'
    output_dir.mkdir(exist_ok=True)
    
    # Generate all visualizations
    print("📊 Creating visualizations...")
    
    # 1. Comprehensive analysis
    fig1 = visualizer.plot_comprehensive_analysis(
        str(output_dir / 'scalability_comprehensive_analysis.pdf')
    )
    
    # 2. Scaling laws
    fig2 = visualizer.plot_scaling_laws(
        str(output_dir / 'scalability_scaling_laws.pdf')
    )
    
    # 3. Bottleneck analysis
    fig3 = visualizer.plot_bottleneck_analysis(
        str(output_dir / 'scalability_bottleneck_analysis.pdf')
    )
    
    # 4. Summary table
    df = visualizer.create_summary_table(
        str(output_dir / 'scalability_summary_table.csv')
    )
    
    print(f"\n✓ All visualizations saved to: {output_dir}")
    print("=" * 120 + "\n")
    
    return output_dir


if __name__ == '__main__':
    # Find latest scalability analysis result
    results_dir = Path(__file__).parent.parent / 'results'
    
    # Get all scalability analysis files
    scalability_files = sorted(results_dir.glob('scalability_analysis_*.json'))
    
    if not scalability_files:
        print("❌ No scalability analysis results found!")
        sys.exit(1)
    
    # Use the latest one
    latest_result = scalability_files[-1]
    print(f"Using results from: {latest_result}\n")
    
    visualize_scalability_results(str(latest_result))
    
    print("✅ Visualization complete!\n")
