#!/usr/bin/env python
"""Regenerate the paper figure set from the current summary CSV files.

Upgraded to use clean, minimal, journal-quality aesthetics matching IEEE standards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "results" / "plots"
PAPER_FIGURES_DIR = ROOT / "paper" / "figures"
RESULTS_FIGURES_DIR = PLOTS_DIR / "figures"
OUTPUT_DIRS = [PAPER_FIGURES_DIR, RESULTS_FIGURES_DIR]

# Configure global publication styling parameters
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.titleweight': 'bold',
    'pdf.fonttype': 42,  # Embed true fonts in PDFs
    'ps.fonttype': 42
})

# Curated Journal-quality Color Palette
COLOR_PRIMARY = '#1A365D'    # Deep Navy
COLOR_SECONDARY = '#2C7A7B'  # Teal/Sage
COLOR_ACCENT = '#D69E2E'     # Muted Gold
COLOR_ALERT = '#9B2C2C'      # Muted Crimson
COLOR_NEUTRAL = '#718096'    # Muted Grey

PALETTE_MUTED = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT, COLOR_NEUTRAL, '#4A5568']


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(PLOTS_DIR / name)


def percent_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype(str).str.rstrip("%"), errors="coerce")


def percent_heights(values: pd.Series) -> pd.Series:
    numeric = percent_series(values)
    if numeric.dropna().max() <= 1.5:
        numeric = numeric * 100.0
    return numeric


def prepare_output_dirs() -> None:
    for directory in OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("*.pdf"):
            path.unlink()
        for path in directory.glob("*.png"):
            path.unlink()


def save_figure(fig: plt.Figure, basename: str) -> None:
    for directory in OUTPUT_DIRS:
        fig.savefig(directory / f"{basename}.pdf", bbox_inches="tight")
        fig.savefig(directory / f"{basename}.png", dpi=300, bbox_inches="tight")


def _beautify_axes(ax: plt.Axes) -> None:
    # Set subtle grid lines
    ax.grid(True, axis="y", color="#E2E8F0", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    
    # Despine (remove top/right borders)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Thicken remaining spines for print clarity
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color('#4A5568')
    ax.spines['bottom'].set_color('#4A5568')


def figure1_main_results() -> None:
    df = load_csv("paper_main_results_table.csv")
    models = df["Model"].tolist()
    auroc = df["AUROC"].astype(float).tolist()
    recall_values = percent_heights(df["Recall"])

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)

    # Plot AUROC
    axes[0].bar(models, auroc, color=PALETTE_MUTED[:len(models)], edgecolor="#2D3748", linewidth=0.5, width=0.55)
    axes[0].set_title("Validation AUROC")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0.82, 0.91)
    axes[0].axhline(0.85, color=COLOR_ALERT, linestyle="--", linewidth=1.2, label="Clinical Target (0.85)")
    axes[0].tick_params(axis="x", rotation=15)
    for bar, value in zip(axes[0].patches, auroc):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.002, 0.905),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#2D3748"
        )
    _beautify_axes(axes[0])

    # Plot Recall
    recall_bars = axes[1].bar(
        models,
        recall_values.fillna(0.0),
        color=PALETTE_MUTED[:len(models)],
        edgecolor="#2D3748",
        linewidth=0.5,
        width=0.55
    )
    for bar, raw_value, model in zip(recall_bars, df["Recall"], models):
        if pd.isna(raw_value) or str(raw_value).strip().upper() == "N/A":
            bar.set_hatch("///")
            bar.set_facecolor("#EDF2F7")
            bar.set_edgecolor(COLOR_NEUTRAL)
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                4.0,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=COLOR_NEUTRAL
            )
        else:
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                min(bar.get_height() + 2.0, 98.0),
                f"{bar.get_height():.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#2D3748"
            )
    axes[1].set_title("Clinical Recall (Sensitivity)")
    axes[1].set_ylabel("Recall (%)")
    axes[1].set_ylim(0, 100)
    axes[1].axhline(80.0, color=COLOR_ALERT, linestyle="--", linewidth=1.2, label="Clinical Target (80%)")
    axes[1].tick_params(axis="x", rotation=15)
    _beautify_axes(axes[1])

    fig.suptitle("Centralized vs. Federated Learning Performance on MIMIC-IV", y=1.02)
    fig.tight_layout()
    save_figure(fig, "figure1_main_results")
    plt.close(fig)


def figure2_scalability() -> None:
    df = load_csv("exp9_scalability_analysis.csv")
    scaling = df[df["dropout_fraction"] == 0.0].sort_values("n_clients")
    dropout = df[df["dropout_fraction"] > 0.0].sort_values("dropout_fraction")

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # Plot 1: Scaling Client Count
    ax = axes[0]
    ax2 = ax.twinx()
    
    line1 = ax.plot(scaling["n_clients"], scaling["auroc"], marker="o", color=COLOR_PRIMARY, linewidth=1.5, label="AUROC")
    line2 = ax2.plot(
        scaling["n_clients"],
        scaling["throughput_samples_per_sec"],
        marker="s",
        color=COLOR_ACCENT,
        linewidth=1.5,
        label="Throughput",
    )
    ax.set_title("Scaling Client Node Count")
    ax.set_xlabel("Number of Client Sites")
    ax.set_ylabel("Validation AUROC", color=COLOR_PRIMARY)
    ax2.set_ylabel("Throughput (samples/sec)", color=COLOR_ACCENT)
    ax.set_ylim(0.86, 0.90)
    ax.tick_params(axis="y", labelcolor=COLOR_PRIMARY)
    ax2.tick_params(axis="y", labelcolor=COLOR_ACCENT)
    
    # Custom combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    _beautify_axes(ax)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # Plot 2: Client Dropout Stress Test
    ax = axes[1]
    dropout_labels = [f"{int(frac * 100)}%" for frac in dropout["dropout_fraction"]]
    x = range(len(dropout_labels))
    width = 0.30
    
    rects1 = ax.bar([i - width / 2 for i in x], dropout["auroc"], width=width, color=COLOR_PRIMARY, edgecolor="#2D3748", linewidth=0.5, label="AUROC")
    ax2 = ax.twinx()
    rects2 = ax2.bar([i + width / 2 for i in x], dropout["throughput_samples_per_sec"], width=width, color=COLOR_ACCENT, edgecolor="#2D3748", linewidth=0.5, label="Throughput")
    
    ax.set_xticks(list(x))
    ax.set_xticklabels(dropout_labels)
    ax.set_title("Client Dropout Stress Test")
    ax.set_xlabel("Simulated Network Offline Rate")
    ax.set_ylabel("Validation AUROC", color=COLOR_PRIMARY)
    ax2.set_ylabel("Throughput (samples/sec)", color=COLOR_ACCENT)
    ax.set_ylim(0.86, 0.90)
    ax.tick_params(axis="y", labelcolor=COLOR_PRIMARY)
    ax2.tick_params(axis="y", labelcolor=COLOR_ACCENT)
    
    _beautify_axes(ax)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    fig.suptitle("System Scalability & Network Robustness Analysis", y=1.02)
    fig.tight_layout()
    save_figure(fig, "figure2_scalability")
    plt.close(fig)


def figure3_aggregation() -> None:
    df = load_csv("paper_main_results_table.csv")

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    subset = df.iloc[:3].copy()
    axes[0].bar(subset["Model"], subset["AUROC"].astype(float), color=[COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT], edgecolor="#2D3748", linewidth=0.5, width=0.5)
    axes[0].set_title("Aggregation Optimization Strategy")
    axes[0].set_ylabel("Validation AUROC")
    axes[0].set_ylim(0.82, 0.91)
    axes[0].tick_params(axis="x", rotation=10)
    _beautify_axes(axes[0])

    recall_df = df[percent_series(df["Recall"]).notna()].copy()
    axes[1].bar(recall_df["Model"], percent_series(recall_df["Recall"]), color=COLOR_NEUTRAL, edgecolor="#2D3748", linewidth=0.5, width=0.5)
    axes[1].set_title("Clinical Sensitivity (Recall)")
    axes[1].set_ylabel("Recall (%)")
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis="x", rotation=10)
    _beautify_axes(axes[1])

    fig.suptitle("Comparing Traditional Federated Aggregation Protocols", y=1.02)
    fig.tight_layout()
    save_figure(fig, "figure3_aggregation")
    plt.close(fig)


def figure4_feature_drift() -> None:
    # Load with feature names as index
    aggregated = pd.read_csv(PLOTS_DIR / "exp5_feature_importance_aggregated.csv", index_col=0)
    drift = pd.read_csv(PLOTS_DIR / "exp5_high_drift_features.csv", index_col=0)

    # Complete mapping of Feature indices to clinical names from config
    feature_names = {
        "Feature_0": "Age",
        "Feature_1": "Gender (Male)",
        "Feature_2": "Emergency Admission",
        "Feature_3": "Medicare Insurance",
        "Feature_4": "Heart Rate (Mean)",
        "Feature_5": "Heart Rate (Min)",
        "Feature_6": "Heart Rate (Max)",
        "Feature_7": "Systolic BP (Mean)",
        "Feature_8": "Systolic BP (Min)",
        "Feature_9": "Mean Arterial Pressure (Mean)",
        "Feature_10": "Mean Arterial Pressure (Min)",
        "Feature_11": "Respiratory Rate (Mean)",
        "Feature_12": "Respiratory Rate (Max)",
        "Feature_13": "Temperature (Mean)",
        "Feature_14": "SpO₂ (Mean)",
        "Feature_15": "SpO₂ (Min)",
        "Feature_16": "Glucose (Mean)",
        "Feature_17": "Creatinine (Max)",
        "Feature_18": "BUN (Max)",
        "Feature_19": "Sodium (Min)",
        "Feature_20": "Sodium (Max)",
        "Feature_21": "Potassium (Max)",
        "Feature_22": "Bicarbonate (Min)",
        "Feature_23": "Hemoglobin (Min)",
        "Feature_24": "WBC (Max)",
        "Feature_25": "Platelets (Min)",
        "Feature_26": "Lactate (Max)",
        "Feature_27": "Bilirubin (Max)",
        "Feature_28": "INR (Max)",
        "Feature_29": "SOFA Score",
        "Feature_30": "SAPS II Score",
        "Feature_31": "Charlson Index",
    }

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    top_mean = aggregated.head(10).iloc[::-1]
    # Rename features in the index
    top_mean_labels = [feature_names.get(label, label) for label in top_mean.index]
    axes[0].barh(range(len(top_mean)), top_mean["mean"], color=COLOR_PRIMARY, edgecolor="#2D3748", linewidth=0.5, height=0.7)
    
    # Explicitly set y-tick labels with proper encoding
    axes[0].set_yticks(range(len(top_mean)), labels=top_mean_labels, fontsize=10)
    axes[0].tick_params(axis='y', labelsize=10)
    
    axes[0].set_title("Mean Feature Importance Across Sites", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Mean SHAP/Weight Importance", fontsize=11)
    axes[0].margins(y=0.01)
    # Horizontal grid
    axes[0].grid(True, axis="x", color="#E2E8F0", linestyle="--", linewidth=0.5, alpha=0.7)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    top_drift = drift.head(10).iloc[::-1]
    # Rename features in the index
    top_drift_labels = [feature_names.get(label, label) for label in top_drift.index]
    axes[1].barh(range(len(top_drift)), top_drift["cv"], color=COLOR_ACCENT, edgecolor="#2D3748", linewidth=0.5, height=0.7)
    
    # Explicitly set y-tick labels with proper encoding
    axes[1].set_yticks(range(len(top_drift)), labels=top_drift_labels, fontsize=10)
    axes[1].tick_params(axis='y', labelsize=10)
    
    axes[1].set_title("High-Drift Clinical Features", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Coefficient of Variation (CV)", fontsize=11)
    axes[1].margins(y=0.01)
    # Horizontal grid
    axes[1].grid(True, axis="x", color="#E2E8F0", linestyle="--", linewidth=0.5, alpha=0.7)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    fig.suptitle("Clinical Feature Drift & Importance Across ICUs", y=0.99, fontsize=13, fontweight='bold')
    # Use subplots_adjust with left margin to accommodate y-labels and space for title
    plt.subplots_adjust(left=0.25, right=0.98, top=0.88, bottom=0.12, wspace=0.35)
    save_figure(fig, "figure4_feature_drift")
    plt.close(fig)


def figure5_privacy_tradeoff() -> None:
    df = load_csv("exp7_differential_privacy.csv")
    df["condition"] = df["use_dp"].map({False: "Baseline", True: "DP Enabled"})

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    metrics = ["auroc", "brier", "ece"]
    titles = ["AUROC (Utility)", "Brier Score (Calibration)", "ECE (Calibration Error)"]
    colors = {"Baseline": COLOR_PRIMARY, "DP Enabled": COLOR_ACCENT}

    for ax, metric, title in zip(axes, metrics, titles):
        sns.boxplot(data=df, x="condition", y=metric, ax=ax, palette=colors, width=0.45, linewidth=1.0, fliersize=0)
        sns.stripplot(data=df, x="condition", y=metric, ax=ax, color="#2D3748", size=3.5, alpha=0.7, jitter=0.1)
        ax.set_title(title)
        ax.set_xlabel("")
        _beautify_axes(ax)

    fig.suptitle("Privacy-Utility Trade-off under DP-SGD (ε=1.0, δ=10^-5)", y=1.03)
    fig.tight_layout()
    save_figure(fig, "figure5_privacy_tradeoff")
    plt.close(fig)


def figure6_byzantine() -> None:
    df = load_csv("exp8_adversarial_robustness.csv")
    df["scenario_label"] = df.apply(
        lambda row: "Clean" if float(row["malicious_fraction"]) == 0.0 else f'{int(float(row["malicious_fraction"]) * 100)}% Byzantine',
        axis=1,
    )

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    metrics = ["auroc", "recall", "precision"]
    titles = ["Validation AUROC", "Clinical Recall", "Precision (PPV)"]
    colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT]

    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        ax.bar(df["scenario_label"], df[metric], color=color, edgecolor="#2D3748", linewidth=0.5, width=0.45)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=10)
        _beautify_axes(ax)

    fig.suptitle("Federated System Robustness to Byzantine Attackers", y=1.02)
    fig.tight_layout()
    save_figure(fig, "figure6_byzantine")
    plt.close(fig)


def figure7_calibration() -> None:
    df = load_csv("exp7_differential_privacy_summary.csv")
    df = df.rename(columns={"experiment": "condition"})

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].scatter(df["brier"], df["auroc"], s=100, color=COLOR_PRIMARY, edgecolor="#2D3748", linewidth=0.8, alpha=0.9)
    for _, row in df.iterrows():
        axes[0].annotate(row["condition"], (row["brier"], row["auroc"]), xytext=(5, 3), textcoords="offset points", fontsize=8.5)
    axes[0].set_xlabel("Brier Score (Lower is Better)")
    axes[0].set_ylabel("Validation AUROC")
    axes[0].set_title("Brier Score vs. Utility")
    _beautify_axes(axes[0])

    axes[1].scatter(df["ece"], df["auroc"], s=100, color=COLOR_ACCENT, edgecolor="#2D3748", linewidth=0.8, alpha=0.9)
    for _, row in df.iterrows():
        axes[1].annotate(row["condition"], (row["ece"], row["auroc"]), xytext=(5, 3), textcoords="offset points", fontsize=8.5)
    axes[1].set_xlabel("Expected Calibration Error (ECE)")
    axes[1].set_ylabel("Validation AUROC")
    axes[1].set_title("ECE vs. Utility")
    _beautify_axes(axes[1])

    fig.suptitle("Calibration View of Differential Privacy Budgets", y=1.02)
    fig.tight_layout()
    save_figure(fig, "figure7_calibration")
    plt.close(fig)


def figure8_statistical_validation() -> None:
    df = load_csv("exp6_statistical_validation_summary.csv")

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    metrics = ["auroc", "brier", "ece"]
    labels = ["Validation AUROC", "Brier Score", "ECE"]

    for ax, metric, label in zip(axes, metrics, labels):
        mean_col = f"{metric}_mean"
        low_col = f"{metric}_ci_low"
        high_col = f"{metric}_ci_high"
        means = df[mean_col].astype(float)
        lower = means - df[low_col].astype(float)
        upper = df[high_col].astype(float) - means
        ax.errorbar(df["strategy"], means, yerr=[lower, upper], fmt="o", capsize=4, color=COLOR_PRIMARY, elinewidth=1.2, markeredgecolor="#2D3748")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=10)
        _beautify_axes(ax)

    fig.suptitle("Multi-Seed Statistical Confidence Intervals (95% CI)", y=1.02)
    fig.tight_layout()
    save_figure(fig, "figure8_statistical_validation")
    plt.close(fig)


def main() -> None:
    prepare_output_dirs()
    figure1_main_results()
    figure2_scalability()
    figure3_aggregation()
    figure4_feature_drift()
    figure5_privacy_tradeoff()
    figure6_byzantine()
    figure7_calibration()
    figure8_statistical_validation()


if __name__ == "__main__":
    main()