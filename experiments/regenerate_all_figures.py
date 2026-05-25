#!/usr/bin/env python
"""Regenerate the paper figure set from the current summary CSV files."""

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
        fig.savefig(directory / f"{basename}.png", dpi=220, bbox_inches="tight")


def _beautify_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def figure1_main_results() -> None:
    df = load_csv("paper_main_results_table.csv")
    models = df["Model"].tolist()
    auroc = df["AUROC"].astype(float).tolist()
    recall_values = percent_heights(df["Recall"])

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)

    palette = ["#2F6B9A", "#3E9B8D", "#D98E04", "#B84E8A", "#4C6A92"]

    axes[0].bar(models, auroc, color=palette[: len(models)], edgecolor="black", linewidth=0.6)
    axes[0].set_title("AUROC")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0.82, 0.91)
    axes[0].axhline(0.85, color="#C83E4D", linestyle="--", linewidth=1.4)
    axes[0].tick_params(axis="x", rotation=20)
    for bar, value in zip(axes[0].patches, auroc):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.002, 0.905),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    _beautify_axes(axes[0])

    recall_bars = axes[1].bar(
        models,
        recall_values.fillna(0.0),
        color=palette[: len(models)],
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, raw_value, model in zip(recall_bars, df["Recall"], models):
        if pd.isna(raw_value) or str(raw_value).strip().upper() == "N/A":
            bar.set_hatch("///")
            bar.set_facecolor("#E6E6E6")
            bar.set_edgecolor("#555555")
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                4.0,
                "†",
                ha="center",
                va="bottom",
                fontsize=11,
                rotation=0,
            )
        else:
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                min(bar.get_height() + 2.0, 98.0),
                f"{bar.get_height():.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    axes[1].set_title("Clinical Recall")
    axes[1].set_ylabel("Recall (%)")
    axes[1].set_ylim(0, 100)
    axes[1].axhline(80.0, color="#C83E4D", linestyle="--", linewidth=1.4)
    axes[1].tick_params(axis="x", rotation=20)
    _beautify_axes(axes[1])

    fig.suptitle("Main Results on MIMIC-IV", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "figure1_main_results")
    plt.close(fig)


def figure2_scalability() -> None:
    df = load_csv("exp9_scalability_analysis.csv")
    scaling = df[df["dropout_fraction"] == 0.0].sort_values("n_clients")
    dropout = df[df["dropout_fraction"] > 0.0].sort_values("dropout_fraction")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(scaling["n_clients"], scaling["auroc"], marker="o", color="#2F6B9A", label="AUROC")
    ax2.plot(
        scaling["n_clients"],
        scaling["throughput_samples_per_sec"],
        marker="s",
        color="#D98E04",
        label="Throughput",
    )
    ax.set_title("Scaling the Number of Clients")
    ax.set_xlabel("Clients")
    ax.set_ylabel("AUROC")
    ax2.set_ylabel("Throughput (samples/sec)")
    ax.set_ylim(0.86, 0.90)
    ax.tick_params(axis="x")
    ax.grid(True, axis="y", alpha=0.25)
    ax2.grid(False)

    ax = axes[1]
    dropout_labels = [f"{int(frac * 100)}%" for frac in dropout["dropout_fraction"]]
    x = range(len(dropout_labels))
    width = 0.38
    ax.bar([i - width / 2 for i in x], dropout["auroc"], width=width, color="#2F6B9A", label="AUROC")
    ax2 = ax.twinx()
    ax2.bar([i + width / 2 for i in x], dropout["throughput_samples_per_sec"], width=width, color="#D98E04", label="Throughput")
    ax.set_xticks(list(x))
    ax.set_xticklabels(dropout_labels)
    ax.set_title("Dropout Stress Test")
    ax.set_xlabel("Client dropout")
    ax.set_ylabel("AUROC")
    ax2.set_ylabel("Throughput (samples/sec)")
    ax.set_ylim(0.86, 0.90)
    ax.grid(True, axis="y", alpha=0.25)
    ax2.grid(False)

    fig.suptitle("Scalability and Failure Robustness", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "figure2_scalability")
    plt.close(fig)


def figure3_aggregation() -> None:
    df = load_csv("paper_main_results_table.csv")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    subset = df.iloc[:3].copy()
    axes[0].bar(subset["Model"], subset["AUROC"].astype(float), color=["#2F6B9A", "#3E9B8D", "#D98E04"], edgecolor="black", linewidth=0.6)
    axes[0].set_title("Aggregation Strategies")
    axes[0].set_ylabel("AUROC")
    axes[0].set_ylim(0.82, 0.91)
    axes[0].tick_params(axis="x", rotation=18)
    _beautify_axes(axes[0])

    recall_df = df[percent_series(df["Recall"]).notna()].copy()
    axes[1].bar(recall_df["Model"], percent_series(recall_df["Recall"]), color="#4C6A92", edgecolor="black", linewidth=0.6)
    axes[1].set_title("Clinical Recall")
    axes[1].set_ylabel("Recall (%)")
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis="x", rotation=18)
    _beautify_axes(axes[1])

    fig.suptitle("Aggregation Comparison", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "figure3_aggregation")
    plt.close(fig)


def figure4_feature_drift() -> None:
    aggregated = load_csv("exp5_feature_importance_aggregated.csv")
    drift = load_csv("exp5_high_drift_features.csv")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    top_mean = aggregated.head(10).iloc[::-1]
    axes[0].barh(top_mean.index, top_mean["mean"], color="#2F6B9A", edgecolor="black", linewidth=0.5)
    axes[0].set_title("Mean Importance Across Clients")
    axes[0].set_xlabel("Mean importance")
    _beautify_axes(axes[0])

    top_drift = drift.head(10).iloc[::-1]
    axes[1].barh(top_drift.index, top_drift["cv"], color="#D98E04", edgecolor="black", linewidth=0.5)
    axes[1].set_title("High-Drift Features")
    axes[1].set_xlabel("Coefficient of variation")
    _beautify_axes(axes[1])

    fig.suptitle("Federated Feature Drift", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "figure4_feature_drift")
    plt.close(fig)


def figure5_privacy_tradeoff() -> None:
    df = load_csv("exp7_differential_privacy.csv")
    df["condition"] = df["use_dp"].map({False: "Baseline", True: "DP Enabled"})

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True)
    metrics = ["auroc", "brier", "ece"]
    titles = ["AUROC", "Brier Score", "ECE"]
    colors = {"Baseline": "#2F6B9A", "DP Enabled": "#D98E04"}

    for ax, metric, title in zip(axes, metrics, titles):
        sns.boxplot(data=df, x="condition", y=metric, ax=ax, palette=colors, width=0.5, fliersize=0)
        sns.stripplot(data=df, x="condition", y=metric, ax=ax, color="black", size=4, alpha=0.75, jitter=0.12)
        ax.set_title(title)
        ax.set_xlabel("")
        _beautify_axes(ax)

    fig.suptitle("Privacy-Utility Tradeoff", y=1.03, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "figure5_privacy_tradeoff")
    plt.close(fig)


def figure6_byzantine() -> None:
    df = load_csv("exp8_adversarial_robustness.csv")
    df["scenario_label"] = df.apply(
        lambda row: "Clean" if float(row["malicious_fraction"]) == 0.0 else f'{int(float(row["malicious_fraction"]) * 100)}% Byzantine',
        axis=1,
    )

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True)
    metrics = ["auroc", "recall", "precision"]
    titles = ["AUROC", "Recall", "Precision"]
    colors = ["#2F6B9A", "#3E9B8D", "#D98E04"]

    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        ax.bar(df["scenario_label"], df[metric], color=color, edgecolor="black", linewidth=0.6)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=18)
        _beautify_axes(ax)

    fig.suptitle("Byzantine Robustness", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "figure6_byzantine")
    plt.close(fig)


def figure7_calibration() -> None:
    df = load_csv("exp7_differential_privacy_summary.csv")
    df = df.rename(columns={"experiment": "condition"})

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].scatter(df["brier"], df["auroc"], s=120, color="#2F6B9A")
    for _, row in df.iterrows():
        axes[0].annotate(row["condition"], (row["brier"], row["auroc"]), xytext=(6, 4), textcoords="offset points", fontsize=9)
    axes[0].set_xlabel("Brier Score")
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("Calibration vs Utility")
    _beautify_axes(axes[0])

    axes[1].scatter(df["ece"], df["auroc"], s=120, color="#D98E04")
    for _, row in df.iterrows():
        axes[1].annotate(row["condition"], (row["ece"], row["auroc"]), xytext=(6, 4), textcoords="offset points", fontsize=9)
    axes[1].set_xlabel("ECE")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("Expected Calibration Error")
    _beautify_axes(axes[1])

    fig.suptitle("Calibration View of Differential Privacy", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "figure7_calibration")
    plt.close(fig)


def figure8_statistical_validation() -> None:
    df = load_csv("exp6_statistical_validation_summary.csv")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True)
    metrics = ["auroc", "brier", "ece"]
    labels = ["AUROC", "Brier", "ECE"]

    for ax, metric, label in zip(axes, metrics, labels):
        mean_col = f"{metric}_mean"
        low_col = f"{metric}_ci_low"
        high_col = f"{metric}_ci_high"
        means = df[mean_col].astype(float)
        lower = means - df[low_col].astype(float)
        upper = df[high_col].astype(float) - means
        ax.errorbar(df["strategy"], means, yerr=[lower, upper], fmt="o", capsize=5, color="#2F6B9A")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=18)
        _beautify_axes(ax)

    fig.suptitle("Statistical Validation Across Seeds", y=1.02, fontsize=14, fontweight="bold")
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