from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "plots" / "paper_main_results_table.csv"
PDF_PATH = ROOT / "results" / "plots" / "paper_robustness_byzantine.pdf"
PNG_PATH = ROOT / "results" / "plots" / "paper_robustness_byzantine.png"


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    models = df["Model"].tolist()
    auroc = df["AUROC"].astype(float).tolist()

    colors = ["#4C78A8", "#72B7B2", "#F58518", "#54A24B", "#E45756"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    bars = ax.bar(models, auroc, color=colors[: len(models)], width=0.7, edgecolor="black", linewidth=0.6)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title("Figure 2. AUROC comparison across main configurations")
    ax.tick_params(axis="x", rotation=18)

    for bar, value in zip(bars, auroc):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.02, 0.98),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(PDF_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
