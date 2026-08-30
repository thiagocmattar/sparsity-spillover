"""Plot peak learning rate against verified final validation loss."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator


RUN_DIR = Path(__file__).resolve().parent
SOURCE = RUN_DIR / "artifacts" / "verification.json"
OUTPUT = RUN_DIR / "figures" / "01-peak-lr-vs-final-val-loss.pdf"


def main() -> None:
    summary = json.loads(SOURCE.read_text(encoding="utf-8"))
    if summary.get("status") != "verified":
        raise ValueError("terminal verification must have status='verified'")

    conditions = sorted(
        summary["conditions"], key=lambda item: item["peak_learning_rate"]
    )
    if len(conditions) != 4:
        raise ValueError(f"expected four LR conditions, found {len(conditions)}")

    learning_rates = [item["peak_learning_rate"] for item in conditions]
    validation_losses = [item["final_validation_loss"] for item in conditions]
    tick_labels = ["5e-4", "1e-3", "2e-3", "4e-3"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.22)
    color = "#0072B2"
    ax.plot(
        learning_rates,
        validation_losses,
        color=color,
        linewidth=2.0,
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=2.0,
    )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(learning_rates))
    ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))
    ax.set_xlim(learning_rates[0] / (2**0.35), learning_rates[-1] * (2**0.35))
    ax.set_ylim(0, 7.0)
    ax.set_xlabel("Peak learning rate (log2 scale)", labelpad=8)
    ax.set_ylabel("Final validation loss")
    ax.set_title("Pythia-14M local learning-rate calibration", pad=12)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for learning_rate, loss in zip(learning_rates, validation_losses, strict=True):
        ax.annotate(
            f"{loss:.3f}",
            (learning_rate, loss),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#202020",
        )

    fig.text(
        0.5,
        0.035,
        "Seed 0; 449 updates per condition; all 338 complete validation blocks",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#404040",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
