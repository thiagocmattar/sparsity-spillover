#!/usr/bin/env python3
"""Plot fresh-process implementation gains at fixed checkpoints and R_model."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
SIZE_ORDER = ("14M", "70M", "410M")
SIZE_COLORS = {"14M": "#6F4C9B", "70M": "#56B4E9", "410M": "#CC79A7"}
TRANSITIONS = {
    "14M": "P0 → K001",
    "70M": "K009 → K016",
    "410M": "K004 → K010",
}


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.6,
        "ytick.labelsize": 8.6,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    })


def plot(pair_rows: list[dict], summaries: list[dict], output: Path) -> None:
    configure()
    values = [float(row[key]) for row in pair_rows for key in ("baseline_speedup", "optimized_speedup")]
    lower = min(0.985, min(values) - 0.025)
    upper = max(1.035, max(values) + 0.035)
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.65), sharey=True)
    for axis, size in zip(axes, SIZE_ORDER):
        color = SIZE_COLORS[size]
        size_pairs = [row for row in pair_rows if row["model_size"] == size]
        size_summaries = [row for row in summaries if row["model_size"] == size]
        families = ("A4-OL1", "A7-OL1")
        for x, family in enumerate(families):
            repeats = sorted(
                (row for row in size_pairs if row["family"] == family),
                key=lambda row: int(row["repeat"]),
            )
            summary = next(row for row in size_summaries if row["family"] == family)
            for row in repeats:
                before = float(row["baseline_speedup"])
                after = float(row["optimized_speedup"])
                axis.plot([x - 0.11, x + 0.11], [before, after], color=color,
                          alpha=0.28, linewidth=1.15, zorder=1)
                axis.scatter(x - 0.11, before, s=23, facecolor="white", edgecolor=color,
                             linewidth=0.9, alpha=0.72, zorder=2)
                axis.scatter(x + 0.11, after, s=23, facecolor=color, edgecolor="white",
                             linewidth=0.45, alpha=0.72, zorder=2)
            before_median = float(summary["baseline_median_speedup"])
            after_median = float(summary["optimized_median_speedup"])
            ratio = float(summary["median_optimized_over_baseline"])
            axis.plot([x - 0.11, x + 0.11], [before_median, after_median], color="#202020",
                      linewidth=2.2, zorder=3)
            axis.scatter(x - 0.11, before_median, s=60, facecolor="white", edgecolor="#202020",
                         linewidth=1.35, zorder=4)
            axis.scatter(x + 0.11, after_median, s=60, facecolor=color, edgecolor="#202020",
                         linewidth=1.05, zorder=4)
            annotation_y = max(before_median, after_median) + 0.012
            axis.text(x, annotation_y, f"{ratio:.3f}×", ha="center", va="bottom",
                      fontsize=8.3, color="#303030")
        axis.axhline(1.0, color="#777777", linewidth=0.9, linestyle=(0, (3, 2)), zorder=0)
        axis.set_xlim(-0.45, 1.45)
        axis.set_ylim(lower, upper)
        tick_labels = []
        for family in families:
            summary = next(row for row in size_summaries if row["family"] == family)
            tick_labels.append(
                f"{family.split('-', 1)[0]}, κ=.5\n$R_{{model}}$={float(summary['R_model_percent']):.1f}%"
            )
        axis.set_xticks((0, 1), tick_labels)
        axis.set_title(f"Pythia-{size}\n{TRANSITIONS[size]}", color=color, fontweight="semibold")
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.65)
    axes[0].set_ylabel("Full-model speedup vs. native (×)")
    legend = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor="#202020", markersize=6.5, label="Baseline"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#777777",
               markeredgecolor="#202020", markersize=6.5, label="Optimized"),
        Line2D([0], [0], color="#777777", linewidth=1.2, alpha=0.55,
               label="Same-repeat pair"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.055),
               ncol=3, frameon=False)
    fig.suptitle("Fresh-process kernel optimization at fixed checkpoint and canonical $R_{model}$",
                 y=1.115, fontsize=12.2, fontweight="semibold")
    fig.text(0.5, -0.005,
             "Thin links: three counterbalanced process pairs. Large markers: medians. All processes use complete validation.",
             ha="center", va="top", fontsize=8.1, color="#555555")
    fig.subplots_adjust(left=0.07, right=0.995, top=0.79, bottom=0.18, wspace=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    plot(
        rows(HERE / "fixed-rmodel-process-pairs.csv"),
        rows(HERE / "fixed-rmodel-replication.csv"),
        HERE / "figures/06-fixed-rmodel-kernel-optimization.pdf",
    )


if __name__ == "__main__":
    main()
