#!/usr/bin/env python3
"""Create the two paper-facing Analysis 016 figures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
SIZE_ORDER = ("14M", "70M", "410M")
SIZE_COLORS = {"14M": "#6F4C9B", "70M": "#56B4E9", "410M": "#CC79A7"}
GPU_ORDER = ("NVIDIA RTX PRO 4500 Blackwell", "NVIDIA H100 NVL")
GPU_LABELS = {
    "NVIDIA RTX PRO 4500 Blackwell": "RTX PRO 4500",
    "NVIDIA H100 NVL": "H100 NVL",
}
GPU_LINESTYLES = {
    "NVIDIA RTX PRO 4500 Blackwell": "-",
    "NVIDIA H100 NVL": (0, (4, 2.4)),
}
FAMILY_MARKERS = {"A0": "o", "A1-H": "s", "A4-OL1": "^", "A7-OL1": "D"}
OUTPUTS = (
    "01-rmodel-vs-speedup.pdf",
    "02-autoresearch-speedup-progress.pdf",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def truth(value: str) -> bool:
    return value.lower() == "true"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.labelsize": 10.2,
            "axes.titlesize": 11.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "pdf.compression": 9,
        }
    )


def style_axis(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.7, alpha=0.78)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#888888")
    axis.spines["bottom"].set_color("#888888")
    axis.tick_params(color="#777777")


def save_pdf(figure: plt.Figure, filename: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        FIGURES / filename,
        format="pdf",
        bbox_inches="tight",
        metadata={"Creator": "Analysis 016", "CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def render_rmodel_speedup() -> None:
    rows = read_csv(HERE / "rmodel-speedup.csv")
    fits = {
        (row["model_size"], row["gpu"]): row
        for row in read_csv(HERE / "rmodel-regressions.csv")
    }
    if len(rows) != 54 or len(fits) != 6:
        raise ValueError("Expected 54 checkpoint summaries and six primary fits")

    figure, axes = plt.subplots(1, 3, figsize=(12.25, 4.95), sharey=True)
    x_limits = {"14M": (-1.0, 29.5), "70M": (-1.5, 43.0), "410M": (-2.5, 84.5)}
    for axis, size in zip(axes, SIZE_ORDER):
        color = SIZE_COLORS[size]
        panel = [row for row in rows if row["model_size"] == size]
        span = x_limits[size][1] - x_limits[size][0]
        jitter = span * 0.0055
        for row in panel:
            gpu = row["gpu"]
            passed = truth(row["qualified"])
            x_true = float(row["measured_R_model_percent"])
            x = x_true + (-jitter if gpu == GPU_ORDER[0] else jitter)
            y = float(row["process_median_speedup"])
            low = float(row["process_min_speedup"])
            high = float(row["process_max_speedup"])
            if passed:
                face = color if gpu == GPU_ORDER[0] else "white"
                edge = color
                marker = FAMILY_MARKERS[row["family"]]
            else:
                face = "white"
                edge = "#B2182B"
                marker = "x"
            axis.errorbar(
                x,
                y,
                yerr=[[y - low], [high - y]],
                fmt=marker,
                markersize=6.7,
                markerfacecolor=face,
                markeredgecolor=edge,
                markeredgewidth=1.15,
                ecolor=edge,
                elinewidth=0.78,
                capsize=1.8,
                alpha=0.96,
                zorder=5,
            )

        fit_text = []
        for gpu in GPU_ORDER:
            fit = fits[(size, gpu)]
            qualified = [row for row in panel if row["gpu"] == gpu and truth(row["qualified"])]
            x_min = min(float(row["measured_R_model_percent"]) for row in qualified)
            x_max = max(float(row["measured_R_model_percent"]) for row in qualified)
            x_line = np.linspace(x_min, x_max, 200)
            y_line = float(fit["intercept"]) + float(fit["slope_per_R_model_fraction"]) * x_line / 100.0
            axis.plot(
                x_line,
                y_line,
                color=color,
                linestyle=GPU_LINESTYLES[gpu],
                linewidth=2.0,
                alpha=0.96 if gpu == GPU_ORDER[0] else 0.78,
                zorder=3,
            )
            fit_text.append(
                f"{GPU_LABELS[gpu]}: "
                rf"$R^2={float(fit['R_squared']):.3f}$, "
                rf"$n={fit['n_checkpoints']}$"
            )

        axis.text(
            0.035,
            0.955,
            "\n".join(fit_text),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.8,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
            zorder=10,
        )
        axis.axhline(1.0, color="#4D4D4D", linewidth=1.0, linestyle=(0, (3.5, 2.6)), zorder=1)
        axis.set_xlim(*x_limits[size])
        axis.set_ylim(0.735, 1.09)
        axis.set_title(f"Pythia-{size}", loc="left", color=color, fontweight="bold", pad=7)
        axis.set_xlabel(r"Measured $R_{\mathrm{model}}$ (%)")
        style_axis(axis)
    axes[0].set_ylabel("Full-model speedup (native / sparse)")

    gpu_handles = [
        Line2D(
            [0],
            [0],
            color="#4D4D4D",
            linestyle=GPU_LINESTYLES[gpu],
            marker="o",
            markerfacecolor="#4D4D4D" if gpu == GPU_ORDER[0] else "white",
            markeredgecolor="#4D4D4D",
            linewidth=1.8,
            markersize=6,
            label=GPU_LABELS[gpu],
        )
        for gpu in GPU_ORDER
    ]
    family_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            color="#666666",
            markerfacecolor="#666666",
            markersize=6,
            label=family,
        )
        for family, marker in FAMILY_MARKERS.items()
    ]
    failure_handle = Line2D(
        [0], [0], marker="x", linestyle="none", color="#B2182B", markersize=6.5, label="failed numerical gate"
    )
    figure.legend(
        handles=gpu_handles + family_handles + [failure_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=7,
        frameon=False,
        columnspacing=1.05,
        handletextpad=0.4,
    )
    figure.suptitle(
        r"Logical opportunity is not a universal speedup law",
        x=0.5,
        y=0.995,
        fontsize=13.0,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "One point per trained checkpoint; bars span fresh-process estimates (RTX: 3, H100: 2). Lines are qualified within-size OLS fits.\n"
        r"Small horizontal offsets separate GPUs only. $R_{\mathrm{model}}$ counts logical zero-product opportunities; it is not runtime savings.",
        ha="center",
        va="bottom",
        fontsize=7.55,
        color="#444444",
        linespacing=1.28,
    )
    figure.subplots_adjust(left=0.072, right=0.992, top=0.77, bottom=0.18, wspace=0.12)
    save_pdf(figure, OUTPUTS[0])


def render_progress() -> None:
    rows = read_csv(HERE / "candidate-progress.csv")
    scored = [row for row in rows if row["endpoint_geomean_speedup"]]
    if len(scored) != 12:
        raise ValueError(f"Expected 12 comparable candidate milestones, found {len(scored)}")

    figure, axes = plt.subplots(3, 1, figsize=(12.25, 7.75), sharex=True, sharey=True)
    notes = {
        "14M": "K012 is endpoint-fast but rejected: only 3/6 development conditions passed complete validation.",
        "70M": "No paired score for P0/K001/K003/K006 (A7 gate failures) or K007/K008 (one-endpoint searches).",
        "410M": "P0 and K001 failed correctness before timing; K004 and K010 produced paired endpoint scores.",
    }
    for axis, size in zip(axes, SIZE_ORDER):
        color = SIZE_COLORS[size]
        panel = sorted(
            [row for row in scored if row["model_size"] == size],
            key=lambda row: int(row["candidate_index"]),
        )
        xs = np.array([int(row["candidate_index"]) for row in panel], dtype=float)
        ys = np.array([float(row["endpoint_geomean_speedup"]) for row in panel])
        lows = np.array([min(float(row["a4_high_speedup"]), float(row["a7_high_speedup"])) for row in panel])
        highs = np.array([max(float(row["a4_high_speedup"]), float(row["a7_high_speedup"])) for row in panel])

        axis.plot(xs, ys, color=color, linewidth=1.65, alpha=0.82, zorder=3)
        for x, y, low, high, row in zip(xs, ys, lows, highs, panel):
            axis.vlines(x, low, high, color=color, linewidth=2.0, alpha=0.4, zorder=2)
            axis.scatter([x, x], [low, high], marker="_", s=70, color=color, linewidth=1.2, zorder=3)
            if row["status"] == "rejected_full_matrix":
                axis.scatter(
                    x,
                    y,
                    marker="D",
                    s=68,
                    facecolor="white",
                    edgecolor="#B2182B",
                    linewidth=1.5,
                    zorder=7,
                )
            elif row["status"] == "frozen_final":
                axis.scatter(
                    x,
                    y,
                    marker="*",
                    s=155,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=7,
                )
                axis.annotate(
                    f"{y:.3f}x",
                    (x, y),
                    xytext=(7, 7),
                    textcoords="offset points",
                    fontsize=8.0,
                    color="#333333",
                    fontweight="bold",
                )
            else:
                axis.scatter(
                    x,
                    y,
                    marker="o",
                    s=50,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.65,
                    zorder=6,
                )

        qualified = [row["status"] != "rejected_full_matrix" for row in panel]
        running: list[float] = []
        best = -np.inf
        for y, is_qualified in zip(ys, qualified):
            if is_qualified:
                best = max(best, y)
            running.append(best)
        axis.step(xs, running, where="post", color=color, linestyle=(0, (3, 2)), linewidth=1.1, alpha=0.48, zorder=1)
        axis.axhline(1.0, color="#4D4D4D", linewidth=1.0, linestyle=(0, (3.5, 2.6)), zorder=1)
        axis.set_ylim(0.55, 1.18)
        axis.set_ylabel("Full-model speedup\n(endpoint geomean)")
        axis.set_title(f"Pythia-{size}", loc="left", color=color, fontweight="bold", pad=6)
        note_y = 0.13 if size == "14M" else 0.93
        axis.text(
            0.995,
            note_y,
            notes[size],
            transform=axis.transAxes,
            ha="right",
            va="bottom" if size == "14M" else "top",
            fontsize=7.7,
            color="#444444",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
        )
        best_index = int(np.argmax(running))
        final_index = next(
            (index for index, row in enumerate(panel) if row["status"] == "frozen_final"),
            None,
        )
        if best_index != final_index:
            axis.annotate(
                f"best {ys[best_index]:.3f}x",
                (xs[best_index], ys[best_index]),
                xytext=(7, 8),
                textcoords="offset points",
                fontsize=7.8,
                color="#333333",
                fontweight="bold",
            )
        style_axis(axis)

    axes[-1].set_xlim(-0.6, 16.6)
    axes[-1].set_xticks(range(17), ["P0"] + [f"K{i:03d}" for i in range(1, 17)])
    axes[-1].set_xlabel("Autoresearch candidate milestone")

    handles = [
        Line2D([0], [0], color="#666666", marker="o", markerfacecolor="#666666", markeredgecolor="white", label="paired endpoint geomean"),
        Line2D([0], [0], color="#666666", linewidth=2.2, marker="_", markersize=9, label="A4/A7 endpoint span"),
        Line2D([0], [0], color="#666666", linestyle=(0, (3, 2)), label="best qualified so far"),
        Line2D([0], [0], color="#666666", marker="*", markerfacecolor="#666666", markeredgecolor="white", linestyle="none", markersize=10, label="frozen final"),
        Line2D([0], [0], color="#B2182B", marker="D", markerfacecolor="white", linestyle="none", markersize=6.5, label="later rejected policy"),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.907),
        ncol=5,
        frameon=False,
        columnspacing=1.15,
        handletextpad=0.45,
    )
    figure.suptitle(
        "Autoresearch progress was architecture-specific and non-monotone",
        x=0.5,
        y=0.995,
        fontsize=13.0,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        r"Common score: geometric mean of A4-OL1 $\kappa=0.5$ and A7-OL1 $\kappa=0.5$ batch-one full-model speedups on RTX PRO 4500. "
        "Vertical spans show the two endpoints.\n"
        "Only milestones with both timings are connected; sample counts vary from 48 to 160 per endpoint. This is one adaptive search trajectory, not optimizer benchmarking.",
        ha="center",
        va="bottom",
        fontsize=7.55,
        color="#444444",
        linespacing=1.28,
    )
    figure.subplots_adjust(left=0.075, right=0.992, top=0.82, bottom=0.13, hspace=0.24)
    save_pdf(figure, OUTPUTS[1])


def main() -> None:
    configure_style()
    render_rmodel_speedup()
    render_progress()
    print(f"Wrote {', '.join(str(FIGURES / name) for name in OUTPUTS)}")


if __name__ == "__main__":
    main()
