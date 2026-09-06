#!/usr/bin/env python3
"""Render the two paper-facing figures for Analysis 015."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
FINAL = HERE / "final-results.csv"
REGRESSIONS = HERE / "regressions.csv"
TRANSITIONS = HERE / "same-rmodel-optimization.csv"

SIZE_ORDER = ("14M", "70M", "410M")
SIZE_COLORS = {"14M": "#6F4C9B", "70M": "#56B4E9", "410M": "#CC79A7"}
FAMILY_ORDER = ("A0", "A1-H", "A4-OL1", "A7-OL1")
FAMILY_MARKERS = {"A0": "P", "A1-H": "^", "A4-OL1": "h", "A7-OL1": "p"}


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
            "axes.labelsize": 10.3,
            "axes.titlesize": 10.8,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "pdf.fonttype": 42,
            "pdf.compression": 9,
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.75, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#888888")
    axis.spines["bottom"].set_color("#888888")
    axis.tick_params(color="#777777")


def render_final_association() -> None:
    rows = read_csv(FINAL)
    fits = {
        row["model_size"]: row
        for row in read_csv(REGRESSIONS)
        if row["stratum"] == "qualified-final"
    }
    if len(rows) != 36 or set(fits) != set(SIZE_ORDER):
        raise ValueError("Expected 36 final results and one qualified fit per model size")

    figure, axis = plt.subplots(figsize=(10.2, 6.15))
    for row in rows:
        size = row["model_size"]
        family = row["family"]
        x = float(row["R_model_percent"])
        y = float(row["paired_geomean_speedup"])
        low = float(row["speedup_input_cluster_ci95_low"])
        high = float(row["speedup_input_cluster_ci95_high"])
        passed = truth(row["correctness_pass"])
        color = SIZE_COLORS[size]
        axis.errorbar(
            x,
            y,
            yerr=[[y - low], [high - y]],
            fmt=FAMILY_MARKERS[family],
            markersize=8.0,
            markerfacecolor=color if passed else "white",
            markeredgecolor=color if passed else "#B2182B",
            markeredgewidth=1.0 if passed else 1.5,
            ecolor=color,
            elinewidth=0.85,
            capsize=2.0,
            alpha=0.96,
            zorder=5,
        )
        if not passed:
            axis.scatter(
                [x], [y], marker="x", s=33, color="#B2182B", linewidth=1.2, zorder=7
            )

    size_handles = []
    for size in SIZE_ORDER:
        fit = fits[size]
        size_rows = [row for row in rows if row["model_size"] == size and truth(row["correctness_pass"])]
        x_min = min(float(row["R_model_percent"]) for row in size_rows)
        x_max = max(float(row["R_model_percent"]) for row in size_rows)
        x = np.linspace(x_min, x_max, 200)
        y = float(fit["intercept"]) + float(fit["slope_per_R_model_fraction"]) * x / 100.0
        axis.plot(x, y, color=SIZE_COLORS[size], linewidth=2.1, alpha=0.94, zorder=3)
        size_handles.append(
            Line2D(
                [0], [0], color=SIZE_COLORS[size], linewidth=2.2,
                label=rf"Pythia-{size}: $R^2={float(fit['R_squared']):.3f}$, "
                      rf"$\rho={float(fit['spearman_rho']):+.2f}$",
            )
        )

    axis.axhline(1.0, color="#555555", linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
    axis.text(82.5, 1.0045, "native break-even", ha="right", va="bottom", color="#555555", fontsize=8.2)
    axis.set_xlim(-2.0, 84.0)
    axis.set_ylim(0.745, 1.105)
    axis.set_xlabel(r"Measured $R_{\mathrm{model}}$ (%)")
    axis.set_ylabel("Full-model paired speedup (native / candidate)")
    style_axis(axis)

    first_legend = axis.legend(
        handles=size_handles,
        loc="lower left",
        frameon=False,
        handlelength=2.5,
        labelspacing=0.65,
    )
    axis.add_artist(first_legend)
    family_handles = [
        Line2D(
            [0], [0], marker=FAMILY_MARKERS[family], linestyle="none",
            markerfacecolor="#777777", markeredgecolor="white", markeredgewidth=0.8,
            markersize=7.5, label=family,
        )
        for family in FAMILY_ORDER
    ]
    family_handles.append(
        Line2D(
            [0], [0], marker="x", linestyle="none", color="#B2182B",
            markersize=7.0, label="failed numerical gate",
        )
    )
    axis.legend(
        handles=family_handles,
        loc="lower right",
        frameon=False,
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.45,
        labelspacing=0.6,
    )

    figure.suptitle(
        r"The $R_{\mathrm{model}}$–speed association is model- and implementation-specific",
        x=0.5, y=0.975, fontsize=13.2, fontweight="bold",
    )
    figure.text(
        0.5,
        0.938,
        "Frozen kernels on one RTX PRO 4500 Blackwell; regression uses only numerically qualified deployments",
        ha="center", va="center", fontsize=9.2, color="#444444",
    )
    figure.text(
        0.5,
        0.018,
        "Each point is one trained checkpoint (one seed). Error bars are 95% input-cluster bootstrap intervals over 16 fixed blocks;\n"
        r"80 paired timings per point. $R_{\mathrm{model}}$ is a logical-product opportunity, not a runtime metric. OLS lines are descriptive.",
        ha="center", va="bottom", fontsize=7.7, color="#444444", linespacing=1.3,
    )
    figure.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.165)
    output = FIGURES / "01-rmodel-vs-full-model-speedup.pdf"
    figure.savefig(
        output,
        format="pdf",
        bbox_inches="tight",
        metadata={"Creator": "Analysis 015", "CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def transition_label(row: dict[str, str]) -> str:
    family = row["family"]
    if family in {"A0", "A1-H"}:
        return family
    return f"{family}, κ={float(row['kappa']):g}"


def render_same_rmodel_transitions() -> None:
    rows = read_csv(TRANSITIONS)
    group_order = (
        "14M exploratory P0-to-K001",
        "14M final-policy repair K012-to-K013",
        "70M complete-validation K009-to-K016",
        "410M development K004-to-K010",
    )
    panel_specs = {
        group_order[0]: ("14M: Sakana-derived start → fused K001", "P0", "K001", "development"),
        group_order[1]: ("14M: fast invalid policy → robust K013", "K012", "K013", "complete validation"),
        group_order[2]: ("70M: generic K009 → topology-specific K016", "K009", "K016", "complete validation"),
        group_order[3]: ("410M: early K004 → frozen K010", "K004", "K010", "development"),
    }
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_group[row["comparison_group"]].append(row)
    if set(by_group) != set(group_order):
        raise ValueError("Unexpected same-R_model comparison groups")

    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.45))
    axes_flat = axes.ravel()
    for axis, group in zip(axes_flat, group_order):
        title, left_label, right_label, scope = panel_specs[group]
        panel_rows = sorted(
            by_group[group],
            key=lambda row: (
                FAMILY_ORDER.index(row["family"]),
                -1.0 if row["kappa"] == "" else float(row["kappa"]),
            ),
        )
        for row in panel_rows:
            y0 = float(row["baseline_speedup"])
            y1 = float(row["optimized_speedup"])
            family = row["family"]
            color = SIZE_COLORS[row["model_size"]]
            both = truth(row["both_correct"])
            baseline_pass = truth(row["baseline_correctness_pass"])
            optimized_pass = truth(row["optimized_correctness_pass"])
            axis.plot(
                [0, 1], [y0, y1], color=color if both else "#8C8C8C",
                linewidth=1.4, alpha=0.72, zorder=2,
            )
            for x, y, passed in ((0, y0, baseline_pass), (1, y1, optimized_pass)):
                axis.scatter(
                    [x], [y], marker=FAMILY_MARKERS[family], s=58,
                    facecolor=color if passed else "white",
                    edgecolor=color if passed else "#B2182B",
                    linewidth=1.0 if passed else 1.5, zorder=4,
                )
                if not passed:
                    axis.scatter([x], [y], marker="x", s=26, color="#B2182B", linewidth=1.1, zorder=5)

        all_y = [float(row[key]) for row in panel_rows for key in ("baseline_speedup", "optimized_speedup")]
        margin = max(0.025, 0.12 * (max(all_y) - min(all_y)))
        axis.set_ylim(min(all_y) - margin, max(all_y) + margin)
        axis.set_xlim(-0.25, 1.25)
        axis.set_xticks([0, 1], [left_label, right_label])
        axis.axhline(1.0, color="#555555", linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        axis.set_title(title, loc="left", fontweight="bold", pad=8)
        axis.text(
            0.98, 0.04, scope, transform=axis.transAxes,
            ha="right", va="bottom", fontsize=7.8, color="#555555",
        )
        style_axis(axis)

    axes[0, 0].set_ylabel("Full-model paired speedup")
    axes[1, 0].set_ylabel("Full-model paired speedup")
    family_handles = [
        Line2D(
            [0], [0], marker=FAMILY_MARKERS[family], linestyle="none",
            markerfacecolor="#777777", markeredgecolor="white", markersize=7.3,
            label=family,
        )
        for family in FAMILY_ORDER
    ]
    family_handles.append(
        Line2D([0], [0], marker="x", linestyle="none", color="#B2182B", markersize=7, label="failed gate")
    )
    figure.legend(
        handles=family_handles, loc="upper center", bbox_to_anchor=(0.5, 0.904),
        ncol=5, frameon=False, columnspacing=1.15, handletextpad=0.4,
    )
    figure.suptitle(
        r"At fixed $R_{\mathrm{model}}$, implementation search changes both speed and validity",
        x=0.5, y=0.985, fontsize=13.2, fontweight="bold",
    )
    figure.text(
        0.5,
        0.016,
        "Lines join the same checkpoint, inputs and logical opportunity under two kernel policies. Ratios are each paired to the native eager path;\n"
        "cross-session transitions remain descriptive. Dense fallback is a valid deployment outcome, but not a sparse-kernel speedup.",
        ha="center", va="bottom", fontsize=7.7, color="#444444", linespacing=1.3,
    )
    figure.subplots_adjust(left=0.085, right=0.985, top=0.82, bottom=0.11, hspace=0.34, wspace=0.22)
    output = FIGURES / "02-same-rmodel-kernel-search-transitions.pdf"
    figure.savefig(
        output,
        format="pdf",
        bbox_inches="tight",
        metadata={"Creator": "Analysis 015", "CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def main() -> None:
    configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    render_final_association()
    render_same_rmodel_transitions()
    print("PASS: wrote two Analysis 015 PDF figures")


if __name__ == "__main__":
    main()
