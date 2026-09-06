#!/usr/bin/env python3
"""Render fresh-process, hardware-transfer, and component figures."""

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
SUMMARY = HERE / "replication-summary.csv"
REGRESSIONS = HERE / "replication-regressions.csv"
TRANSFER = HERE / "hardware-transfer.csv"
COMPONENTS = HERE / "component-results.csv"

RTX = "rtxpro4500-003"
H100 = "h100nvl-002"
HARDWARE_ORDER = (RTX, H100)
HARDWARE_LABELS = {
    RTX: "RTX PRO 4500 Blackwell",
    H100: "H100 NVL Hopper",
}
SIZE_ORDER = ("14M", "70M", "410M")
SIZE_COLORS = {"14M": "#6F4C9B", "70M": "#56B4E9", "410M": "#CC79A7"}
FAMILY_ORDER = ("A0", "A1-H", "A4-OL1", "A7-OL1")
FAMILY_MARKERS = {"A0": "P", "A1-H": "^", "A4-OL1": "h", "A7-OL1": "p"}

CONDITION_STYLES = {
    ("A0", ""): ("#4D4D4D", "P", "A0"),
    ("A1-H", ""): ("#E69F00", "^", "A1-H"),
    ("A4-OL1", "0"): ("#56B4E9", "h", r"A4-OL1, $\kappa=0$"),
    ("A4-OL1", "0.5"): ("#0072B2", "H", r"A4-OL1, $\kappa=0.5$"),
    ("A7-OL1", "0"): ("#CC79A7", "p", r"A7-OL1, $\kappa=0$"),
    ("A7-OL1", "0.5"): ("#6F4C9B", "X", r"A7-OL1, $\kappa=0.5$"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def truth(value: str) -> bool:
    return value.lower() == "true"


def kappa_key(row: dict[str, str]) -> str:
    if not row["kappa"]:
        return ""
    value = float(row["kappa"])
    return "0" if value == 0 else f"{value:g}"


def condition_style(row: dict[str, str]) -> tuple[str, str, str]:
    return CONDITION_STYLES[(row["family"], kappa_key(row))]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.labelsize": 10.2,
            "axes.titlesize": 10.7,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "pdf.fonttype": 42,
            "pdf.compression": 9,
        }
    )


def style_axis(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.72, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#888888")
    axis.spines["bottom"].set_color("#888888")
    axis.tick_params(color="#777777")


def save_pdf(figure: plt.Figure, name: str) -> None:
    figure.savefig(
        FIGURES / name,
        format="pdf",
        bbox_inches="tight",
        metadata={"Creator": "Analysis 015", "CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def render_fresh_process_association() -> None:
    rows = read_csv(SUMMARY)
    fits = {
        (row["hardware_attempt"], row["model_size"]): row
        for row in read_csv(REGRESSIONS)
        if row["stratum"] == "primary-qualified"
    }
    if len(rows) != 54 or len(fits) != 6:
        raise ValueError("Expected 54 summaries and six primary fits")

    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.65), sharex=True, sharey=True)
    for axis, hardware in zip(axes, HARDWARE_ORDER):
        panel_rows = [row for row in rows if row["hardware_attempt"] == hardware]
        for row in panel_rows:
            x = float(row["measured_R_model_percent"])
            y = float(row["process_median_speedup"])
            low = float(row["process_min_speedup"])
            high = float(row["process_max_speedup"])
            passed = truth(row["qualified"])
            color = SIZE_COLORS[row["model_size"]]
            axis.errorbar(
                x,
                y,
                yerr=[[y - low], [high - y]],
                fmt=FAMILY_MARKERS[row["family"]],
                markersize=7.4,
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
                axis.scatter([x], [y], marker="x", s=31, color="#B2182B", linewidth=1.15, zorder=7)

        fit_labels = []
        for size in SIZE_ORDER:
            fit = fits[(hardware, size)]
            fit_rows = [
                row for row in panel_rows if row["model_size"] == size and truth(row["qualified"])
            ]
            x_min = min(float(row["measured_R_model_percent"]) for row in fit_rows)
            x_max = max(float(row["measured_R_model_percent"]) for row in fit_rows)
            x_line = np.linspace(x_min, x_max, 200)
            y_line = float(fit["intercept"]) + float(fit["slope_per_R_model_fraction"]) * x_line / 100.0
            axis.plot(x_line, y_line, color=SIZE_COLORS[size], linewidth=2.0, alpha=0.92, zorder=3)
            fit_labels.append(
                Line2D(
                    [0],
                    [0],
                    color=SIZE_COLORS[size],
                    linewidth=2.1,
                    label=(
                        rf"{size}: $R^2={float(fit['R_squared']):.3f}$, "
                        rf"$\rho={float(fit['spearman_rho']):+.2f}$, "
                        f"n={fit['n_checkpoints']}"
                    ),
                )
            )

        axis.axhline(1.0, color="#555555", linewidth=1.05, linestyle=(0, (4, 3)), zorder=1)
        axis.set_title(HARDWARE_LABELS[hardware], loc="left", fontweight="bold", pad=8)
        axis.legend(handles=fit_labels, loc="lower right", frameon=False, handlelength=2.2, labelspacing=0.52)
        axis.set_xlim(-2.0, 84.0)
        axis.set_ylim(0.72, 1.155)
        axis.set_xlabel(r"Measured $R_{\mathrm{model}}$ (%)")
        style_axis(axis)
    axes[0].set_ylabel("Full-model speedup (native / candidate)")

    family_handles = [
        Line2D(
            [0],
            [0],
            marker=FAMILY_MARKERS[family],
            linestyle="none",
            markerfacecolor="#777777",
            markeredgecolor="white",
            markersize=7.2,
            label=family,
        )
        for family in FAMILY_ORDER
    ]
    family_handles.append(
        Line2D([0], [0], marker="x", linestyle="none", color="#B2182B", markersize=7, label="failed numerical gate")
    )
    figure.legend(
        handles=family_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=5,
        frameon=False,
        columnspacing=1.15,
        handletextpad=0.4,
    )
    figure.suptitle(
        r"The $R_{\mathrm{model}}$-speed association depends on model shape and GPU",
        x=0.5,
        y=0.985,
        fontsize=13.2,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.020,
        "Each point is one trained checkpoint (one seed); bars span independent process estimates. RTX uses 3 eager-only processes; H100 uses 2.\n"
        r"Lines are qualified within-size OLS fits. $R_{\mathrm{model}}$ is a logical-product opportunity, not measured runtime savings.",
        ha="center",
        va="bottom",
        fontsize=7.7,
        color="#444444",
        linespacing=1.3,
    )
    figure.subplots_adjust(left=0.075, right=0.988, top=0.81, bottom=0.16, wspace=0.12)
    save_pdf(figure, "03-fresh-process-rmodel-vs-speedup.pdf")


def render_hardware_transfer() -> None:
    rows = read_csv(TRANSFER)
    if len(rows) != 18 or not all(truth(row["both_qualified"]) for row in rows):
        raise ValueError("Expected 18 qualified matched hardware transfers")

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 5.35), sharey=True)
    for axis, size in zip(axes, SIZE_ORDER):
        panel = [row for row in rows if row["model_size"] == size]
        panel.sort(key=lambda row: list(CONDITION_STYLES).index((row["family"], kappa_key(row))))
        for row in panel:
            color, marker, _ = condition_style(row)
            y0 = float(row["rtx_process_median_speedup"])
            y1 = float(row["h100_process_median_speedup"])
            rtx_low = float(row["rtx_process_min_speedup"])
            rtx_high = float(row["rtx_process_max_speedup"])
            h100_low = float(row["h100_process_min_speedup"])
            h100_high = float(row["h100_process_max_speedup"])
            axis.plot([0, 1], [y0, y1], color=color, linewidth=1.5, alpha=0.76, zorder=2)
            axis.errorbar(
                [0, 1],
                [y0, y1],
                yerr=[[y0 - rtx_low, y1 - h100_low], [rtx_high - y0, h100_high - y1]],
                fmt=marker,
                markersize=7.2,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.65,
                ecolor=color,
                elinewidth=0.85,
                capsize=2.0,
                zorder=4,
            )
        axis.axhline(1.0, color="#555555", linewidth=1.05, linestyle=(0, (4, 3)), zorder=1)
        axis.set_xlim(-0.25, 1.25)
        axis.set_ylim(0.735, 1.095)
        axis.set_xticks([0, 1], ["RTX PRO 4500", "H100 NVL"])
        axis.set_title(f"Pythia-{size}", loc="left", fontweight="bold", pad=8)
        style_axis(axis)
    axes[0].set_ylabel("Full-model speedup (native / candidate)")

    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linewidth=1.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=6.8,
            label=label,
        )
        for color, marker, label in CONDITION_STYLES.values()
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.45,
    )
    figure.suptitle(
        "The same sparse policy changes value across GPU architectures",
        x=0.5,
        y=0.985,
        fontsize=13.2,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.020,
        "Lines join the same checkpoint, frozen implementation, batch-one T=2048 BF16 workload, and canonical logical counts.\n"
        "Speedup is relative to each GPU's own native eager baseline; bars span fresh-process estimates, not training seeds.",
        ha="center",
        va="bottom",
        fontsize=7.7,
        color="#444444",
        linespacing=1.3,
    )
    figure.subplots_adjust(left=0.075, right=0.988, top=0.78, bottom=0.16, wspace=0.13)
    save_pdf(figure, "04-matched-hardware-transfer.pdf")


def render_component_contributions() -> None:
    rows = read_csv(COMPONENTS)
    if len(rows) != 40:
        raise ValueError("Expected 40 component measurements")
    by_panel: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_panel[(row["hardware_attempt"], row["model_size"])].append(row)

    figure, axes = plt.subplots(2, 2, figsize=(10.9, 7.4), sharex=True, sharey=True)
    for row_index, hardware in enumerate(HARDWARE_ORDER):
        for column_index, size in enumerate(("14M", "70M")):
            axis = axes[row_index, column_index]
            panel = by_panel[(hardware, size)]
            by_condition: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in panel:
                by_condition[row["condition_id"]].append(row)
            for condition_rows in by_condition.values():
                base = condition_rows[0]
                color, marker, _ = condition_style(base)
                full = float(base["full_policy_process_median_speedup"])
                full_low = float(base["full_policy_process_min_speedup"])
                full_high = float(base["full_policy_process_max_speedup"])
                for component_row in condition_rows:
                    x = 0 if component_row["component"] == "ffn" else 1
                    component_speed = float(component_row["component_speedup"])
                    component_low = float(component_row["component_speedup_input_cluster_ci95_low"])
                    component_high = float(component_row["component_speedup_input_cluster_ci95_high"])
                    axis.plot([x, 2], [component_speed, full], color=color, linewidth=1.15, alpha=0.35, zorder=2)
                    axis.errorbar(
                        x,
                        component_speed,
                        yerr=[[component_speed - component_low], [component_high - component_speed]],
                        fmt=marker,
                        markersize=6.8,
                        markerfacecolor=color,
                        markeredgecolor="white",
                        markeredgewidth=0.55,
                        ecolor=color,
                        elinewidth=0.72,
                        capsize=1.8,
                        zorder=4,
                    )
                axis.errorbar(
                    2,
                    full,
                    yerr=[[full - full_low], [full_high - full]],
                    fmt=marker,
                    markersize=7.4,
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=0.65,
                    ecolor=color,
                    elinewidth=0.78,
                    capsize=2.0,
                    zorder=5,
                )
            axis.axhline(1.0, color="#555555", linewidth=1.05, linestyle=(0, (4, 3)), zorder=1)
            axis.set_xlim(-0.35, 2.35)
            axis.set_ylim(0.735, 1.095)
            axis.set_xticks([0, 1, 2], ["FFN only", "Attn projection only", "Full policy"])
            axis.set_title(f"{HARDWARE_LABELS[hardware]} - Pythia-{size}", loc="left", fontweight="bold", pad=7)
            style_axis(axis)
    axes[0, 0].set_ylabel("Full-model speedup")
    axes[1, 0].set_ylabel("Full-model speedup")

    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linewidth=1.2,
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=6.7,
            label=label,
        )
        for color, marker, label in CONDITION_STYLES.values()
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.902),
        ncol=3,
        frameon=False,
        columnspacing=1.15,
        handletextpad=0.42,
    )
    figure.suptitle(
        "Projection and FFN eligibility contribute differently to end-to-end speed",
        x=0.5,
        y=0.988,
        fontsize=13.0,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.016,
        "Component points are single-process full-model timings (bars: 95% input-cluster interval); full-policy points are process medians (bars: process range).\n"
        "Lines aid matching and do not imply additive effects. QK/PV remain dense SDPA. K010 at 410M is already z-projection-only and is omitted.",
        ha="center",
        va="bottom",
        fontsize=7.55,
        color="#444444",
        linespacing=1.3,
    )
    figure.subplots_adjust(left=0.08, right=0.988, top=0.81, bottom=0.12, hspace=0.28, wspace=0.11)
    save_pdf(figure, "05-component-contributions.pdf")


def main() -> None:
    configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    render_fresh_process_association()
    render_hardware_transfer()
    render_component_contributions()
    print("PASS: wrote three fresh-process Analysis 015 PDFs")


if __name__ == "__main__":
    main()
