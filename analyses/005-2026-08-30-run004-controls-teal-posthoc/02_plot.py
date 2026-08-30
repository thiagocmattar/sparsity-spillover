"""Reduce Analysis 005 and plot the TEAL-style quality-opportunity frontier."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


ANALYSIS_DIR = Path(__file__).resolve().parent
SOURCE = ANALYSIS_DIR / "teal_frontier.json"
TABLE = ANALYSIS_DIR / "tables.md"
OUTPUT = ANALYSIS_DIR / "figures" / "01-r-model-vs-final-validation-loss.pdf"
CONDITIONS = ("gelu-control", "relu-control")
TARGETS = tuple(index / 10 for index in range(10))
STYLES = {
    "gelu-control": {
        "label": "GeLU control + TEAL",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    "relu-control": {
        "label": "ReLU control + TEAL",
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
    },
}


def _load() -> dict[str, Any]:
    result = json.loads(SOURCE.read_text(encoding="utf-8"))
    if result.get("schema_version") != 1 or result.get("status") != "complete_verified":
        raise ValueError("Analysis 005 source is not complete verified evidence.")
    rows = result.get("conditions", [])
    if len(rows) != 20:
        raise ValueError("Analysis 005 must contain exactly twenty evaluation points.")
    for condition_id in CONDITIONS:
        selected = sorted(
            (row for row in rows if row["condition_id"] == condition_id),
            key=lambda row: float(row["target_sparsity"]),
        )
        if tuple(float(row["target_sparsity"]) for row in selected) != TARGETS:
            raise ValueError(f"Incomplete target-sparsity grid for {condition_id}.")
        baseline = float(selected[0]["validation"]["loss"])
        for row in selected:
            coverage = row["validation"]
            if (
                coverage["sequences"] != 338
                or coverage["input_tokens"] != 692_224
                or coverage["excluded_tail_tokens"] != 1_444
                or coverage["complete_block_coverage"] is not True
            ):
                raise ValueError("A plotted point lacks complete validation coverage.")
            logical = row["logical_products"]
            operations = logical["per_operation"].values()
            if sum(int(item["zero_product_count"]) for item in operations) != int(
                logical["block_zero_product_count"]
            ):
                raise ValueError("A plotted logical numerator does not reconcile.")
            expected_r = int(logical["block_zero_product_count"]) / int(
                logical["model_product_count"]
            )
            if not math.isclose(expected_r, float(logical["R_model"]), rel_tol=0.0, abs_tol=1e-16):
                raise ValueError("A plotted R_model value is not count-derived.")
            expected_delta = float(coverage["loss"]) - baseline
            if not math.isclose(
                expected_delta,
                float(row["loss_delta_from_zero_threshold"]),
                rel_tol=0.0,
                abs_tol=1e-14,
            ):
                raise ValueError("A plotted loss delta is not paired to target zero.")
    return result


def _nondominated(rows: list[Mapping[str, Any]]) -> list[bool]:
    flags = []
    for index, row in enumerate(rows):
        loss = float(row["validation"]["loss"])
        opportunity = float(row["logical_products"]["R_model"])
        dominated = any(
            other_index != index
            and float(other["validation"]["loss"]) <= loss
            and float(other["logical_products"]["R_model"]) >= opportunity
            and (
                float(other["validation"]["loss"]) < loss
                or float(other["logical_products"]["R_model"]) > opportunity
            )
            for other_index, other in enumerate(rows)
        )
        flags.append(not dominated)
    return flags


def _write_table(result: Mapping[str, Any], global_flags: list[bool]) -> None:
    rows = result["conditions"]
    lines = [
        "# Analysis 005 tables",
        "",
        "Every fraction below is recalculated from stored integer counts. Loss deltas are paired",
        "to the target-zero point from the same control sweep. `Global frontier` is nondominance",
        "over all twenty GeLU and ReLU points under lower loss and higher measured `R_model`.",
        "",
        "| control | target sparsity | validation loss | loss delta | R_model (%) | a zero (%) | m zero (%) | h zero (%) | z zero (%) | global frontier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row, flag in zip(rows, global_flags, strict=True):
        sites = {item["site"]: item for item in row["activations_by_site"]}
        lines.append(
            "| {activation} | {target:.1f} | {loss:.6f} | {delta:+.6f} | {rmodel:.6f} | "
            "{a:.3f} | {m:.3f} | {h:.3f} | {z:.3f} | {frontier} |".format(
                activation="ReLU" if row["activation"] == "relu" else "GeLU",
                target=float(row["target_sparsity"]),
                loss=float(row["validation"]["loss"]),
                delta=float(row["loss_delta_from_zero_threshold"]),
                rmodel=100.0 * float(row["logical_products"]["R_model"]),
                a=100.0 * float(sites["a"]["exact_zero_fraction"]),
                m=100.0 * float(sites["m"]["exact_zero_fraction"]),
                h=100.0 * float(sites["h"]["exact_zero_fraction"]),
                z=100.0 * float(sites["z"]["exact_zero_fraction"]),
                frontier="yes" if flag else "no",
            )
        )
    lines.extend(
        [
            "",
            "Coverage for every row: 500 validation documents, 338 complete 2,048-token blocks,",
            "692,224 input tokens, and a reported 1,444-token excluded tail. There is one seed.",
            "`R_model` is a logical zero-product opportunity, not measured speedup.",
            "",
        ]
    )
    TABLE.write_text("\n".join(lines), encoding="utf-8")


def _style_axis(axis: Any) -> None:
    axis.grid(True, color="#D8D8D8", linewidth=0.65, alpha=0.72)
    axis.set_axisbelow(True)
    axis.tick_params(direction="out", length=3.5, width=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")


def _annotate(axis: Any, row: Mapping[str, Any], *, full_panel: bool) -> None:
    target = float(row["target_sparsity"])
    if full_panel and target < 0.6:
        return
    if not full_panel and target > 0.5:
        return
    condition_id = str(row["condition_id"])
    offsets = {
        "gelu-control": (-4, 9),
        "relu-control": (5, -12),
    }
    if target == 0.0:
        offsets = {"gelu-control": (6, 7), "relu-control": (6, -11)}
    axis.annotate(
        rf"$p={target:.1f}$",
        xy=(
            100.0 * float(row["logical_products"]["R_model"]),
            float(row["validation"]["loss"]),
        ),
        xytext=offsets[condition_id],
        textcoords="offset points",
        color=STYLES[condition_id]["color"],
        fontsize=7.7,
        fontweight="bold",
        ha="left",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.10",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
        },
        zorder=7,
    )


def main() -> Path:
    result = _load()
    rows = result["conditions"]
    flags = _nondominated(rows)
    _write_table(result, flags)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10.2,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "pdf.fonttype": 42,
        }
    )
    fig, (axis_full, axis_detail) = plt.subplots(
        1,
        2,
        figsize=(11.2, 5.55),
        gridspec_kw={"width_ratios": (1.34, 1.0)},
    )

    row_flags = {id(row): flag for row, flag in zip(rows, flags, strict=True)}
    for condition_id in CONDITIONS:
        selected = sorted(
            (row for row in rows if row["condition_id"] == condition_id),
            key=lambda row: float(row["target_sparsity"]),
        )
        style = STYLES[condition_id]
        for axis in (axis_full, axis_detail):
            axis_rows = (
                selected
                if axis is axis_full
                else [row for row in selected if float(row["target_sparsity"]) <= 0.5]
            )
            axis_x = [
                100.0 * float(row["logical_products"]["R_model"]) for row in axis_rows
            ]
            axis_y = [float(row["validation"]["loss"]) for row in axis_rows]
            axis.plot(
                axis_x,
                axis_y,
                label=style["label"] if axis is axis_full else None,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                alpha=0.88,
                zorder=3,
            )
            axis.scatter(
                axis_x,
                axis_y,
                marker=style["marker"],
                s=49,
                facecolors=[
                    style["color"] if row_flags[id(row)] else "white" for row in axis_rows
                ],
                edgecolors=style["color"],
                linewidths=1.25,
                zorder=5,
            )
        for row in selected:
            _annotate(axis_full, row, full_panel=True)
            _annotate(axis_detail, row, full_panel=False)

    frontier_rows = sorted(
        (row for row, flag in zip(rows, flags, strict=True) if flag),
        key=lambda row: float(row["logical_products"]["R_model"]),
    )
    for axis in (axis_full, axis_detail):
        axis_frontier = (
            frontier_rows
            if axis is axis_full
            else [row for row in frontier_rows if float(row["target_sparsity"]) <= 0.5]
        )
        axis.plot(
            [100.0 * float(row["logical_products"]["R_model"]) for row in axis_frontier],
            [float(row["validation"]["loss"]) for row in axis_frontier],
            color="#222222",
            linestyle=":",
            linewidth=1.15,
            alpha=0.72,
            zorder=2,
        )
        axis.set_xlabel(r"Measured $R_{\mathrm{model}}$ (%)")
        _style_axis(axis)

    axis_full.set_xlim(-0.35, 12.35)
    axis_full.set_ylim(5.05, 9.05)
    axis_full.set_ylabel("Final validation loss (lower is better)")
    axis_full.set_title("(a) Complete target-sparsity sweep", loc="left", pad=8, fontweight="bold")

    axis_detail.set_xlim(-0.25, 7.2)
    axis_detail.set_ylim(5.18, 6.16)
    axis_detail.set_title("(b) Low-loss detail", loc="left", pad=8, fontweight="bold")

    handles = [
        Line2D(
            [0],
            [0],
            color=STYLES[condition_id]["color"],
            marker=STYLES[condition_id]["marker"],
            linestyle=STYLES[condition_id]["linestyle"],
            markerfacecolor=STYLES[condition_id]["color"],
            markeredgecolor=STYLES[condition_id]["color"],
            linewidth=1.8,
            label=STYLES[condition_id]["label"],
        )
        for condition_id in CONDITIONS
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#222222",
            marker="o",
            linestyle=":",
            markerfacecolor="#222222",
            markeredgecolor="#222222",
            linewidth=1.15,
            label="Global nondominated envelope",
        )
    )
    fig.suptitle(
        "Pythia-14M controls under uniform TEAL-style post-hoc clipping",
        x=0.5,
        y=0.977,
        fontsize=14.0,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.927,
        "Per-matrix thresholds calibrated on 10 training blocks; all 338 validation blocks evaluated",
        ha="center",
        va="center",
        fontsize=9.4,
        color="#444444",
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=3,
        frameon=False,
        handlelength=2.7,
        columnspacing=1.45,
    )
    fig.text(
        0.5,
        0.027,
        "Labels give uniform target sparsity p. Filled points lie on the global nondominated envelope; open points are dominated.\n"
        r"Lines connect increasing targets only. $R_{\mathrm{model}}$ is exact-zero logical-product opportunity, not measured speedup; one seed.",
        ha="center",
        va="bottom",
        fontsize=8.1,
        color="#444444",
        linespacing=1.34,
    )
    fig.subplots_adjust(left=0.076, right=0.987, top=0.79, bottom=0.19, wspace=0.27)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".pdf.tmp")
    fig.savefig(
        temporary,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": "Pythia-14M controls under uniform TEAL-style post-hoc clipping",
            "Author": "sparsity-spillover analysis 005",
            "Subject": "Measured R_model versus final validation loss",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    temporary.replace(OUTPUT)
    print(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    main()
