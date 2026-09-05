#!/usr/bin/env python3
"""Plot every verified full-model sentinel timing against canonical R_model."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = {
    "70M": "023-2026-09-04-pythia70m-sakana-sparse-kernel-sentinels",
    "410M": "024-2026-09-04-pythia410m-sakana-sparse-kernel-sentinels",
}
# Analysis-010 palette; colour now identifies model size, not intervention.
COLORS = {"70M": "#6F4C9B", "410M": "#56B4E9"}
LABELS = {
    "a0-gelu": "A0",
    "a1h-relu": "A1-H",
    "a4-ol1-kappa-0": r"A4, $\kappa=0$",
    "a4-ol1-kappa-0p5": r"A4, $\kappa=0.5$",
    "a7-ol1-kappa-0": r"A7, $\kappa=0$",
    "a7-ol1-kappa-0p5": r"A7, $\kappa=0.5$",
}


def collect_points() -> list[dict]:
    """Retain both batches and verify coordinates against raw timing/counts."""
    with (HERE / "condition-comparison.csv").open(newline="", encoding="utf-8") as stream:
        comparisons = list(csv.DictReader(stream))
    expected = {(size, condition) for size in RUNS for condition in LABELS}
    observed = [(row["model_size"], row["condition_id"]) for row in comparisons]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Comparison must contain all 12 unique verified sentinels.")
    for run in RUNS.values():
        verification = ROOT / "runs" / run / "results" / "raw" / "verification.json"
        if not json.loads(verification.read_text(encoding="utf-8"))["passed"]:
            raise ValueError(f"Unverified source: {verification}")

    points = []
    for row in comparisons:
        size, condition = row["model_size"], row["condition_id"]
        source = ROOT / "runs" / RUNS[size] / "results" / "raw" / f"{condition}.json"
        raw_bytes = source.read_bytes()
        raw = json.loads(raw_bytes)
        canonical = raw["canonical_logical_products"]
        counts = canonical["measured"]
        coverage = canonical["coverage"]
        if not coverage["complete_block_coverage"] or coverage["sequences"] != 338:
            raise ValueError(f"Incomplete logical-product coverage: {source}")
        rmodel = counts["block_zero_product_count"] / counts["model_product_count"]
        if not math.isclose(rmodel, float(row["R_model"]), abs_tol=1e-14, rel_tol=0):
            raise ValueError(f"R_model reduction mismatch: {source}")
        for batch in (1, 32):
            timing = raw["full_model"][str(batch)]["paired_speedup_vs_native"]["sparse_linear"]
            samples = timing["paired_block_speedups"]
            speedup = statistics.median(samples)
            if len(samples) != 7 or timing["blocks"] != 7:
                raise ValueError(f"Expected seven paired timing blocks: {source}")
            for value in (timing["median"], float(row[f"b{batch}_sparse_speedup"])):
                if not math.isclose(speedup, value, abs_tol=1e-14, rel_tol=0):
                    raise ValueError(f"Timing reduction mismatch: {source}")
            if not 0 < timing["p10"] <= speedup <= timing["p90"]:
                raise ValueError(f"Invalid timing interval: {source}")
            points.append({
                "model_size": size,
                "condition_id": condition,
                "batch_size": batch,
                "R_model": rmodel,
                "block_zero_product_count": counts["block_zero_product_count"],
                "model_product_count": counts["model_product_count"],
                "speedup_median": speedup,
                "speedup_p10": timing["p10"],
                "speedup_p90": timing["p90"],
                "timing_blocks": len(samples),
                **{f"speedup_block_{i}": value for i, value in enumerate(samples, 1)},
                "source": source.relative_to(ROOT).as_posix(),
                "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            })
    return points


def label_offset(point: dict) -> tuple[int, int]:
    condition, size, batch = point["condition_id"], point["model_size"], point["batch_size"]
    if size == "410M":
        return {
            "a0-gelu": (8, -2) if batch == 1 else (8, 7),
            "a4-ol1-kappa-0": (-9, 20) if batch == 1 else (-9, -15),
            "a7-ol1-kappa-0": (9, 9),
            "a4-ol1-kappa-0p5": (-9, -14) if batch == 1 else (-9, 9),
            "a7-ol1-kappa-0p5": (-9, 9) if batch == 1 else (-9, -15),
        }.get(condition, (8, 7))
    return {
        "a7-ol1-kappa-0": (-9, -15),
        "a4-ol1-kappa-0p5": (8, -14) if batch == 1 else (8, 7),
    }.get(condition, (8, 7))


def fit_regressions(points: list[dict]) -> list[dict]:
    """Equal-weight OLS with an intercept; one observation per checkpoint/batch."""
    fits = []
    for batch in (1, 32):
        for size in RUNS:
            group = [p for p in points if p["model_size"] == size and p["batch_size"] == batch]
            x = [p["R_model"] for p in group]
            y = [p["speedup_median"] for p in group]
            slope, intercept = statistics.linear_regression(x, y)
            residual_ss = math.fsum((yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y))
            mean_y = statistics.fmean(y)
            total_ss = math.fsum((yi - mean_y) ** 2 for yi in y)
            if total_ss == 0:
                raise ValueError(f"R-squared is undefined for constant speedup: {size}, B{batch}")
            fits.append({
                "model_size": size, "batch_size": batch, "n_checkpoints": len(group),
                "intercept": intercept, "slope": slope,
                "R_squared": 1 - residual_ss / total_ss,
                "R_model_min": min(x), "R_model_max": max(x),
                "residual_sum_squares": residual_ss, "total_sum_squares": total_ss,
                "method": "unweighted OLS with intercept on checkpoint medians",
            })
    return fits


def render(points: list[dict], fits: list[dict]) -> Path:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.2,
        "axes.labelsize": 10.3, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "pdf.fonttype": 42, "pdf.compression": 9,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.073, right=0.985, bottom=0.21, top=0.82, wspace=0.12)
    fig.text(0.073, 0.96, "Logical opportunity vs. measured speedup", fontsize=14, weight="bold")
    fig.text(0.073, 0.911, "Same H100 NVL  |  BF16  |  Uncached 2,048-token prefill", color="#444444")
    fig.legend(handles=[
        Line2D([], [], color=color, marker="o", linestyle="none", markersize=7,
               label=f"Pythia-{size}") for size, color in COLORS.items()
    ], loc="upper right", bbox_to_anchor=(0.988, 0.993), frameon=False)

    for panel, (ax, batch) in enumerate(zip(axes, (1, 32))):
        ax.set_title(f"({'ab'[panel]}) Batch size {batch}", loc="left", fontsize=10, pad=10)
        ax.set_xlim(-0.04, 0.92)
        ax.set_ylim(0, 1.115)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}×"))
        ax.grid(True, color="#D8D8D8", linewidth=0.65, alpha=0.72)
        ax.set_axisbelow(True)
        ax.tick_params(direction="out", length=3.5, width=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color("#555555")
        ax.axhline(1, color="#555555", linestyle=(0, (4, 3)), linewidth=1, zorder=2)
        ax.text(0.0, 1.031, "Break-even", fontsize=8.5, color="#555555")
        ax.set_xlabel(r"$R_{\mathrm{model}}$ (logical zero-product fraction)", labelpad=8)
        fit_handles = []
        for fit in fits:
            if fit["batch_size"] != batch:
                continue
            size = fit["model_size"]
            x_range = [fit["R_model_min"], fit["R_model_max"]]
            line, = ax.plot(
                x_range, [fit["intercept"] + fit["slope"] * x for x in x_range],
                color=COLORS[size], linestyle="--" if size == "70M" else "-.",
                linewidth=1.5, zorder=3,
                label=rf"{size}: $R^2={fit['R_squared']:.3f}$",
            )
            fit_handles.append(line)
        ax.legend(handles=fit_handles, title="Linear fits (n=6 each)", title_fontsize=8.5,
                  loc="upper right", bbox_to_anchor=(0.99, 0.91), frameon=False,
                  handlelength=2.7, fontsize=9)
        for point in points:
            if point["batch_size"] != batch:
                continue
            x, y = point["R_model"], point["speedup_median"]
            color = COLORS[point["model_size"]]
            ax.errorbar(
                x, y, yerr=[[y - point["speedup_p10"]], [point["speedup_p90"] - y]],
                fmt="o" if batch == 1 else "^", markersize=6.5, color=color,
                markeredgecolor="white", markeredgewidth=0.55,
                capsize=3, elinewidth=1.2, zorder=4,
            )
            dx, dy = label_offset(point)
            ax.annotate(
                LABELS[point["condition_id"]], xy=(x, y), xytext=(dx, dy),
                textcoords="offset points", ha="left" if dx > 0 else "right",
                va="bottom" if dy > 0 else "top", fontsize=8.3, color="#333333",
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.55,
                            "shrinkA": 1, "shrinkB": 5}, zorder=5,
            )
    axes[0].set_ylabel("Full-model speedup (dense / sparse)", labelpad=8)
    fig.text(0.073, 0.09,
             "A4/A7 denote OL1. Points: paired medians; whiskers: 10th-90th percentiles (7 blocks). Lines: descriptive OLS with intercept.",
             fontsize=8.3, color="#444444")
    fig.text(0.073, 0.05,
             "Sparse linears only; attention and LM head stay dense. 14M has no valid sparse timing (correctness gate failed).",
             fontsize=8.3, color="#444444")
    output = HERE / "figures" / "01-rmodel-vs-speedup.pdf"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, metadata={"Title": "R_model vs. full-model speedup", "CreationDate": None})
    plt.close(fig)
    return output


def main() -> None:
    points = collect_points()
    fits = fit_regressions(points)
    with (HERE / "rmodel-speedup-points.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    with (HERE / "rmodel-speedup-regressions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fits[0]))
        writer.writeheader()
        writer.writerows(fits)
    output = render(points, fits)
    print(f"PASS: {len(points)} points, {sum(p['timing_blocks'] for p in points)} paired blocks; {output}")
    for fit in fits:
        print(f"{fit['model_size']} B{fit['batch_size']}: n={fit['n_checkpoints']}, R2={fit['R_squared']:.6f}")


if __name__ == "__main__":
    main()
