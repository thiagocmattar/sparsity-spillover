"""Plot per-boundary gradient-conflict trajectories for Analysis 003."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
OUTPUT_CONTEXT = ANALYSIS_DIR / "figures/02-gradient-conflict-trajectories.pdf"
OUTPUT_OL1 = ANALYSIS_DIR / "figures/03-ol1-orthogonalization-trajectories.pdf"
RUNS = {
    "l1_naive": REPO_ROOT / "runs/004-2026-08-29-pythia14m-full-pass-l1n",
    "orthogonal_l1": REPO_ROOT / "runs/009-2026-08-30-pythia14m-full-pass-ol1",
}
LAMBDAS = (0.05, 0.1, 0.5, 1.0)
EXPECTED_STEPS = 712
ROLLING_WINDOW = 51


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _rolling(values: list[float]) -> dict[str, list[float]]:
    if ROLLING_WINDOW % 2 != 1 or len(values) < ROLLING_WINDOW:
        raise RuntimeError("Rolling window must be odd and fit the event history.")
    half = ROLLING_WINDOW // 2
    result = {"index": [], "mean": [], "p05": [], "p95": []}
    for center in range(half, len(values) - half):
        selected = values[center - half : center + half + 1]
        result["index"].append(center)
        result["mean"].append(sum(selected) / len(selected))
        result["p05"].append(_quantile(selected, 0.05))
        result["p95"].append(_quantile(selected, 0.95))
    return result


def _load() -> dict[str, dict[float, list[dict[str, Any]]]]:
    result: dict[str, dict[float, list[dict[str, Any]]]] = {}
    identities: dict[str, tuple[str, str]] = {}
    token_schedules: list[tuple[int, ...]] = []
    for method, run_dir in RUNS.items():
        verification = json.loads(
            (run_dir / "artifacts/verification.json").read_text(encoding="utf-8")
        )
        if (
            verification.get("status") != "verified"
            or verification.get("evidence_label") != "valid"
        ):
            raise RuntimeError(f"Run is not verified valid evidence: {run_dir}")
        identities[method] = (
            verification["initial_parameter_sha256"],
            verification["training_schedule_sha256"],
        )
        conditions: dict[float, list[dict[str, Any]]] = {}
        for verified in verification["conditions"]:
            condition = verified["condition"]
            if condition["pressure_method"] != method:
                continue
            if (
                condition["activation"] != "relu"
                or condition["pressure_sites"] != ["h"]
                or condition["is_control"]
            ):
                raise RuntimeError(f"Unexpected pressure condition: {condition!r}")
            weight = float(condition["pressure_weight"])
            if weight in conditions:
                raise RuntimeError(f"Duplicate lambda for {method}: {weight}")
            events_path = (
                run_dir
                / "artifacts/attempts"
                / verified["attempt_id"]
                / "events.jsonl"
            )
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            rows = [row for row in events if row.get("event") == "train"]
            if len(rows) != EXPECTED_STEPS:
                raise RuntimeError(f"Incomplete event history: {events_path}")
            if [int(row["step"]) for row in rows] != list(range(1, EXPECTED_STEPS + 1)):
                raise RuntimeError(f"Non-contiguous steps: {events_path}")
            token_schedules.append(tuple(int(row["input_tokens_seen"]) for row in rows))
            for row in rows:
                raw_dot = float(row["task_pressure_gradient_dot"])
                raw_cosine = float(row["task_pressure_gradient_cosine"])
                if (
                    not math.isfinite(raw_dot)
                    or not math.isfinite(raw_cosine)
                    or abs(raw_cosine) > 1.0 + 1e-9
                ):
                    raise RuntimeError(f"Invalid raw gradient interaction: {events_path}")
                if bool(row["gradient_conflict"]) != (raw_dot < 0.0):
                    raise RuntimeError(f"Conflict flag differs from dot sign: {events_path}")
                if method == "orthogonal_l1":
                    before = float(row["task_pressure_cosine_before"])
                    after = float(row["task_pressure_cosine_after"])
                    adaptive_dot = float(row["task_pressure_dot_before"])
                    if not all(math.isfinite(value) for value in (before, after, adaptive_dot)):
                        raise RuntimeError(f"Invalid OL1 adaptive interaction: {events_path}")
                    if bool(row["projection_applied"]) != (adaptive_dot < 0.0):
                        raise RuntimeError(f"OL1 projection differs from dot sign: {events_path}")
            conditions[weight] = rows
        if set(conditions) != set(LAMBDAS):
            raise RuntimeError(f"Unexpected lambda grid for {method}: {sorted(conditions)}")
        result[method] = conditions

    if identities["l1_naive"] != identities["orthogonal_l1"]:
        raise RuntimeError("Run initialization or training schedule differs.")
    if len(set(token_schedules)) != 1:
        raise RuntimeError("Per-boundary token schedules differ across conditions.")
    return result


def _plot_series(
    ax: Any,
    x: list[float],
    values: list[float],
    style: dict[str, Any],
) -> None:
    rolling = _rolling(values)
    rolling_x = [x[index] for index in rolling["index"]]
    ax.scatter(
        x,
        values,
        color=style["color"],
        marker=style["marker"],
        s=4.0,
        alpha=style["point_alpha"],
        linewidths=0.0,
        zorder=1,
    )
    ax.fill_between(
        rolling_x,
        rolling["p05"],
        rolling["p95"],
        color=style["color"],
        alpha=style["band_alpha"],
        linewidth=0.0,
        zorder=2,
    )
    ax.plot(
        rolling_x,
        rolling["mean"],
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=1.55,
        zorder=3,
    )


def _base_figure(
    title: str,
    *,
    top: float = 0.76,
    bottom: float = 0.23,
    xlabel_y: float = 0.06,
) -> tuple[Any, list[Any]]:
    import matplotlib.pyplot as plt

    fig, axes_array = plt.subplots(
        1,
        4,
        figsize=(7.15, 2.72),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    axes = list(axes_array)
    fig.suptitle(title, x=0.535, y=0.985, fontsize=10.5, fontweight="bold")
    fig.supxlabel("Training input tokens (billions)", x=0.535, y=xlabel_y, fontsize=8.8)
    fig.subplots_adjust(left=0.09, right=0.995, top=top, bottom=bottom, wspace=0.15)
    return fig, axes


def _format_axes(axes: list[Any], ylim: tuple[float, float], yticks: tuple[float, ...]) -> None:
    for ax, weight in zip(axes, LAMBDAS, strict=True):
        ax.axhline(0.0, color="#666666", linewidth=0.7, zorder=0)
        ax.set_title(rf"$\lambda={weight:g}$", pad=4)
        ax.set_xlim(0.0, 1.52)
        ax.set_xticks((0.0, 0.5, 1.0, 1.5))
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#888888")
        ax.spines["bottom"].set_color("#888888")
        ax.tick_params(length=2.5, color="#777777")
    axes[0].set_ylabel("Gradient-conflict cosine\n(negative = conflict)")


def _legend(
    fig: Any,
    styles: list[dict[str, Any]],
    *,
    y: float = 0.91,
    location: str = "upper center",
    fontsize: float | None = None,
) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4.0,
            linewidth=1.5,
            label=style["label"],
        )
        for style in styles
    ]
    fig.legend(
        handles=handles,
        loc=location,
        bbox_to_anchor=(0.535, y),
        ncols=len(handles),
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
        fontsize=fontsize,
    )


def _save(fig: Any, output: Path, title: str, subject: str) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pdf.tmp")
    fig.savefig(
        temporary,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": title,
            "Author": "sparsity-spillover analysis 003",
            "Subject": subject,
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    temporary.replace(output)


def _context_figure(events: dict[str, dict[float, list[dict[str, Any]]]]) -> Path:
    styles = [
        {
            "label": "Naive L1: raw task-pressure",
            "color": "#0072B2",
            "linestyle": "-",
            "marker": "o",
            "point_alpha": 0.19,
            "band_alpha": 0.09,
        },
        {
            "label": "OL1: Adam-relative before",
            "color": "#D55E00",
            "linestyle": "--",
            "marker": "s",
            "point_alpha": 0.22,
            "band_alpha": 0.13,
        },
        {
            "label": "OL1: Adam-relative after",
            "color": "#009E73",
            "linestyle": "-.",
            "marker": "^",
            "point_alpha": 0.24,
            "band_alpha": 0.14,
        },
    ]
    fig, axes = _base_figure("Gradient conflict: naive L1 context and OL1 projection")
    for ax, weight in zip(axes, LAMBDAS, strict=True):
        naive_rows = events["l1_naive"][weight]
        ol1_rows = events["orthogonal_l1"][weight]
        x = [float(row["input_tokens_seen"]) / 1e9 for row in naive_rows]
        _plot_series(
            ax,
            x,
            [float(row["task_pressure_gradient_cosine"]) for row in naive_rows],
            styles[0],
        )
        _plot_series(
            ax,
            x,
            [float(row["task_pressure_cosine_before"]) for row in ol1_rows],
            styles[1],
        )
        _plot_series(
            ax,
            x,
            [float(row["task_pressure_cosine_after"]) for row in ol1_rows],
            styles[2],
        )
    _format_axes(axes, (-0.7, 0.8), (-0.6, -0.3, 0.0, 0.3, 0.6))
    _legend(fig, styles)
    _save(
        fig,
        OUTPUT_CONTEXT,
        "Gradient conflict: naive L1 context and OL1 projection",
        "Raw naive-L1 interaction and Adam-relative OL1 interaction before and after projection",
    )
    return OUTPUT_CONTEXT


def _ol1_figure(events: dict[str, dict[float, list[dict[str, Any]]]]) -> Path:
    styles = [
        {
            "label": "Before projection",
            "color": "#D55E00",
            "linestyle": "--",
            "marker": "s",
            "point_alpha": 0.25,
            "band_alpha": 0.14,
        },
        {
            "label": "After projection",
            "color": "#009E73",
            "linestyle": "-",
            "marker": "^",
            "point_alpha": 0.28,
            "band_alpha": 0.16,
        },
    ]
    fig, axes = _base_figure(
        "OL1 removes Adam-relative gradient conflict",
        top=0.84,
        bottom=0.30,
        xlabel_y=0.14,
    )
    for ax, weight in zip(axes, LAMBDAS, strict=True):
        rows = events["orthogonal_l1"][weight]
        x = [float(row["input_tokens_seen"]) / 1e9 for row in rows]
        _plot_series(
            ax,
            x,
            [float(row["task_pressure_cosine_before"]) for row in rows],
            styles[0],
        )
        _plot_series(
            ax,
            x,
            [float(row["task_pressure_cosine_after"]) for row in rows],
            styles[1],
        )
    _format_axes(axes, (-0.19, 0.02), (-0.18, -0.12, -0.06, 0.0))
    _legend(fig, styles, y=0.035, location="center", fontsize=6.8)
    _save(
        fig,
        OUTPUT_OL1,
        "OL1 removes Adam-relative gradient conflict",
        "OL1 Adam-relative task-pressure cosine before and after projection",
    )
    return OUTPUT_OL1


def main() -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.8,
            "pdf.fonttype": 42,
        }
    )
    events = _load()
    outputs = (_context_figure(events), _ol1_figure(events))
    for output in outputs:
        print(output)
    return outputs


if __name__ == "__main__":
    main()
