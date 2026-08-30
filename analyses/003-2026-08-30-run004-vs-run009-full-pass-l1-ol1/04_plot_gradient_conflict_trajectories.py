"""Plot matched raw task-pressure gradient-conflict trajectories."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
OUTPUT = ANALYSIS_DIR / "figures/02-gradient-conflict-trajectories.pdf"
RUNS = {
    "l1_naive": REPO_ROOT / "runs/004-2026-08-29-pythia14m-full-pass-l1n",
    "orthogonal_l1": REPO_ROOT / "runs/009-2026-08-30-pythia14m-full-pass-ol1",
}
LAMBDAS = (0.05, 0.1, 0.5, 1.0)
EXPECTED_STEPS = 712
BIN_COUNT = 24


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _load() -> dict[str, dict[float, list[dict[str, Any]]]]:
    result: dict[str, dict[float, list[dict[str, Any]]]] = {}
    identities: dict[str, tuple[str, str]] = {}
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
            weight = float(condition["pressure_weight"])
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
            for row in rows:
                dot = float(row["task_pressure_gradient_dot"])
                cosine = float(row["task_pressure_gradient_cosine"])
                if not math.isfinite(dot) or not math.isfinite(cosine) or abs(cosine) > 1.0 + 1e-9:
                    raise RuntimeError(f"Invalid gradient interaction: {events_path}")
                if bool(row["gradient_conflict"]) != (dot < 0.0):
                    raise RuntimeError(f"Conflict flag differs from dot sign: {events_path}")
            conditions[weight] = rows
        if set(conditions) != set(LAMBDAS):
            raise RuntimeError(f"Unexpected lambda grid for {method}: {sorted(conditions)}")
        result[method] = conditions
    if identities["l1_naive"] != identities["orthogonal_l1"]:
        raise RuntimeError("Run initialization or training schedule differs.")
    return result


def _bins(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    values = {
        "tokens_billions": [],
        "cosine_q25": [],
        "cosine_median": [],
        "cosine_q75": [],
        "conflict_percent": [],
    }
    for index in range(BIN_COUNT):
        start = index * len(rows) // BIN_COUNT
        stop = (index + 1) * len(rows) // BIN_COUNT
        selected = rows[start:stop]
        cosines = [float(row["task_pressure_gradient_cosine"]) for row in selected]
        values["tokens_billions"].append(
            _quantile([float(row["input_tokens_seen"]) for row in selected], 0.5) / 1e9
        )
        values["cosine_q25"].append(_quantile(cosines, 0.25))
        values["cosine_median"].append(_quantile(cosines, 0.50))
        values["cosine_q75"].append(_quantile(cosines, 0.75))
        values["conflict_percent"].append(
            100.0 * sum(bool(row["gradient_conflict"]) for row in selected) / len(selected)
        )
    return values


def main() -> Path:
    events = _load()
    reduced = {
        method: {weight: _bins(rows) for weight, rows in conditions.items()}
        for method, conditions in events.items()
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8.2,
            "pdf.fonttype": 42,
        }
    )
    styles = {
        "l1_naive": {
            "label": "Naive L1",
            "color": "#0072B2",
            "linestyle": "-",
            "marker": "o",
        },
        "orthogonal_l1": {
            "label": "OL1",
            "color": "#D55E00",
            "linestyle": "--",
            "marker": "s",
        },
    }

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(7.15, 4.35),
        sharex=True,
        sharey="row",
        constrained_layout=False,
    )
    for column, weight in enumerate(LAMBDAS):
        cosine_ax = axes[0, column]
        conflict_ax = axes[1, column]
        for method, style in styles.items():
            row = reduced[method][weight]
            x = row["tokens_billions"]
            cosine_ax.fill_between(
                x,
                row["cosine_q25"],
                row["cosine_q75"],
                color=style["color"],
                alpha=0.14,
                linewidth=0.0,
                zorder=1,
            )
            cosine_ax.plot(
                x,
                row["cosine_median"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.45,
                zorder=2,
            )
            conflict_ax.plot(
                x,
                row["conflict_percent"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=2.5,
                markevery=2,
                linewidth=1.25,
                markeredgewidth=0.0,
                zorder=2,
            )

        cosine_ax.axhline(0.0, color="#666666", linewidth=0.7, zorder=0)
        conflict_ax.axhline(50.0, color="#777777", linewidth=0.7, zorder=0)
        cosine_ax.set_title(rf"$\lambda={weight:g}$", pad=4)
        cosine_ax.set_ylim(-0.5, 0.5)
        conflict_ax.set_ylim(0.0, 100.0)
        cosine_ax.set_yticks((-0.5, -0.25, 0.0, 0.25, 0.5))
        conflict_ax.set_yticks((0.0, 25.0, 50.0, 75.0, 100.0))
        conflict_ax.set_xlim(0.0, 1.52)
        conflict_ax.set_xticks((0.0, 0.5, 1.0, 1.5))
        for ax in (cosine_ax, conflict_ax):
            ax.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.75)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#888888")
            ax.spines["bottom"].set_color("#888888")
            ax.tick_params(length=2.5, color="#777777")

    axes[0, 0].set_ylabel("Raw gradient cosine")
    axes[1, 0].set_ylabel("Conflict boundaries (%)")
    fig.supxlabel("Training input tokens (billions)", x=0.53, y=0.045, fontsize=8.8)
    fig.suptitle(
        "Task-pressure gradient conflict through training",
        x=0.53,
        y=0.985,
        fontsize=10.5,
        fontweight="bold",
    )
    legend = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4.2,
            linewidth=1.5,
            label=style["label"],
        )
        for style in styles.values()
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.945),
        ncols=2,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.7,
    )
    fig.text(
        0.53,
        0.905,
        "One seed; 24 bins (29-30 boundaries): median/IQR above, negative-dot fraction below",
        ha="center",
        va="top",
        fontsize=7.4,
        color="#444444",
    )
    fig.subplots_adjust(left=0.085, right=0.995, top=0.835, bottom=0.145, wspace=0.16, hspace=0.18)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".pdf.tmp")
    fig.savefig(
        temporary,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": "Task-pressure gradient conflict trajectories",
            "Author": "sparsity-spillover analysis 003",
            "Subject": "Run 004 naive L1 and Run 009 OL1",
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
