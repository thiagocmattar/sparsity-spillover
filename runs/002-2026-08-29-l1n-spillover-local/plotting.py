"""Publication figure and observation from terminally verified Run 002 data."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from run_config import DEFAULT_CONFIG, RUN_DIR, load_config, write_json


DEFAULT_SUMMARY = RUN_DIR / "artifacts" / "verification.json"


def reduce_points(summary: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if summary.get("status") != "verified":
        raise ValueError("Figure input must be a terminally verified summary.")
    grouped = {"gelu": [], "relu": []}
    for row in summary["conditions"]:
        condition = row["condition"]
        activation = condition["activation"]
        if activation not in grouped:
            raise ValueError(f"Unexpected activation: {activation!r}")
        point = {
            "condition_id": condition["id"],
            "order": int(condition["order"]),
            "activation": activation,
            "label": "control" if condition["is_control"] else f"lambda={condition['pressure_weight']:g}",
            "pressure_weight": float(condition["pressure_weight"]),
            "is_control": bool(condition["is_control"]),
            "h_near_zero_percent": 100.0
            * float(row["h_near_zero_fraction_epsilon_0p001"]),
            "attention_mean_near_zero_percent": 100.0
            * float(row["attention_mean_near_zero_fraction_epsilon_0p001"]),
            "attention_site_near_zero_percent": {
                site: 100.0 * float(value)
                for site, value in row[
                    "attention_site_near_zero_fractions_epsilon_0p001"
                ].items()
            },
            "final_validation_loss": float(row["final_validation_loss"]),
        }
        if not all(
            math.isfinite(float(point[key]))
            for key in (
                "h_near_zero_percent",
                "attention_mean_near_zero_percent",
                "final_validation_loss",
            )
        ):
            raise ValueError(f"Nonfinite figure point: {condition['id']}")
        grouped[activation].append(point)
    for activation, points in grouped.items():
        points.sort(key=lambda point: point["order"])
        if len(points) != 5 or [point["label"] for point in points] != [
            "control",
            "lambda=0.1",
            "lambda=0.5",
            "lambda=1",
            "lambda=5",
        ]:
            raise ValueError(f"Incomplete or misordered {activation} line.")
    return grouped


def generate_figure(
    summary_path: str | Path = DEFAULT_SUMMARY,
    config_path: str | Path = DEFAULT_CONFIG,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    config = load_config(config_path)
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    grouped = reduce_points(summary)
    output = RUN_DIR / str(config["figure"]["output"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
        }
    )
    styles = {
        "gelu": {
            "display": "GeLU",
            "color": "#0072B2",
            "marker": "o",
            "linestyle": "-",
            "offset": 8,
            "vertical": "bottom",
        },
        "relu": {
            "display": "ReLU",
            "color": "#D55E00",
            "marker": "s",
            "linestyle": "--",
            "offset": -10,
            "vertical": "top",
        },
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.22)
    for activation in ("gelu", "relu"):
        points = grouped[activation]
        style = styles[activation]
        x_values = [point["h_near_zero_percent"] for point in points]
        y_values = [point["attention_mean_near_zero_percent"] for point in points]
        ax.plot(
            x_values,
            y_values,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.8,
            label=style["display"],
        )
        for point in points:
            ax.annotate(
                point["label"],
                (point["h_near_zero_percent"], point["attention_mean_near_zero_percent"]),
                xytext=(0, style["offset"]),
                textcoords="offset points",
                ha="center",
                va=style["vertical"],
                fontsize=8.5,
                color="#202020",
            )

    all_points = grouped["gelu"] + grouped["relu"]
    x_max = max(point["h_near_zero_percent"] for point in all_points)
    y_max = max(point["attention_mean_near_zero_percent"] for point in all_points)
    ax.set_xlim(0.0, max(1.0, x_max * 1.12))
    ax.set_ylim(0.0, max(0.01, y_max * 1.18))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.set_xlabel("Near-zero mass at h, |x| <= 1e-3 (%)", labelpad=8)
    ax.set_ylabel("Mean near-zero mass at q_post, k_post, v (%)", labelpad=8)
    ax.set_title("Pythia-14M L1N pressure and attention spillover", pad=12)
    ax.grid(axis="both", color="#D9D9D9", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    steps = int(summary["conditions"][0]["completed_steps"])
    fig.text(
        0.5,
        0.035,
        f"Seed 0; {steps} updates per condition; all 338 validation blocks; lines follow control to increasing lambda",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#404040",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)

    figure_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    figure_data = {
        "schema_version": 1,
        "source": str(Path(summary_path).relative_to(RUN_DIR)).replace("\\", "/"),
        "epsilon": 0.001,
        "x_estimand": "count-pooled h near-zero percent",
        "y_estimand": "unweighted mean of separately count-pooled q_post, k_post, and v near-zero percents",
        "series": grouped,
        "figure": {
            "path": str(output.relative_to(RUN_DIR)).replace("\\", "/"),
            "bytes": output.stat().st_size,
            "sha256": figure_sha,
        },
        "plotting_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(RUN_DIR / "artifacts" / "figure_data.json", figure_data)
    _write_observation(summary, grouped, output)
    completion = {
        "schema_version": 1,
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "verification": "artifacts/verification.json",
        "figure": figure_data["figure"],
        "observation": "observations/O001-h-vs-attention-near-zero.md",
    }
    driver_path = RUN_DIR / "artifacts" / "driver.json"
    if driver_path.exists():
        driver = json.loads(driver_path.read_text(encoding="utf-8"))
        completion["driver_started_at"] = driver.get("started_at")
        if driver.get("started_at"):
            started = datetime.fromisoformat(driver["started_at"])
            completion["inclusive_wall_seconds"] = (
                datetime.fromisoformat(completion["completed_at"]) - started
            ).total_seconds()
    write_json(RUN_DIR / "artifacts" / "completion.json", completion)
    write_json(
        RUN_DIR / "artifacts" / "progress.json",
        {
            "status": "completed",
            "condition_count": 10,
            "completed_conditions": 10,
            "inclusive_wall_seconds": completion.get("inclusive_wall_seconds"),
            "figure": figure_data["figure"]["path"],
        },
    )
    return output


def _write_observation(
    summary: Mapping[str, Any],
    grouped: Mapping[str, list[Mapping[str, Any]]],
    output: Path,
) -> None:
    rows = []
    for activation in ("gelu", "relu"):
        for point in grouped[activation]:
            rows.append(
                "| {activation} | {label} | {h:.6f} | {attention:.6f} | {loss:.6f} |".format(
                    activation="GeLU" if activation == "gelu" else "ReLU",
                    label=point["label"],
                    h=point["h_near_zero_percent"],
                    attention=point["attention_mean_near_zero_percent"],
                    loss=point["final_validation_loss"],
                )
            )
    text = f"""# O001 - L1N pressure at h versus attention near-zero mass

## Question

As naive L1 pressure at `h` increases, does count-pooled near-zero mass move
toward `h` while the untargeted post-RoPE query, key, and value sites lose
near-zero mass, and does the trajectory differ between GeLU and ReLU?

## Method and coverage

The figure uses ten matched randomly initialized Pythia-14M conditions: one
control and four `l1_naive` weights per activation. Every point is measured at
the reloaded final checkpoint over all 500 MiniPile validation documents, 338
complete 2,048-token blocks, 692,224 input tokens, and the declared 1,444-token
excluded tail. Model/data seeds are both 0 and all conditions share the same
initial-parameter and training-schedule hashes. The near-zero threshold is
`abs(x) <= 1e-3`.

X is formed by pooling integer `h` hits and denominators across validation
batches and all six layers, then dividing once. For Y, each of `q_post`,
`k_post`, and `v` is pooled the same way and the three resulting fractions are
averaged without weights. Their denominators are equal, so the result also
equals a joint count pool across those sites. No condition or seed averaging is
performed.

## Values

| Activation | Pressure label | h near-zero (%) | Attention mean near-zero (%) | Final validation loss |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(rows)}

## Figure caption and encoding

**Figure 01. Near-zero mass at pressured FFN `h` versus mean near-zero mass at
untargeted attention sites for local Pythia-14M pretraining.** Blue circles and
a solid line denote GeLU; orange squares and a dashed line denote ReLU. Labels
identify the no-pressure control and naive-L1 weight. Lines connect conditions
in ascending pressure order and are visual guides, not fitted relationships.
Both axes begin at zero and report percent at epsilon `1e-3`.

## Interpretation and limits

The plotted geometry is descriptive. A down-right within-activation trajectory
is the prespecified simple spillover signature; the observed pattern must be
interpreted together with final validation loss and whether `h` responded to
pressure. This one-seed, two-hour local cohort does not establish a causal
route, seed uncertainty, a long-horizon optimum, removable logical products,
or measured speedup. GeLU-versus-ReLU differences intentionally include the
operator change. The attempts are labeled `{summary['evidence_label']}`.

## Provenance

- Source: `../artifacts/verification.json`
- Numerical reduction: `../artifacts/figure_data.json`
- Source script: `../04_plot.py` and `../plotting.py`
- Output: `../{output.relative_to(RUN_DIR).as_posix()}`
"""
    observation = RUN_DIR / "observations" / "O001-h-vs-attention-near-zero.md"
    observation.write_text(text, encoding="utf-8", newline="\n")
    index = """# Observations

| ID | Observation | Evidence |
| --- | --- | --- |
| O001 | L1N-at-h versus untargeted attention near-zero mass for GeLU and ReLU. | valid - ten matched conditions, full validation, verified checkpoints |
"""
    (RUN_DIR / "observations" / "INDEX.md").write_text(
        index, encoding="utf-8", newline="\n"
    )
