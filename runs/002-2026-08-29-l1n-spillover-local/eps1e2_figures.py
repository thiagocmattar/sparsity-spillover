"""Count-first epsilon-1e-2 reductions and publication figures for Run 002."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from run_config import RUN_DIR, write_json


EPSILON = 0.01
THRESHOLD_KEY = "0.01"
VERIFICATION_PATH = RUN_DIR / "artifacts" / "verification.json"
M_DIAGNOSTIC_PATH = RUN_DIR / "artifacts" / "posthoc-m-activation-statistics.json"
OUTPUT_DIAGNOSTIC_PATH = (
    RUN_DIR / "artifacts" / "posthoc-attention-output-activation-statistics.json"
)
FIGURE_PATHS = {
    "attention_mean": RUN_DIR / "figures" / "04-h-vs-attention-near-zero-eps1e-2.pdf",
    "sitewise": RUN_DIR / "figures" / "05-h-vs-site-near-zero-grid-eps1e-2.pdf",
    "attention_output": RUN_DIR
    / "figures"
    / "06-h-vs-attention-output-near-zero-eps1e-2.pdf",
}
DATA_PATHS = {
    "attention_mean": RUN_DIR / "artifacts" / "figure_data_attention_mean_eps1e-2.json",
    "sitewise": RUN_DIR / "artifacts" / "figure_data_sitewise_eps1e-2.json",
    "attention_output": RUN_DIR
    / "artifacts"
    / "figure_data_attention_output_eps1e-2.json",
}
OBSERVATION_PATHS = {
    "attention_mean": RUN_DIR / "observations" / "O004-h-vs-attention-near-zero-eps1e-2.md",
    "sitewise": RUN_DIR / "observations" / "O005-h-vs-site-near-zero-grid-eps1e-2.md",
    "attention_output": RUN_DIR
    / "observations"
    / "O006-h-vs-attention-output-near-zero-eps1e-2.md",
}
COMPLETION_PATH = RUN_DIR / "artifacts" / "eps1e-2_figure_completion.json"
SITEWISE_Y = ("q_post", "k_post", "v", "m")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pooled_counts(statistics: Mapping[str, Any], site: str) -> dict[str, Any]:
    matches = [row for row in statistics["pooled_by_site"] if row["name"] == site]
    if len(matches) != 1:
        raise ValueError(f"Expected one pooled row for site {site!r}.")
    row = matches[0]
    hits = int(row["threshold_hits"][THRESHOLD_KEY])
    total = int(row["total"])
    if total <= 0 or not 0 <= hits <= total:
        raise ValueError(f"Invalid near-zero counts for site {site!r}.")
    fraction = hits / total
    if not math.isclose(
        fraction,
        float(row["threshold_fractions"][THRESHOLD_KEY]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(f"Stored fraction disagrees with integer counts for {site!r}.")
    return {"hits": hits, "total": total, "fraction": fraction, "percent": 100.0 * fraction}


def reduce_points(
    verification: Mapping[str, Any],
    m_diagnostic: Mapping[str, Any],
    output_diagnostic: Mapping[str, Any],
    statistics_loader: Callable[[str], Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build all three figure inputs from epsilon-1e-2 integer counts."""

    if verification.get("status") != "verified" or verification.get("evidence_label") != "valid":
        raise ValueError("Figure input must be valid terminal verification evidence.")
    if m_diagnostic.get("status") != "verified" or output_diagnostic.get("status") != "verified":
        raise ValueError("Both post-hoc diagnostic sources must be verified.")
    m_by_attempt = {row["attempt_id"]: row for row in m_diagnostic["conditions"]}
    output_by_attempt = {
        row["attempt_id"]: row for row in output_diagnostic["conditions"]
    }
    grouped: dict[str, list[dict[str, Any]]] = {"gelu": [], "relu": []}
    for source in verification["conditions"]:
        attempt_id = source["attempt_id"]
        condition = source["condition"]
        activation = condition["activation"]
        if (
            activation not in grouped
            or attempt_id not in m_by_attempt
            or attempt_id not in output_by_attempt
        ):
            raise ValueError(f"Incomplete epsilon-1e-2 input for {attempt_id}.")
        original = statistics_loader(attempt_id)
        counts = {
            site: _pooled_counts(
                m_by_attempt[attempt_id]["statistics"]
                if site == "m"
                else output_by_attempt[attempt_id]["statistics"]
                if site == "attention_output"
                else original,
                site,
            )
            for site in ("h", "q_post", "k_post", "v", "m", "attention_output")
        }
        attention_mean = sum(counts[site]["fraction"] for site in ("q_post", "k_post", "v")) / 3.0
        point = {
            "attempt_id": attempt_id,
            "condition_id": condition["id"],
            "activation": activation,
            "order": int(condition["order"]),
            "label": "control" if condition["is_control"] else f"lambda={condition['pressure_weight']:g}",
            "pressure_weight": float(condition["pressure_weight"]),
            "is_control": bool(condition["is_control"]),
            "near_zero": counts,
            "attention_mean": {
                "fraction": attention_mean,
                "percent": 100.0 * attention_mean,
                "definition": "unweighted mean of separately count-pooled q_post, k_post, and v fractions",
            },
            "final_validation_loss": float(source["final_validation_loss"]),
        }
        grouped[activation].append(point)

    expected = ["control", "lambda=0.1", "lambda=0.5", "lambda=1", "lambda=5"]
    for activation, points in grouped.items():
        points.sort(key=lambda point: point["order"])
        if [point["label"] for point in points] != expected:
            raise ValueError(f"Incomplete or misordered {activation} series.")
    return grouped


def _load_sources() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    verification = json.loads(VERIFICATION_PATH.read_text(encoding="utf-8"))
    m_diagnostic = json.loads(M_DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
    output_diagnostic = json.loads(OUTPUT_DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
    m_by_attempt = {row["attempt_id"]: row for row in m_diagnostic["conditions"]}
    output_by_attempt = {
        row["attempt_id"]: row for row in output_diagnostic["conditions"]
    }

    def load_statistics(attempt_id: str) -> Mapping[str, Any]:
        m_record = m_by_attempt[attempt_id]
        output_record = output_by_attempt[attempt_id]
        if m_record["original_activation_statistics"] != output_record["original_activation_statistics"]:
            raise ValueError(f"Post-hoc sources disagree on terminal statistics for {attempt_id}.")
        source = m_record["original_activation_statistics"]
        path = RUN_DIR / source["path"]
        if _sha256(path) != source["sha256"]:
            raise ValueError(f"Original activation diagnostic changed for {attempt_id}.")
        return json.loads(path.read_text(encoding="utf-8"))

    grouped = reduce_points(
        verification, m_diagnostic, output_diagnostic, load_statistics
    )
    return verification, m_diagnostic, output_diagnostic, grouped


def _style() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[tuple[int, int, str]]],
    dict[str, list[tuple[int, int, str]]],
]:
    styles = {
        "gelu": {"display": "GeLU", "color": "#0072B2", "marker": "o", "linestyle": "-"},
        "relu": {"display": "ReLU", "color": "#D55E00", "marker": "s", "linestyle": "--"},
    }
    full_label_layout = {
        "gelu": [
            (0, -11, "center"),
            (0, 7, "center"),
            (0, 7, "center"),
            (0, 7, "center"),
            (-7, -11, "right"),
        ],
        "relu": [
            (0, -11, "center"),
            (-28, 7, "right"),
            (-7, -11, "right"),
            (4, 7, "left"),
            (0, -11, "center"),
        ],
    }
    short_label_layout = {
        "gelu": [
            (0, -11, "center"),
            (0, 7, "center"),
            (0, 7, "center"),
            (0, 7, "center"),
            (-4, -11, "right"),
        ],
        "relu": [
            (0, -11, "center"),
            (-4, 7, "right"),
            (-4, -11, "right"),
            (3, 7, "left"),
            (0, -11, "center"),
        ],
    }
    return styles, full_label_layout, short_label_layout


def _setup_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    return plt


def _draw_line(
    ax: Any,
    grouped: Mapping[str, list[Mapping[str, Any]]],
    *,
    x_value: Callable[[Mapping[str, Any]], float],
    y_value: Callable[[Mapping[str, Any]], float],
    short_labels: bool,
) -> None:
    styles, full_label_layout, short_label_layout = _style()
    layouts = short_label_layout if short_labels else full_label_layout
    for activation in ("gelu", "relu"):
        points = grouped[activation]
        style = styles[activation]
        ax.plot(
            [x_value(point) for point in points],
            [y_value(point) for point in points],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=6.0 if short_labels else 6.5,
            markerfacecolor="white",
            markeredgewidth=1.7,
            label=style["display"],
        )
        for index, point in enumerate(points):
            dx, dy, horizontal_alignment = layouts[activation][index]
            label = (
                "ctrl" if point["is_control"] else f"{point['pressure_weight']:g}"
            ) if short_labels else point["label"]
            ax.annotate(
                label,
                (x_value(point), y_value(point)),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=horizontal_alignment,
                va="bottom" if dy > 0 else "top",
                fontsize=7.7 if short_labels else 8.2,
                color="#202020",
            )


def _finish_axis(ax: Any, *, x_max: float, y_max: float) -> None:
    from matplotlib.ticker import MaxNLocator

    ax.set_xlim(0.0, max(1.0, x_max * 1.10))
    ax.set_ylim(0.0, max(0.01, y_max * 1.25))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.grid(axis="both", color="#D9D9D9", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _footer(fig: Any) -> None:
    fig.text(
        0.5,
        0.035,
        "Seed 0; |x| <= 1e-2; 451 updates per condition; all 338 validation blocks; lines follow increasing lambda",
        ha="center",
        va="bottom",
        fontsize=8.1,
        color="#404040",
    )


def _save_figure(fig: Any, path: Path, plt: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return {"path": path.relative_to(RUN_DIR).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _provenance() -> dict[str, Any]:
    return {
        "verification": {"path": VERIFICATION_PATH.relative_to(RUN_DIR).as_posix(), "sha256": _sha256(VERIFICATION_PATH)},
        "m_diagnostic": {"path": M_DIAGNOSTIC_PATH.relative_to(RUN_DIR).as_posix(), "sha256": _sha256(M_DIAGNOSTIC_PATH)},
        "attention_output_diagnostic": {"path": OUTPUT_DIAGNOSTIC_PATH.relative_to(RUN_DIR).as_posix(), "sha256": _sha256(OUTPUT_DIAGNOSTIC_PATH)},
        "source_code": {
            "10_plot_eps1e-2_figures.py": _sha256(RUN_DIR / "10_plot_eps1e-2_figures.py"),
            "eps1e2_figures.py": _sha256(Path(__file__)),
        },
    }


def _figure_attention_mean(grouped: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.22)
    _draw_line(
        ax,
        grouped,
        x_value=lambda point: point["near_zero"]["h"]["percent"],
        y_value=lambda point: point["attention_mean"]["percent"],
        short_labels=False,
    )
    all_points = grouped["gelu"] + grouped["relu"]
    _finish_axis(
        ax,
        x_max=max(point["near_zero"]["h"]["percent"] for point in all_points),
        y_max=max(point["attention_mean"]["percent"] for point in all_points),
    )
    ax.set_xlabel("Near-zero mass at h, |x| <= 1e-2 (%)", labelpad=8)
    ax.set_ylabel("Mean near-zero mass at q_post, k_post, v (%)", labelpad=8)
    ax.set_title("Pythia-14M L1N pressure and attention spillover (epsilon=1e-2)", pad=12)
    ax.legend(frameon=False, loc="best")
    _footer(fig)
    return _save_figure(fig, FIGURE_PATHS["attention_mean"], plt)


def _figure_sitewise(grouped: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    plt = _setup_matplotlib()
    all_points = grouped["gelu"] + grouped["relu"]
    x_max = max(point["near_zero"]["h"]["percent"] for point in all_points)
    y_max = max(point["near_zero"][site]["percent"] for point in all_points for site in SITEWISE_Y)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.975, top=0.86, bottom=0.14, hspace=0.28, wspace=0.20)
    handles = []
    for ax, site in zip(axes.flat, SITEWISE_Y, strict=True):
        _draw_line(
            ax,
            grouped,
            x_value=lambda point: point["near_zero"]["h"]["percent"],
            y_value=lambda point, selected=site: point["near_zero"][selected]["percent"],
            short_labels=True,
        )
        _finish_axis(ax, x_max=x_max, y_max=y_max)
        ax.set_title(site)
        if site == "q_post":
            handles = ax.get_lines()
    for ax in axes[-1, :]:
        ax.set_xlabel("Near-zero mass at h (%)")
    for ax, site in zip(axes[:, 0], ("q_post", "v"), strict=True):
        ax.set_ylabel(f"Near-zero mass at {site} (%)")
    for ax, site in zip(axes[:, 1], ("k_post", "m"), strict=True):
        ax.set_ylabel(f"Near-zero mass at {site} (%)")
    fig.suptitle("Pythia-14M L1N sitewise near-zero mass (epsilon=1e-2)", y=0.955, fontsize=15)
    fig.legend(handles, ["GeLU", "ReLU"], loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False)
    _footer(fig)
    return _save_figure(fig, FIGURE_PATHS["sitewise"], plt)


def _figure_attention_output(grouped: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.22)
    _draw_line(
        ax,
        grouped,
        x_value=lambda point: point["near_zero"]["h"]["percent"],
        y_value=lambda point: point["near_zero"]["attention_output"]["percent"],
        short_labels=False,
    )
    all_points = grouped["gelu"] + grouped["relu"]
    _finish_axis(
        ax,
        x_max=max(point["near_zero"]["h"]["percent"] for point in all_points),
        y_max=max(point["near_zero"]["attention_output"]["percent"] for point in all_points),
    )
    ax.set_xlabel("Near-zero mass at h, |x| <= 1e-2 (%)", labelpad=8)
    ax.set_ylabel("Near-zero mass after W_o (%)", labelpad=8)
    ax.set_title("Pythia-14M attention output before residual addition (epsilon=1e-2)", pad=12)
    ax.legend(frameon=False, loc="best")
    _footer(fig)
    return _save_figure(fig, FIGURE_PATHS["attention_output"], plt)


def _write_data(
    kind: str,
    grouped: Mapping[str, list[Mapping[str, Any]]],
    figure: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    estimands = {
        "attention_mean": {"x": "count-pooled h near-zero percent", "y": "unweighted mean of separately count-pooled q_post, k_post, and v near-zero percents"},
        "sitewise": {"x": "count-pooled h near-zero percent", "y": "separately count-pooled q_post, k_post, v, and m near-zero percents"},
        "attention_output": {"x": "count-pooled h near-zero percent", "y": "count-pooled post-W_o/pre-residual attention-output near-zero percent"},
    }
    write_json(
        DATA_PATHS[kind],
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "epsilon": EPSILON,
            "estimands": estimands[kind],
            "series": grouped,
            "axes": {"x_starts_at_zero": True, "y_starts_at_zero": True, "shared_y_scale": kind == "sitewise"},
            "figure": dict(figure),
            "provenance": dict(provenance),
        },
    )


def _rows(grouped: Mapping[str, list[Mapping[str, Any]]], columns: tuple[str, ...]) -> str:
    lines = []
    for activation in ("gelu", "relu"):
        for point in grouped[activation]:
            values = []
            for column in columns:
                if column == "attention_mean":
                    values.append(point["attention_mean"]["percent"])
                else:
                    values.append(point["near_zero"][column]["percent"])
            lines.append(
                "| {activation} | {label} | {values} |".format(
                    activation="GeLU" if activation == "gelu" else "ReLU",
                    label=point["label"],
                    values=" | ".join(f"{value:.6f}" for value in values),
                )
            )
    return "\n".join(lines)


def _write_observations(grouped: Mapping[str, list[Mapping[str, Any]]]) -> None:
    common_method = """All coordinates are recalculated from previously stored integer hits and denominators at `abs(x) <= 1e-2`; no checkpoint rerun was required. Counts are pooled across all validation batches and six layers before division. Coverage is all 500 MiniPile validation documents, 338 complete 2,048-token blocks, 692,224 input tokens, and the declared 1,444-token excluded tail. `h`, `q_post`, `k_post`, and `v` come from the terminal diagnostic; `m` and the post-`W_o` attention output come from their verified full-validation post-hoc diagnostics. There is one seed and no condition averaging."""
    limitations = """These are threshold-dependent descriptive trajectories from one seed and a short training horizon. GeLU-versus-ReLU differences include the activation operator change. Near-zero activation mass is not an exact-zero product count, removable compute, measured speedup, a causal route, or a long-horizon optimum."""
    texts = {
        "attention_mean": f"""# O004 - L1N spillover at epsilon 1e-2

## Question

How does the Figure 01 relationship change when near-zero mass uses `abs(x) <= 1e-2`?

## Method and coverage

{common_method}

## Values

| Activation | Pressure label | h (%) | Attention mean (%) |
| --- | --- | ---: | ---: |
{_rows(grouped, ('h', 'attention_mean'))}

## Figure caption and encoding

**Figure 04. Epsilon-1e-2 version of Figure 01.** X is count-pooled `h` near-zero mass; Y is the unweighted mean of separately count-pooled `q_post`, `k_post`, and `v` fractions. Blue circles/solid lines denote GeLU and orange squares/dashed lines denote ReLU. Labels identify control and naive-L1 weight. Axes begin at zero.

## Interpretation and limits

{limitations}

## Provenance

- Numerical reduction: `../artifacts/figure_data_attention_mean_eps1e-2.json`
- Source script: `../10_plot_eps1e-2_figures.py` and `../eps1e2_figures.py`
- Output: `../figures/04-h-vs-attention-near-zero-eps1e-2.pdf`
""",
        "sitewise": f"""# O005 - Sitewise L1N trajectories at epsilon 1e-2

## Question

How does the Figure 02 sitewise view change when near-zero mass uses `abs(x) <= 1e-2`?

## Method and coverage

{common_method}

## Values

| Activation | Pressure label | h (%) | q_post (%) | k_post (%) | v (%) | m (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{_rows(grouped, ('h', 'q_post', 'k_post', 'v', 'm'))}

## Figure caption and encoding

**Figure 05. Epsilon-1e-2 version of Figure 02.** The 2x2 panels show count-pooled `q_post`, `k_post`, `v`, and `m` near-zero mass versus count-pooled `h`. Blue circles/solid lines denote GeLU and orange squares/dashed lines denote ReLU. Point labels identify control or naive-L1 weight. All panels use the same zero-based x and y scales.

## Interpretation and limits

{limitations}

## Provenance

- Numerical reduction: `../artifacts/figure_data_sitewise_eps1e-2.json`
- Source script: `../10_plot_eps1e-2_figures.py` and `../eps1e2_figures.py`
- Output: `../figures/05-h-vs-site-near-zero-grid-eps1e-2.pdf`
""",
        "attention_output": f"""# O006 - Attention-output trajectory at epsilon 1e-2

## Question

How does the Figure 03 post-`W_o`, pre-residual relationship change when near-zero mass uses `abs(x) <= 1e-2`?

## Method and coverage

{common_method}

## Values

| Activation | Pressure label | h (%) | Attention output (%) |
| --- | --- | ---: | ---: |
{_rows(grouped, ('h', 'attention_output'))}

## Figure caption and encoding

**Figure 06. Epsilon-1e-2 version of Figure 03.** X is count-pooled `h` near-zero mass and Y is count-pooled attention output immediately after `W_o`, equal under zero dropout to the tensor entering residual addition. Blue circles/solid lines denote GeLU and orange squares/dashed lines denote ReLU. Labels identify control and naive-L1 weight. Axes begin at zero.

## Interpretation and limits

{limitations}

## Provenance

- Numerical reduction: `../artifacts/figure_data_attention_output_eps1e-2.json`
- Source script: `../10_plot_eps1e-2_figures.py` and `../eps1e2_figures.py`
- Output: `../figures/06-h-vs-attention-output-near-zero-eps1e-2.pdf`
""",
    }
    for kind, text in texts.items():
        OBSERVATION_PATHS[kind].write_text(text, encoding="utf-8", newline="\n")


def generate_all() -> list[Path]:
    verification, _m_diagnostic, _output_diagnostic, grouped = _load_sources()
    if verification.get("evidence_label") != "valid":
        raise ValueError("Run 002 evidence is not valid.")
    provenance = _provenance()
    figures = {
        "attention_mean": _figure_attention_mean(grouped),
        "sitewise": _figure_sitewise(grouped),
        "attention_output": _figure_attention_output(grouped),
    }
    for kind, figure in figures.items():
        _write_data(kind, grouped, figure, provenance)
    _write_observations(grouped)
    write_json(
        COMPLETION_PATH,
        {
            "schema_version": 1,
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "epsilon": EPSILON,
            "figures": figures,
            "figure_data": {kind: path.relative_to(RUN_DIR).as_posix() for kind, path in DATA_PATHS.items()},
            "observations": {kind: path.relative_to(RUN_DIR).as_posix() for kind, path in OBSERVATION_PATHS.items()},
        },
    )
    return [FIGURE_PATHS[kind] for kind in ("attention_mean", "sitewise", "attention_output")]
