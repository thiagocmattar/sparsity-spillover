"""Count-first Run 004 analogues of Run 002 sitewise spillover figures."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


RUN_DIR = Path(__file__).resolve().parent
ATTEMPTS_DIR = RUN_DIR / "artifacts" / "attempts"
VERIFICATION_PATH = RUN_DIR / "artifacts" / "verification.json"
SITEWISE_DATA_PATH = RUN_DIR / "artifacts" / "figure_data_sitewise.json"
ATTENTION_OUTPUT_DATA_PATH = RUN_DIR / "artifacts" / "figure_data_attention_output.json"
COMPLETION_PATH = RUN_DIR / "artifacts" / "spillover_figure_completion.json"
SITEWISE_FIGURE_PATH = RUN_DIR / "figures" / "01-h-vs-site-near-zero-grid.pdf"
ATTENTION_OUTPUT_FIGURE_PATH = (
    RUN_DIR / "figures" / "02-h-vs-attention-output-near-zero.pdf"
)
EPSILON = 0.001
SITES = ("q_post", "k_post", "v", "m")
ALL_SITES = ("h", *SITES, "attention_output")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pooled_counts(
    statistics: Mapping[str, Any], site: str, epsilon: float = EPSILON
) -> dict[str, int | float]:
    """Recompute one pooled fraction from integer hits and denominators."""
    key = f"{epsilon:g}"
    pooled_matches = [
        row for row in statistics.get("pooled_by_site", []) if row.get("name") == site
    ]
    if len(pooled_matches) != 1:
        raise ValueError(f"Expected one pooled row for site {site!r}.")
    pooled = pooled_matches[0]
    hits = int(pooled["threshold_hits"][key])
    total = int(pooled["total"])
    if total <= 0 or hits < 0 or hits > total or int(pooled.get("nonfinite", 0)) != 0:
        raise ValueError(f"Invalid pooled near-zero counts for site {site!r}.")

    layer_rows = [
        row
        for row in statistics.get("rows", [])
        if str(row.get("name", "")).startswith(f"{site}.layer_")
    ]
    if len(layer_rows) != 6:
        raise ValueError(f"Expected six layer rows for site {site!r}.")
    layer_hits = sum(int(row["threshold_hits"][key]) for row in layer_rows)
    layer_total = sum(int(row["total"]) for row in layer_rows)
    if any(int(row.get("nonfinite", 0)) != 0 for row in layer_rows):
        raise ValueError(f"Non-finite activations found for site {site!r}.")
    if (layer_hits, layer_total) != (hits, total):
        raise ValueError(f"Layer and pooled counts disagree for site {site!r}.")

    fraction = hits / total
    return {
        "hits": hits,
        "total": total,
        "fraction": fraction,
        "percent": 100.0 * fraction,
    }


def reduce_points(
    verification: Mapping[str, Any],
    statistics_for_attempt: Callable[[str], Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build the GeLU baseline and ordered ReLU dose trajectory."""
    if verification.get("status") != "verified" or int(
        verification.get("condition_count", -1)
    ) != 6:
        raise ValueError("Run 004 must be six-condition verified before plotting.")

    grouped: dict[str, list[dict[str, Any]]] = {"gelu": [], "relu": []}
    conditions = sorted(
        verification.get("conditions", []), key=lambda row: int(row["condition"]["order"])
    )
    for row in conditions:
        attempt_id = str(row["attempt_id"])
        condition = dict(row["condition"])
        activation = str(condition["activation"])
        if activation not in grouped:
            raise ValueError(f"Unexpected activation {activation!r}.")
        statistics = statistics_for_attempt(attempt_id)
        near_zero = {
            site: _pooled_counts(statistics, site, EPSILON) for site in ALL_SITES
        }
        grouped[activation].append(
            {
                "attempt_id": attempt_id,
                "condition_id": condition["id"],
                "activation": activation,
                "is_control": bool(condition["is_control"]),
                "label": "control"
                if bool(condition["is_control"])
                else f"lambda={float(condition['pressure_weight']):g}",
                "pressure_weight": float(condition["pressure_weight"]),
                "final_validation_loss": float(row["final_validation_loss"]),
                "near_zero": near_zero,
            }
        )

    if len(grouped["gelu"]) != 1 or len(grouped["relu"]) != 5:
        raise ValueError("Expected one GeLU control and five ordered ReLU points.")
    if not grouped["gelu"][0]["is_control"] or not grouped["relu"][0]["is_control"]:
        raise ValueError("Both activation series must begin with their controls.")
    relu_weights = [point["pressure_weight"] for point in grouped["relu"]]
    if relu_weights != [0.0, 0.05, 0.1, 0.5, 1.0]:
        raise ValueError(f"Unexpected ReLU pressure order: {relu_weights!r}.")
    return grouped


def load_points() -> dict[str, list[dict[str, Any]]]:
    verification = _read_json(VERIFICATION_PATH)

    def load_statistics(attempt_id: str) -> Mapping[str, Any]:
        path = ATTEMPTS_DIR / attempt_id / "diagnostics" / "activation_statistics.json"
        return _read_json(path)

    return reduce_points(verification, load_statistics)


def _configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
        }
    )


def _styles() -> dict[str, dict[str, Any]]:
    return {
        "gelu": {
            "display": "GeLU control",
            "color": "#0072B2",
            "marker": "o",
            "linestyle": "None",
        },
        "relu": {
            "display": "ReLU trajectory",
            "color": "#D55E00",
            "marker": "s",
            "linestyle": "--",
        },
    }


def _point_label(point: Mapping[str, Any], *, long: bool = False) -> str:
    if bool(point["is_control"]):
        return "control" if long else "ctrl"
    weight = float(point["pressure_weight"])
    return f"lambda={weight:g}" if long else f"{weight:g}"


def plot_sitewise(grouped: Mapping[str, list[dict[str, Any]]]) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    styles = _styles()
    offsets = {
        "q_post": {
            "gelu": [(5, -11)],
            "relu": [(0, -11), (0, 7), (-2, -11), (-2, 7), (0, -11)],
        },
        "k_post": {
            "gelu": [(5, -11)],
            "relu": [(0, -11), (0, 7), (-2, 7), (-2, -11), (0, 7)],
        },
        "v": {
            "gelu": [(5, -11)],
            "relu": [(0, -11), (0, 7), (-2, -11), (-2, 7), (0, -11)],
        },
        "m": {
            "gelu": [(5, 7)],
            "relu": [(0, -11), (0, 7), (-2, -11), (-2, 7), (0, -11)],
        },
    }
    all_points = grouped["gelu"] + grouped["relu"]
    x_max = max(float(point["near_zero"]["h"]["percent"]) for point in all_points)
    y_max = max(
        float(point["near_zero"][site]["percent"])
        for point in all_points
        for site in SITES
    )

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.10, right=0.975, top=0.86, bottom=0.14, hspace=0.28, wspace=0.20
    )
    legend_handles = []
    for ax, site in zip(axes.flat, SITES, strict=True):
        for activation in ("gelu", "relu"):
            points = grouped[activation]
            style = styles[activation]
            line = ax.plot(
                [float(point["near_zero"]["h"]["percent"]) for point in points],
                [float(point["near_zero"][site]["percent"]) for point in points],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.7,
                markersize=5.7,
                markerfacecolor="white",
                markeredgewidth=1.6,
                label=style["display"],
            )[0]
            if site == SITES[0]:
                legend_handles.append(line)
            for index, point in enumerate(points):
                dx, dy = offsets[site][activation][index]
                ax.annotate(
                    _point_label(point),
                    (
                        float(point["near_zero"]["h"]["percent"]),
                        float(point["near_zero"][site]["percent"]),
                    ),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha="left" if dx >= 5 else "center",
                    va="bottom" if dy > 0 else "top",
                    fontsize=7.6,
                    color="#202020",
                )
        ax.set_title(site)
        ax.set_xlim(0.0, max(1.0, x_max * 1.10))
        ax.set_ylim(0.0, max(0.01, y_max * 1.30))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.grid(axis="both", color="#D9D9D9", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Near-zero mass at h (%)")
    for ax, site in zip(axes[:, 0], ("q_post", "v"), strict=True):
        ax.set_ylabel(f"Near-zero mass at {site} (%)")
    for ax, site in zip(axes[:, 1], ("k_post", "m"), strict=True):
        ax.set_ylabel(f"Near-zero mass at {site} (%)")
    fig.suptitle(
        "Pythia-14M full-pass L1N sitewise near-zero mass", y=0.955, fontsize=15
    )
    fig.legend(
        legend_handles,
        [styles[activation]["display"] for activation in ("gelu", "relu")],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.5,
        0.035,
        "Seed 1234; |x| <= 1e-3; 712 updates; all 338 validation blocks; labels are control or lambda",
        ha="center",
        va="bottom",
        fontsize=8.1,
        color="#404040",
    )
    SITEWISE_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = SITEWISE_FIGURE_PATH.with_suffix(".tmp.pdf")
    fig.savefig(temporary, format="pdf", bbox_inches="tight")
    plt.close(fig)
    temporary.replace(SITEWISE_FIGURE_PATH)
    return SITEWISE_FIGURE_PATH


def plot_attention_output(grouped: Mapping[str, list[dict[str, Any]]]) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    styles = _styles()
    offsets = {
        "gelu": [(5, 7)],
        "relu": [(0, -11), (-2, 7), (-2, 7), (-2, 7), (0, -11)],
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.22)
    for activation in ("gelu", "relu"):
        points = grouped[activation]
        style = styles[activation]
        ax.plot(
            [float(point["near_zero"]["h"]["percent"]) for point in points],
            [
                float(point["near_zero"]["attention_output"]["percent"])
                for point in points
            ],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.8,
            label=style["display"],
        )
        for index, point in enumerate(points):
            dx, dy = offsets[activation][index]
            ax.annotate(
                _point_label(point, long=True),
                (
                    float(point["near_zero"]["h"]["percent"]),
                    float(point["near_zero"]["attention_output"]["percent"]),
                ),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx >= 5 else "center",
                va="bottom" if dy > 0 else "top",
                fontsize=8.2,
                color="#202020",
            )

    all_points = grouped["gelu"] + grouped["relu"]
    x_max = max(float(point["near_zero"]["h"]["percent"]) for point in all_points)
    y_max = max(
        float(point["near_zero"]["attention_output"]["percent"])
        for point in all_points
    )
    ax.set_xlim(0.0, max(1.0, x_max * 1.10))
    ax.set_ylim(0.0, max(0.01, y_max * 1.25))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.set_xlabel("Near-zero mass at h, |x| <= 1e-3 (%)", labelpad=8)
    ax.set_ylabel("Near-zero mass after W_o (%)", labelpad=8)
    ax.set_title("Pythia-14M full-pass attention output before residual addition", pad=12)
    ax.grid(axis="both", color="#D9D9D9", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.text(
        0.5,
        0.035,
        "Seed 1234; 712 updates; all 338 validation blocks; GeLU is a standalone control",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#404040",
    )
    ATTENTION_OUTPUT_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = ATTENTION_OUTPUT_FIGURE_PATH.with_suffix(".tmp.pdf")
    fig.savefig(temporary, format="pdf", bbox_inches="tight")
    plt.close(fig)
    temporary.replace(ATTENTION_OUTPUT_FIGURE_PATH)
    return ATTENTION_OUTPUT_FIGURE_PATH


def _figure_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(RUN_DIR).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def generate() -> tuple[Path, Path]:
    _configure_matplotlib()
    grouped = load_points()
    sitewise = plot_sitewise(grouped)
    attention_output = plot_attention_output(grouped)
    generated_at = datetime.now(timezone.utc).isoformat()
    common = {
        "schema_version": 1,
        "generated_at": generated_at,
        "epsilon": EPSILON,
        "coverage": {
            "documents": 500,
            "sequences": 338,
            "input_tokens": 692224,
            "excluded_tail_tokens": 1444,
            "layers": 6,
        },
        "pooling": "sum integer hits and totals across all batches and layers, then divide once",
        "series": grouped,
        "source_verification": "artifacts/verification.json",
        "source_diagnostics": "artifacts/attempts/*/diagnostics/activation_statistics.json",
    }
    _write_json(
        SITEWISE_DATA_PATH,
        {
            **common,
            "x_site": "h",
            "y_sites": list(SITES),
            "figure": _figure_record(sitewise),
        },
    )
    _write_json(
        ATTENTION_OUTPUT_DATA_PATH,
        {
            **common,
            "x_site": "h",
            "y_site": "attention_output",
            "site_definition": "output of attention.dense (W_o), before residual addition",
            "figure": _figure_record(attention_output),
        },
    )
    _write_json(
        COMPLETION_PATH,
        {
            "schema_version": 1,
            "status": "completed",
            "completed_at": generated_at,
            "epsilon": EPSILON,
            "source_script": {
                "path": Path(__file__).name,
                "sha256": _sha256(Path(__file__)),
            },
            "figures": {
                "sitewise": _figure_record(sitewise),
                "attention_output": _figure_record(attention_output),
            },
            "reduced_data": {
                "sitewise": SITEWISE_DATA_PATH.relative_to(RUN_DIR).as_posix(),
                "attention_output": ATTENTION_OUTPUT_DATA_PATH.relative_to(
                    RUN_DIR
                ).as_posix(),
            },
        },
    )
    return sitewise, attention_output


if __name__ == "__main__":
    for output in generate():
        print(output)
