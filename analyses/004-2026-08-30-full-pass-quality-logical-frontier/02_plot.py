"""Plot the combined full-pass quality versus logical-opportunity endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ANALYSIS_DIR = Path(__file__).resolve().parent
SOURCE = ANALYSIS_DIR / "comparison.json"
OUTPUT = ANALYSIS_DIR / "figures" / "01-r-model-vs-final-validation-loss.pdf"

SERIES_ORDER = ("a1h_naive_l1", "a1h_ol1", "a4z_threshold")
STYLES = {
    "a1h_naive_l1": {
        "label": "A1-H naive L1",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    "a1h_ol1": {
        "label": "A1-H OL1",
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
    },
    "a4z_threshold": {
        "label": "A4",
        "color": "#009E73",
        "marker": "D",
        "linestyle": "-.",
    },
}


def _load() -> dict[str, Any]:
    value = json.loads(SOURCE.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise RuntimeError("Unsupported comparison schema")
    if value.get("evidence_status") != "complete_verified_matched_cohorts":
        raise RuntimeError("Comparison is not complete verified evidence")
    if len(value.get("conditions", [])) != 15:
        raise RuntimeError("Expected fifteen unique endpoints")
    return value


def _series_rows(value: dict[str, Any], series_id: str) -> list[dict[str, Any]]:
    return sorted(
        (row for row in value["conditions"] if row["series_id"] == series_id),
        key=lambda row: float(row["dose"]),
    )


def _plot_series(ax: Any, rows: list[dict[str, Any]], series_id: str, label: bool) -> None:
    style = STYLES[series_id]
    ax.plot(
        [100.0 * float(row["R_model"]) for row in rows],
        [float(row["final_validation_loss"]) for row in rows],
        label=style["label"] if label else None,
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
        linewidth=2.0,
        markersize=7.0,
        markeredgecolor="white",
        markeredgewidth=0.9,
        zorder=4,
    )


def _annotate(
    ax: Any,
    row: dict[str, Any],
    text: str,
    offset: tuple[float, float],
    color: str,
) -> None:
    ax.annotate(
        text,
        xy=(100.0 * float(row["R_model"]), float(row["final_validation_loss"])),
        xytext=offset,
        textcoords="offset points",
        fontsize=8.2,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.14",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.86,
        },
        zorder=6,
    )


def _style_axis(ax: Any) -> None:
    ax.grid(True, color="#D8D8D8", linewidth=0.65, alpha=0.72)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3.5, width=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#555555")


def main() -> Path:
    value = _load()

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

    fig, ax = plt.subplots(figsize=(10.2, 5.65))

    all_series = {series_id: _series_rows(value, series_id) for series_id in SERIES_ORDER}
    for series_id in SERIES_ORDER:
        _plot_series(ax, all_series[series_id], series_id, label=True)

    controls = {
        row["series_id"]: row
        for row in value["conditions"]
        if row["series_id"] in {"gelu_control", "relu_control"}
    }
    control_styles = {
        "gelu_control": ("GeLU control", "P", "#777777"),
        "relu_control": ("A1-H ReLU control", "^", "#222222"),
    }
    for control_id, (_, marker, color) in control_styles.items():
        row = controls[control_id]
        ax.scatter(
            [100.0 * float(row["R_model"])],
            [float(row["final_validation_loss"])],
            marker=marker,
            color=color,
            s=61,
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
        )

    threshold_offsets = {
        0.0: (-17, -15),
        0.01: (8, 8),
        0.05: (7, 9),
        0.1: (8, -11),
        0.5: (-46, -15),
    }
    for row in all_series["a4z_threshold"]:
        dose = float(row["dose"])
        _annotate(
            ax,
            row,
            rf"$\kappa={dose:g}$",
            threshold_offsets[dose],
            STYLES["a4z_threshold"]["color"],
        )

    pressure_offsets = {
        ("a1h_naive_l1", 0.05): (-38, 12),
        ("a1h_ol1", 0.05): (7, -10),
        ("a1h_naive_l1", 0.1): (-42, 10),
        ("a1h_ol1", 0.1): (7, -11),
        ("a1h_naive_l1", 0.5): (-45, -4),
        ("a1h_ol1", 0.5): (-45, 12),
        ("a1h_naive_l1", 1.0): (7, -8),
        ("a1h_ol1", 1.0): (7, 9),
    }
    for series_id in ("a1h_naive_l1", "a1h_ol1"):
        for row in all_series[series_id]:
            dose = float(row["dose"])
            _annotate(
                ax,
                row,
                rf"$\lambda={dose:g}$",
                pressure_offsets[(series_id, dose)],
                STYLES[series_id]["color"],
            )

    _annotate(ax, controls["gelu_control"], "GeLU", (7, -7), "#666666")
    _annotate(ax, controls["relu_control"], "ReLU", (7, 7), "#222222")

    ax.set_xlim(-0.25, 10.65)
    ax.set_ylim(5.075, 5.69)
    ax.set_xlabel(r"Measured $R_{\mathrm{model}}$ (%)")
    ax.set_ylabel("Final validation loss (lower is better)")
    _style_axis(ax)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=STYLES[series_id]["color"],
            marker=STYLES[series_id]["marker"],
            linestyle=STYLES[series_id]["linestyle"],
            linewidth=2.0,
            markersize=6.8,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=STYLES[series_id]["label"],
        )
        for series_id in SERIES_ORDER
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="none",
                marker=marker,
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=7.0,
                label=label,
            )
            for label, marker, color in control_styles.values()
        ]
    )

    fig.suptitle(
        "Pythia-14M: validation quality versus logical sparsity opportunity",
        x=0.5,
        y=0.965,
        fontsize=14.0,
        fontweight="bold",
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=5,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.25,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.80, bottom=0.13)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".pdf.tmp")
    fig.savefig(
        temporary,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": "Pythia-14M validation quality versus logical sparsity opportunity",
            "Author": "sparsity-spillover analysis 004",
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
