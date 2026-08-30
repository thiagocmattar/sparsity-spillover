"""Plot measured R_model against final validation loss for Analysis 003."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ANALYSIS_DIR = Path(__file__).resolve().parent
SOURCE = ANALYSIS_DIR / "comparison.json"
OUTPUT = ANALYSIS_DIR / "figures/01-r-model-vs-final-validation-loss.pdf"
LAMBDAS = (0.05, 0.1, 0.5, 1.0)


def _load() -> dict[str, Any]:
    value = json.loads(SOURCE.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise RuntimeError("Unsupported comparison schema")
    if value.get("evidence_status") != "complete_verified_matched_cohorts":
        raise RuntimeError("Comparison is not complete verified evidence")
    if len(value.get("conditions", [])) != 10:
        raise RuntimeError("Expected ten endpoints")
    return value


def _method_rows(value: dict[str, Any], method: str) -> list[dict[str, Any]]:
    rows = sorted(
        (
            row
            for row in value["conditions"]
            if row["pressure_method"] == method and row["lambda"] is not None
        ),
        key=lambda row: float(row["lambda"]),
    )
    if tuple(float(row["lambda"]) for row in rows) != LAMBDAS:
        raise RuntimeError(f"Unexpected lambda grid for {method}")
    return rows


def main() -> Path:
    value = _load()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9.5,
            "pdf.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=False)
    styles = {
        "l1_naive": {
            "label": "Run 004 naive L1",
            "color": "#0072B2",
            "marker": "o",
            "linestyle": "-",
        },
        "orthogonal_l1": {
            "label": "Run 009 OL1",
            "color": "#D55E00",
            "marker": "s",
            "linestyle": "--",
        },
    }
    label_offsets = {
        ("l1_naive", 0.05): (-70, 8),
        ("orthogonal_l1", 0.05): (10, -9),
        ("l1_naive", 0.1): (-70, -11),
        ("orthogonal_l1", 0.1): (10, 8),
        ("l1_naive", 0.5): (-70, 10),
        ("orthogonal_l1", 0.5): (-75, -17),
        ("l1_naive", 1.0): (10, -6),
        ("orthogonal_l1", 1.0): (10, -15),
    }

    for method, style in styles.items():
        rows = _method_rows(value, method)
        x = [100.0 * float(row["R_model"]) for row in rows]
        y = [float(row["final_validation_loss"]) for row in rows]
        ax.plot(
            x,
            y,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=7.2,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )
        for row, point_x, point_y in zip(rows, x, y, strict=True):
            pressure_weight = float(row["lambda"])
            dx, dy = label_offsets[(method, pressure_weight)]
            ax.annotate(
                f"lambda={pressure_weight:g}",
                xy=(point_x, point_y),
                xytext=(dx, dy),
                textcoords="offset points",
                color=style["color"],
                fontsize=8.7,
                fontweight="bold",
                bbox={
                    "boxstyle": "square,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": style["color"],
                    "alpha": 0.55,
                    "linewidth": 0.7,
                },
                zorder=4,
            )

    controls = {
        row["condition_id"]: row
        for row in value["conditions"]
        if row["run"] == "run004" and row["lambda"] is None
    }
    control_styles = {
        "gelu-control": ("GeLU control", "D", "#666666", (8, -5)),
        "relu-control": ("ReLU control", "^", "#111111", (8, -5)),
    }
    for condition_id, (label, marker, color, offset) in control_styles.items():
        row = controls[condition_id]
        point_x = 100.0 * float(row["R_model"])
        point_y = float(row["final_validation_loss"])
        ax.scatter(
            [point_x],
            [point_y],
            label=label,
            marker=marker,
            color=color,
            s=56,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            label,
            xy=(point_x, point_y),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.7,
            color=color,
            va="center",
        )

    ax.set_xlim(-0.15, 4.35)
    ax.set_ylim(5.09, 5.29)
    ax.set_xlabel(r"Measured $R_{model}$ (%) - higher logical opportunity to the right")
    ax.set_ylabel("Final validation loss (lower is better)")
    ax.set_title(
        "Pythia-14M full-pass endpoints: naive L1 versus OL1\n"
        "One seed; all 338 complete MiniPile validation blocks",
        pad=10,
    )
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", ncols=2, frameon=False, handlelength=2.6)
    fig.text(
        0.5,
        0.018,
        "Endpoint detail: the validation-loss axis does not start at zero. "
        "R_model is logical-product opportunity, not measured speedup.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.86, bottom=0.17)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".pdf.tmp")
    fig.savefig(
        temporary,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": "Run 004 naive L1 versus Run 009 OL1",
            "Author": "sparsity-spillover analysis 003",
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
