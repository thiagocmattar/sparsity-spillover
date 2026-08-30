"""Extend Analysis 004 with post-hoc clipping curves anchored to its controls."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ANALYSIS_DIR = Path(__file__).resolve().parent
SOURCE_004 = (
    ANALYSIS_DIR.parent
    / "004-2026-08-30-full-pass-quality-logical-frontier"
    / "comparison.json"
)
SOURCE_005 = ANALYSIS_DIR / "teal_frontier.json"
FIGURE_DATA = ANALYSIS_DIR / "figure_data_02.json"
OUTPUT = ANALYSIS_DIR / "figures" / "02-full-pass-frontier-with-posthoc-controls.pdf"

Y_MIN = 5.075
Y_MAX = 6.0
TRAINED_SERIES = ("a1h_naive_l1", "a1h_ol1", "a4z_threshold")
TRAINED_STYLES = {
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
        "label": "A4 (trained threshold)",
        "color": "#009E73",
        "marker": "D",
        "linestyle": "-.",
    },
}
POSTHOC_STYLES = {
    "gelu-control": {
        "label": "Post-hoc clipping on GeLU control",
        "color": "#CC79A7",
        "marker": "X",
        "linestyle": ":",
    },
    "relu-control": {
        "label": "Post-hoc clipping on ReLU control",
        "color": "#222222",
        "marker": "v",
        "linestyle": (0, (4.0, 1.5, 1.0, 1.5)),
    },
}
CONTROL_STYLES = {
    "gelu_control": {"label": "GeLU ctrl.", "marker": "P", "color": "#777777"},
    "relu_control": {"label": "ReLU ctrl.", "marker": "^", "color": "#222222"},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _logical_fraction(row: Mapping[str, Any]) -> float:
    logical = row["logical_products"]
    zero_count = sum(
        int(operation["zero_product_count"])
        for operation in logical["per_operation"].values()
    )
    if zero_count != int(logical["block_zero_product_count"]):
        raise ValueError("Post-hoc logical numerator does not reconcile.")
    fraction = zero_count / int(logical["model_product_count"])
    if not math.isclose(fraction, float(logical["R_model"]), rel_tol=0.0, abs_tol=1e-16):
        raise ValueError("Post-hoc R_model is not derived from integer counts.")
    return fraction


def _posthoc_point(row: Mapping[str, Any]) -> dict[str, Any]:
    coverage = row["validation"]
    expected_coverage = {
        "sequences": 338,
        "input_tokens": 692_224,
        "excluded_tail_tokens": 1_444,
        "complete_block_coverage": True,
    }
    mismatches = [
        key for key, expected in expected_coverage.items() if coverage.get(key) != expected
    ]
    if mismatches:
        raise ValueError("Post-hoc coverage mismatch: " + ", ".join(mismatches))
    return {
        "condition_id": row["condition_id"],
        "attempt_id": row["attempt_id"],
        "target_sparsity": float(row["target_sparsity"]),
        "R_model": _logical_fraction(row),
        "final_validation_loss": float(coverage["loss"]),
        "loss_delta_from_zero_threshold": float(row["loss_delta_from_zero_threshold"]),
    }


def build_figure_data() -> dict[str, Any]:
    analysis_004 = json.loads(SOURCE_004.read_text(encoding="utf-8"))
    analysis_005 = json.loads(SOURCE_005.read_text(encoding="utf-8"))
    if (
        analysis_004.get("schema_version") != 1
        or analysis_004.get("evidence_status") != "complete_verified_matched_cohorts"
        or len(analysis_004.get("conditions", [])) != 15
    ):
        raise ValueError("Analysis 004 is not complete verified 15-point evidence.")
    if (
        analysis_005.get("schema_version") != 1
        or analysis_005.get("status") != "complete_verified"
        or len(analysis_005.get("conditions", [])) != 20
    ):
        raise ValueError("Analysis 005 is not complete verified 20-point evidence.")

    trained = [dict(row) for row in analysis_004["conditions"]]
    controls = {
        row["condition_id"]: row
        for row in trained
        if row["condition_id"] in {"gelu-control", "relu-control"}
    }
    if set(controls) != {"gelu-control", "relu-control"}:
        raise ValueError("Analysis 004 controls are incomplete.")
    source_005 = {row["condition_id"]: row for row in analysis_005["sources"]}

    posthoc = {}
    expected_visible = {
        "gelu-control": [0.0, 0.1, 0.2, 0.3, 0.4],
        "relu-control": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    }
    for condition_id in ("gelu-control", "relu-control"):
        canonical = controls[condition_id]
        source = source_005.get(condition_id)
        if source is None or source["attempt_id"] != canonical["attempt_id"]:
            raise ValueError(f"Analysis 004/005 source attempt mismatch for {condition_id}.")
        rows = sorted(
            (
                _posthoc_point(row)
                for row in analysis_005["conditions"]
                if row["condition_id"] == condition_id
            ),
            key=lambda row: row["target_sparsity"],
        )
        if [row["target_sparsity"] for row in rows] != [index / 10 for index in range(10)]:
            raise ValueError(f"Incomplete post-hoc target grid for {condition_id}.")

        local_zero = rows[0]
        canonical_anchor = {
            "condition_id": condition_id,
            "attempt_id": canonical["attempt_id"],
            "target_sparsity": 0.0,
            "R_model": float(canonical["R_model"]),
            "final_validation_loss": float(canonical["final_validation_loss"]),
            "anchor_source": "Analysis 004 canonical control endpoint",
        }
        loss_difference = (
            local_zero["final_validation_loss"] - canonical_anchor["final_validation_loss"]
        )
        r_difference = local_zero["R_model"] - canonical_anchor["R_model"]
        if abs(loss_difference) > 5e-5 or abs(r_difference) > 1e-6:
            raise ValueError(f"Post-hoc zero row does not reproduce {condition_id}.")

        visible = [canonical_anchor] + [
            row
            for row in rows[1:]
            if row["final_validation_loss"] <= Y_MAX
        ]
        omitted = [
            row
            for row in rows[1:]
            if row["final_validation_loss"] > Y_MAX
        ]
        if [row["target_sparsity"] for row in visible] != expected_visible[condition_id]:
            raise ValueError(f"Unexpected visible post-hoc points for {condition_id}.")
        posthoc[condition_id] = {
            "anchor_reproduction": {
                "local_zero_loss_minus_canonical": loss_difference,
                "local_zero_R_model_minus_canonical": r_difference,
            },
            "visible_points": visible,
            "omitted_above_y_cap": omitted,
        }

    return {
        "schema_version": 1,
        "status": "complete_verified_figure_data",
        "question": "Where do trained full-pass endpoints and TEAL-style post-hoc clipping curves attached to the Run 004 GeLU/ReLU controls lie below validation loss 6?",
        "display": {
            "y_min": Y_MIN,
            "y_max": Y_MAX,
            "rule": "Post-hoc points above the explicit loss cap are omitted rather than clipped to the boundary.",
        },
        "sources": {
            "analysis_004": {
                "path": SOURCE_004.relative_to(ANALYSIS_DIR.parent.parent).as_posix(),
                "sha256": _sha256(SOURCE_004),
                "conditions": 15,
            },
            "analysis_005": {
                "path": SOURCE_005.relative_to(ANALYSIS_DIR.parent.parent).as_posix(),
                "sha256": _sha256(SOURCE_005),
                "conditions": 20,
            },
        },
        "coverage": {
            "documents": 500,
            "sequences": 338,
            "input_tokens": 692_224,
            "excluded_tail_tokens": 1_444,
            "seed_count": 1,
        },
        "trained_endpoints": trained,
        "posthoc_control_curves": posthoc,
    }


def _annotate(
    axis: Any,
    row: Mapping[str, Any],
    text: str,
    offset: tuple[float, float],
    color: str,
    *,
    fontsize: float = 7.8,
) -> None:
    axis.annotate(
        text,
        xy=(100.0 * float(row["R_model"]), float(row["final_validation_loss"])),
        xytext=offset,
        textcoords="offset points",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.11",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.84,
        },
        zorder=8,
    )


def _style_axis(axis: Any) -> None:
    axis.grid(True, color="#D8D8D8", linewidth=0.65, alpha=0.72)
    axis.set_axisbelow(True)
    axis.tick_params(direction="out", length=3.5, width=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")


def _plot(data: Mapping[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.3,
            "axes.labelsize": 10.3,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "pdf.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(11.4, 6.15))

    trained = data["trained_endpoints"]
    series_rows = {
        series_id: sorted(
            (row for row in trained if row["series_id"] == series_id),
            key=lambda row: float(row["dose"]),
        )
        for series_id in TRAINED_SERIES
    }
    for series_id in TRAINED_SERIES:
        rows = series_rows[series_id]
        style = TRAINED_STYLES[series_id]
        axis.plot(
            [100.0 * float(row["R_model"]) for row in rows],
            [float(row["final_validation_loss"]) for row in rows],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=6.6,
            markeredgecolor="white",
            markeredgewidth=0.8,
            alpha=0.92,
            zorder=4,
        )

    controls = {
        row["series_id"]: row
        for row in trained
        if row["series_id"] in CONTROL_STYLES
    }
    for condition_id, control_series in (
        ("gelu-control", "gelu_control"),
        ("relu-control", "relu_control"),
    ):
        points = data["posthoc_control_curves"][condition_id]["visible_points"]
        style = POSTHOC_STYLES[condition_id]
        axis.plot(
            [100.0 * float(row["R_model"]) for row in points],
            [float(row["final_validation_loss"]) for row in points],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            zorder=5,
        )
        nonzero = points[1:]
        axis.scatter(
            [100.0 * float(row["R_model"]) for row in nonzero],
            [float(row["final_validation_loss"]) for row in nonzero],
            color=style["color"],
            marker=style["marker"],
            s=54,
            edgecolor="white",
            linewidth=0.8,
            zorder=7,
        )
        control = controls[control_series]
        control_style = CONTROL_STYLES[control_series]
        axis.scatter(
            [100.0 * float(control["R_model"])],
            [float(control["final_validation_loss"])],
            color=control_style["color"],
            marker=control_style["marker"],
            s=69,
            edgecolor="white",
            linewidth=0.9,
            zorder=9,
        )

    threshold_offsets = {
        0.0: (-18, 12),
        0.01: (-17, -13),
        0.05: (7, 10),
        0.1: (7, -11),
        0.5: (-47, 11),
    }
    for row in series_rows["a4z_threshold"]:
        dose = float(row["dose"])
        _annotate(
            axis,
            row,
            rf"$\kappa={dose:g}$",
            threshold_offsets[dose],
            TRAINED_STYLES["a4z_threshold"]["color"],
        )

    pressure_offsets = {
        ("a1h_naive_l1", 0.05): (-42, 12),
        ("a1h_ol1", 0.05): (7, -10),
        ("a1h_naive_l1", 0.1): (-45, 10),
        ("a1h_ol1", 0.1): (7, -11),
        ("a1h_naive_l1", 0.5): (-47, -4),
        ("a1h_ol1", 0.5): (-47, 12),
        ("a1h_naive_l1", 1.0): (7, -8),
        ("a1h_ol1", 1.0): (7, 9),
    }
    for series_id in ("a1h_naive_l1", "a1h_ol1"):
        for row in series_rows[series_id]:
            dose = float(row["dose"])
            _annotate(
                axis,
                row,
                rf"$\lambda={dose:g}$",
                pressure_offsets[(series_id, dose)],
                TRAINED_STYLES[series_id]["color"],
            )

    posthoc_offsets = {
        ("gelu-control", 0.1): (-8, 11),
        ("gelu-control", 0.2): (-36, -13),
        ("gelu-control", 0.3): (-8, 11),
        ("gelu-control", 0.4): (-8, 11),
        ("relu-control", 0.1): (6, 11),
        ("relu-control", 0.2): (6, -12),
        ("relu-control", 0.3): (6, -12),
        ("relu-control", 0.4): (6, 11),
        ("relu-control", 0.5): (6, -13),
    }
    for condition_id in ("gelu-control", "relu-control"):
        for row in data["posthoc_control_curves"][condition_id]["visible_points"][1:]:
            target = float(row["target_sparsity"])
            _annotate(
                axis,
                row,
                rf"$p={target:.1f}$",
                posthoc_offsets[(condition_id, target)],
                POSTHOC_STYLES[condition_id]["color"],
                fontsize=7.6,
            )

    _annotate(axis, controls["gelu_control"], "GeLU ctrl.", (7, -12), "#666666")
    _annotate(axis, controls["relu_control"], "ReLU ctrl.", (-28, 15), "#222222")

    axis.set_xlim(-0.25, 10.65)
    axis.set_ylim(Y_MIN, Y_MAX)
    axis.set_xlabel(r"Measured $R_{\mathrm{model}}$ (%)")
    axis.set_ylabel("Final validation loss (lower is better)")
    _style_axis(axis)

    handles = [
        Line2D(
            [0],
            [0],
            color=TRAINED_STYLES[series_id]["color"],
            marker=TRAINED_STYLES[series_id]["marker"],
            linestyle=TRAINED_STYLES[series_id]["linestyle"],
            linewidth=1.8,
            markersize=6.4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=TRAINED_STYLES[series_id]["label"],
        )
        for series_id in TRAINED_SERIES
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            color=POSTHOC_STYLES[condition_id]["color"],
            marker=POSTHOC_STYLES[condition_id]["marker"],
            linestyle=POSTHOC_STYLES[condition_id]["linestyle"],
            linewidth=2.2,
            markersize=6.6,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=POSTHOC_STYLES[condition_id]["label"],
        )
        for condition_id in ("gelu-control", "relu-control")
    )

    figure.suptitle(
        "Pythia-14M: trained endpoints and post-hoc clipping on controls",
        x=0.5,
        y=0.976,
        fontsize=14.0,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.933,
        "Uniform TEAL-style clipping is applied only at evaluation; displayed validation loss is capped at 6",
        ha="center",
        va="center",
        fontsize=9.4,
        color="#444444",
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.898),
        ncol=3,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.15,
        labelspacing=0.75,
    )
    figure.text(
        0.5,
        0.027,
        "Post-hoc points above loss 6 are omitted by the explicit display cap (GeLU p >= 0.5; ReLU p >= 0.6).\n"
        r"Lines connect dose/target order only. $R_{\mathrm{model}}$ is exact-zero logical-product opportunity, not measured speedup; one seed.",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#444444",
        linespacing=1.32,
    )
    figure.subplots_adjust(left=0.082, right=0.988, top=0.76, bottom=0.16)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".pdf.tmp")
    figure.savefig(
        temporary,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": "Pythia-14M trained endpoints and post-hoc clipping on controls",
            "Author": "sparsity-spillover analysis 005",
            "Subject": "Measured R_model versus final validation loss capped at 6",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    temporary.replace(OUTPUT)
    print(OUTPUT)
    return OUTPUT


def main() -> Path:
    data = build_figure_data()
    _write_json(FIGURE_DATA, data)
    return _plot(data)


if __name__ == "__main__":
    main()
