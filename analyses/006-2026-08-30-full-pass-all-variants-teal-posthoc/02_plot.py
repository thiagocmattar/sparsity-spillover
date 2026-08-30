"""Plot all trained full-pass endpoints and their post-hoc TEAL trajectories."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_DIR = ANALYSIS_DIR.parents[1]
RESULT_PATH = ANALYSIS_DIR / "teal_all_variants.json"
COMPARISON_PATH = (
    REPO_DIR
    / "analyses"
    / "004-2026-08-30-full-pass-quality-logical-frontier"
    / "comparison.json"
)
FIGURE_DATA_PATH = ANALYSIS_DIR / "figure_data.json"
TABLE_PATH = ANALYSIS_DIR / "tables.md"
OUTPUT_PATH = ANALYSIS_DIR / "figures" / "01-all-full-pass-teal-frontiers.pdf"
TARGETS = tuple(index / 10 for index in range(10))
Y_MIN = 5.075
Y_MAX = 6.0

SERIES_STYLES = {
    "a1h_naive_l1": {
        "label": "A1-H naive L1 trained endpoints",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    "a1h_ol1": {
        "label": "A1-H OL1 trained endpoints",
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
    },
    "a4z_threshold": {
        "label": "A4-Z trained threshold endpoints",
        "color": "#009E73",
        "marker": "D",
        "linestyle": "-.",
    },
    "gelu_control": {
        "label": "GeLU control",
        "color": "#CC79A7",
        "marker": "P",
        "linestyle": ":",
    },
    "relu_control": {
        "label": "ReLU control",
        "color": "#222222",
        "marker": "^",
        "linestyle": (0, (4.0, 1.5, 1.0, 1.5)),
    },
}
TRAINED_SERIES = ("a1h_naive_l1", "a1h_ol1", "a4z_threshold")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_point(row: Mapping[str, Any]) -> None:
    coverage = row["validation"]
    expected = {
        "sequences": 338,
        "input_tokens": 692_224,
        "source_tokens": 693_668,
        "excluded_tail_tokens": 1_444,
        "complete_block_coverage": True,
    }
    mismatches = [key for key, value in expected.items() if coverage.get(key) != value]
    if mismatches:
        raise ValueError("Incomplete plotted validation coverage: " + ", ".join(mismatches))
    logical = row["logical_products"]
    operation_rows = logical["per_operation"].values()
    numerator = sum(int(item["zero_product_count"]) for item in operation_rows)
    denominator = sum(int(item["product_count"]) for item in logical["per_operation"].values())
    if numerator != int(logical["block_zero_product_count"]):
        raise ValueError("A plotted logical numerator does not reconcile.")
    if denominator != int(logical["block_product_count"]):
        raise ValueError("A plotted block denominator does not reconcile.")
    if numerator < 0 or numerator > denominator:
        raise ValueError("A plotted logical numerator is outside its denominator.")
    expected_r = numerator / int(logical["model_product_count"])
    if not math.isclose(expected_r, float(logical["R_model"]), rel_tol=0.0, abs_tol=1e-16):
        raise ValueError("A plotted R_model is not count-derived.")


def _nondominated(rows: list[Mapping[str, Any]]) -> list[bool]:
    flags = []
    for index, row in enumerate(rows):
        loss = float(row["final_validation_loss"])
        opportunity = float(row["R_model"])
        dominated = any(
            other_index != index
            and float(other["final_validation_loss"]) <= loss
            and float(other["R_model"]) >= opportunity
            and (
                float(other["final_validation_loss"]) < loss
                or float(other["R_model"]) > opportunity
            )
            for other_index, other in enumerate(rows)
        )
        flags.append(not dominated)
    return flags


def _frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row, flag in zip(rows, _nondominated(rows), strict=True) if flag]
    unique: dict[tuple[float, float], dict[str, Any]] = {}
    for row in selected:
        key = (float(row["R_model"]), float(row["final_validation_loss"]))
        unique.setdefault(key, row)
    return sorted(unique.values(), key=lambda row: (float(row["R_model"]), float(row["final_validation_loss"])))


def build_figure_data(
    result: Mapping[str, Any] | None = None,
    comparison: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = result or json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    comparison = comparison or json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    if result.get("schema_version") != 1 or result.get("status") != "complete_verified":
        raise ValueError("Analysis 006 is not complete verified evidence.")
    if (
        comparison.get("schema_version") != 1
        or comparison.get("evidence_status") != "complete_verified_matched_cohorts"
        or len(comparison.get("conditions", [])) != 15
    ):
        raise ValueError("Analysis 004 is not complete verified 15-endpoint evidence.")
    rows = result.get("conditions", [])
    if len(rows) != 150:
        raise ValueError("Analysis 006 must contain fifteen complete ten-target curves.")

    comparison_by_id = {row["condition_id"]: row for row in comparison["conditions"]}
    rows_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        _validate_point(row)
        rows_by_id.setdefault(str(row["condition_id"]), []).append(row)
    if set(rows_by_id) != set(comparison_by_id):
        raise ValueError("Analysis 004/006 condition sets differ.")

    curves = []
    visible_all: list[dict[str, Any]] = []
    for condition in comparison["conditions"]:
        condition_id = condition["condition_id"]
        selected = sorted(rows_by_id[condition_id], key=lambda row: float(row["target_sparsity"]))
        if tuple(float(row["target_sparsity"]) for row in selected) != TARGETS:
            raise ValueError(f"Incomplete target sweep for {condition_id}.")
        baseline = float(selected[0]["validation"]["loss"])
        points = []
        for index, row in enumerate(selected):
            expected_delta = float(row["validation"]["loss"]) - baseline
            if not math.isclose(
                expected_delta,
                float(row["loss_delta_from_zero_threshold"]),
                rel_tol=0.0,
                abs_tol=1e-14,
            ):
                raise ValueError(f"Unpaired loss delta for {condition_id}.")
            point = {
                "condition_id": condition_id,
                "series_id": condition["series_id"],
                "target_sparsity": float(row["target_sparsity"]),
                "R_model": float(row["logical_products"]["R_model"]),
                "final_validation_loss": float(row["validation"]["loss"]),
                "loss_delta_from_zero_threshold": float(
                    row["loss_delta_from_zero_threshold"]
                ),
                "evidence_origin": row["evidence_origin"],
            }
            if index == 0:
                point["evaluated_zero_R_model"] = point["R_model"]
                point["evaluated_zero_validation_loss"] = point["final_validation_loss"]
                point["R_model"] = float(condition["R_model"])
                point["final_validation_loss"] = float(condition["final_validation_loss"])
                point["anchor_source"] = "Analysis 004 canonical endpoint"
            points.append(point)
        visible = [point for point in points if point["final_validation_loss"] <= Y_MAX]
        omitted = [point for point in points if point["final_validation_loss"] > Y_MAX]
        visible_all.extend(visible)
        curves.append(
            {
                "condition_id": condition_id,
                "run": condition["run"],
                "series_id": condition["series_id"],
                "series_label": condition["series_label"],
                "topology": condition["topology"],
                "pressure_method": condition["pressure_method"],
                "dose_name": condition["dose_name"],
                "dose": condition["dose"],
                "visible_points": visible,
                "omitted_above_y_cap": omitted,
            }
        )
    return {
        "schema_version": 1,
        "status": "complete_verified_figure_data",
        "question": "How do trained A1-H/A4-Z endpoints and evaluation-only TEAL trajectories compare in one quality-opportunity space?",
        "display": {
            "y_min": Y_MIN,
            "y_max": Y_MAX,
            "rule": "Points above loss 6 are omitted rather than clipped to the boundary.",
            "layout": "single_panel",
        },
        "coverage": result["coverage"],
        "sources": {
            "analysis_004": {
                "path": COMPARISON_PATH.relative_to(REPO_DIR).as_posix(),
                "sha256": _sha256(COMPARISON_PATH),
            },
            "analysis_006": {
                "path": RESULT_PATH.relative_to(REPO_DIR).as_posix(),
                "sha256": _sha256(RESULT_PATH) if RESULT_PATH.exists() else "injected-test-data",
            },
        },
        "trained_endpoints": comparison["conditions"],
        "posthoc_curves": curves,
        "teal_augmented_nondominated_envelope": _frontier(visible_all),
        "counts": {
            "trained_endpoints": 15,
            "posthoc_curves": 15,
            "posthoc_points": 150,
            "visible_posthoc_points": len(visible_all),
            "omitted_posthoc_points": 150 - len(visible_all),
        },
    }


def _write_table(data: Mapping[str, Any]) -> None:
    envelope_keys = {
        (
            row["condition_id"],
            float(row["target_sparsity"]),
        )
        for row in data["teal_augmented_nondominated_envelope"]
    }
    lines = [
        "# Analysis 006 complete target-sparsity table",
        "",
        "TEAL is uniform, per-checkpoint, per-site-layer magnitude clipping applied only",
        "during evaluation. All points retain complete validation evidence; `visible` only",
        "describes the loss-6 figure cap.",
        "",
        "| source condition | trained family | trained dose | TEAL target | validation loss | loss delta | R_model (%) | visible | TEAL envelope |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | :---: |",
    ]
    for curve in data["posthoc_curves"]:
        points = curve["visible_points"] + curve["omitted_above_y_cap"]
        for point in sorted(points, key=lambda row: float(row["target_sparsity"])):
            key = (point["condition_id"], float(point["target_sparsity"]))
            dose = "-" if curve["dose"] is None else f"{float(curve['dose']):g}"
            lines.append(
                "| {condition} | {series} | {dose} | {target:.1f} | {loss:.6f} | "
                "{delta:+.6f} | {rmodel:.6f} | {visible} | {envelope} |".format(
                    condition=curve["condition_id"],
                    series=curve["series_label"],
                    dose=dose,
                    target=float(point["target_sparsity"]),
                    loss=float(point["final_validation_loss"]),
                    delta=float(point["loss_delta_from_zero_threshold"]),
                    rmodel=100.0 * float(point["R_model"]),
                    visible="yes" if float(point["final_validation_loss"]) <= Y_MAX else "no",
                    envelope="yes" if key in envelope_keys else "no",
                )
            )
    lines.extend(
        [
            "",
            "Every row covers 500 validation documents, 338 complete blocks, and 692,224",
            "input tokens; the 1,444-token tail is excluded. `R_model` is logical-product",
            "opportunity, not measured speedup. One seed.",
            "",
        ]
    )
    TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")


def _style_axis(axis: Any) -> None:
    axis.grid(True, color="#D8D8D8", linewidth=0.65, alpha=0.72)
    axis.set_axisbelow(True)
    axis.tick_params(direction="out", length=3.5, width=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")


def _annotate_anchor(axis: Any, row: Mapping[str, Any], text: str, offset: tuple[int, int]) -> None:
    style = SERIES_STYLES[row["series_id"]]
    axis.annotate(
        text,
        xy=(100.0 * float(row["R_model"]), float(row["final_validation_loss"])),
        xytext=offset,
        textcoords="offset points",
        fontsize=7.1,
        fontweight="bold",
        color=style["color"],
        ha="left",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.08",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
        },
        zorder=9,
    )


def _plot(data: Mapping[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.labelsize": 10.3,
            "legend.fontsize": 8.4,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "pdf.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(11.5, 6.3))
    endpoints = data["trained_endpoints"]

    for curve in data["posthoc_curves"]:
        points = curve["visible_points"]
        if not points:
            continue
        style = SERIES_STYLES[curve["series_id"]]
        control = curve["series_id"] in {"gelu_control", "relu_control"}
        axis.plot(
            [100.0 * float(row["R_model"]) for row in points],
            [float(row["final_validation_loss"]) for row in points],
            color=style["color"],
            linestyle=style["linestyle"] if control else ":",
            linewidth=2.0 if control else 1.05,
            alpha=0.9 if control else 0.38,
            zorder=3 if control else 2,
        )
        if len(points) > 1:
            axis.scatter(
                [100.0 * float(row["R_model"]) for row in points[1:]],
                [float(row["final_validation_loss"]) for row in points[1:]],
                color=style["color"],
                marker=".",
                s=24 if control else 15,
                alpha=0.9 if control else 0.52,
                zorder=4,
            )

    for series_id in TRAINED_SERIES:
        selected = sorted(
            (row for row in endpoints if row["series_id"] == series_id),
            key=lambda row: float(row["dose"]),
        )
        style = SERIES_STYLES[series_id]
        axis.plot(
            [100.0 * float(row["R_model"]) for row in selected],
            [float(row["final_validation_loss"]) for row in selected],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=7,
        )

    controls = [row for row in endpoints if row["series_id"] in {"gelu_control", "relu_control"}]
    for row in controls:
        style = SERIES_STYLES[row["series_id"]]
        axis.scatter(
            [100.0 * float(row["R_model"])],
            [float(row["final_validation_loss"])],
            color=style["color"],
            marker=style["marker"],
            s=70,
            edgecolor="white",
            linewidth=0.9,
            zorder=9,
        )

    envelope = data["teal_augmented_nondominated_envelope"]
    envelope_line = axis.plot(
        [100.0 * float(row["R_model"]) for row in envelope],
        [float(row["final_validation_loss"]) for row in envelope],
        color="#111111",
        linewidth=2.8,
        linestyle="-",
        zorder=6,
    )[0]
    envelope_line.set_path_effects(
        [path_effects.Stroke(linewidth=4.6, foreground="white", alpha=0.9), path_effects.Normal()]
    )

    annotation_offsets = {
        ("a1h_naive_l1", 0.05): (-46, 11),
        ("a1h_naive_l1", 0.1): (-47, -4),
        ("a1h_naive_l1", 0.5): (-45, -14),
        ("a1h_naive_l1", 1.0): (7, -8),
        ("a1h_ol1", 0.05): (7, 10),
        ("a1h_ol1", 0.1): (7, -10),
        ("a1h_ol1", 0.5): (7, 9),
        ("a1h_ol1", 1.0): (7, 9),
        ("a4z_threshold", 0.0): (-18, 12),
        ("a4z_threshold", 0.01): (-18, -12),
        ("a4z_threshold", 0.05): (7, 10),
        ("a4z_threshold", 0.1): (7, -11),
        ("a4z_threshold", 0.5): (-48, 11),
    }
    for row in endpoints:
        series_id = row["series_id"]
        if series_id in TRAINED_SERIES:
            symbol = r"$\lambda$" if row["dose_name"] == "lambda" else r"$\kappa$"
            _annotate_anchor(
                axis,
                row,
                f"{symbol}={float(row['dose']):g}",
                annotation_offsets[(series_id, float(row["dose"]))],
            )
    _annotate_anchor(axis, controls[0], "GeLU ctrl.", (7, -12))
    _annotate_anchor(axis, controls[1], "ReLU ctrl.", (-30, 15))

    all_visible = [
        point for curve in data["posthoc_curves"] for point in curve["visible_points"]
    ]
    x_max = max(100.0 * float(point["R_model"]) for point in all_visible)
    x_max = max(x_max, max(100.0 * float(row["R_model"]) for row in endpoints))
    axis.set_xlim(-0.25, x_max * 1.045 + 0.25)
    axis.set_ylim(Y_MIN, Y_MAX)
    axis.set_xlabel(r"Measured $R_{\mathrm{model}}$ (%)")
    axis.set_ylabel("Final validation loss (lower is better)")
    _style_axis(axis)

    handles = [
        Line2D(
            [0],
            [0],
            color=SERIES_STYLES[series_id]["color"],
            marker=SERIES_STYLES[series_id]["marker"],
            linestyle=SERIES_STYLES[series_id]["linestyle"],
            linewidth=2.0,
            markersize=6.4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=SERIES_STYLES[series_id]["label"],
        )
        for series_id in TRAINED_SERIES
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#777777",
                marker=".",
                linestyle=":",
                linewidth=1.2,
                label="Post-hoc TEAL trajectory from each checkpoint",
            ),
            Line2D(
                [0],
                [0],
                color="#111111",
                linestyle="-",
                linewidth=2.8,
                label="TEAL-augmented nondominated envelope",
            ),
        ]
    )
    figure.suptitle(
        "Pythia-14M: trained A1/A4 endpoints with post-hoc TEAL on every checkpoint",
        x=0.5,
        y=0.978,
        fontsize=13.6,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.938,
        "Uniform magnitude clipping is evaluation-only; displayed validation loss is capped at 6",
        ha="center",
        va="center",
        fontsize=9.3,
        color="#444444",
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.902),
        ncol=3,
        frameon=False,
        handlelength=2.7,
        columnspacing=1.1,
        labelspacing=0.7,
    )
    figure.text(
        0.5,
        0.024,
        f"Thin trajectories contain the complete p=0.0,...,0.9 sweep; {data['counts']['omitted_posthoc_points']} points above loss 6 are omitted only from this view.\n"
        r"Lines connect trained dose or TEAL target order only. $R_{\mathrm{model}}$ is exact-zero logical-product opportunity, not measured speedup; one seed.",
        ha="center",
        va="bottom",
        fontsize=7.9,
        color="#444444",
        linespacing=1.32,
    )
    figure.subplots_adjust(left=0.082, right=0.988, top=0.755, bottom=0.158)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(".pdf.tmp")
    figure.savefig(
        temporary,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": "Pythia-14M trained A1/A4 endpoints with post-hoc TEAL",
            "Author": "sparsity-spillover analysis 006",
            "Subject": "Complete TEAL target sweeps with displayed validation loss capped at 6",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    temporary.replace(OUTPUT_PATH)
    return OUTPUT_PATH


def main() -> Path:
    data = build_figure_data()
    _write_json(FIGURE_DATA_PATH, data)
    _write_table(data)
    output = _plot(data)
    print(output)
    return output


if __name__ == "__main__":
    main()
