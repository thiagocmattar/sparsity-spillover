"""Augment Analysis 005's frontier with verified Run 012 A4-OL1 endpoints."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent.parent
SOURCE_005 = (
    ANALYSIS_DIR.parent
    / "005-2026-08-30-run004-controls-teal-posthoc"
    / "figure_data_02.json"
)
RUN_012 = REPO_ROOT / "runs" / "012-2026-08-30-pythia14m-full-pass-a4-ol1"
RUN_012_VERIFICATION = RUN_012 / "artifacts" / "verification.json"
FIGURE_DATA = ANALYSIS_DIR / "figure_data.json"
TABLES = ANALYSIS_DIR / "tables.md"
OUTPUT = ANALYSIS_DIR / "figures" / "01-full-pass-frontier-with-a4-ol1.pdf"

Y_MIN = 5.075
Y_MAX = 6.0
SELECTED_SITES = ("a", "m", "h", "z")
BASE_TRAINED_SERIES = ("a1h_naive_l1", "a1h_ol1", "a4z_threshold")
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
    "a4z_ol1": {
        "label": "A4-OL1 (trained)",
        "color": "#6F4C9B",
        "marker": "h",
        "linestyle": "--",
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
    "gelu_control": {"marker": "P", "color": "#777777"},
    "relu_control": {"marker": "^", "color": "#222222"},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _close(actual: float, expected: float, *, tolerance: float = 1e-15) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"Numeric mismatch: {actual!r} != {expected!r}.")


def _validate_base(data: Mapping[str, Any]) -> None:
    expected_coverage = {
        "documents": 500,
        "sequences": 338,
        "input_tokens": 692_224,
        "excluded_tail_tokens": 1_444,
        "seed_count": 1,
    }
    if data.get("schema_version") != 1:
        raise ValueError("Analysis 005 figure data has an unexpected schema.")
    if data.get("status") != "complete_verified_figure_data":
        raise ValueError("Analysis 005 figure data is not complete verified evidence.")
    if data.get("coverage") != expected_coverage:
        raise ValueError("Analysis 005 coverage is not the complete validation contract.")
    trained = data.get("trained_endpoints", [])
    if len(trained) != 15:
        raise ValueError("Analysis 005 does not contain the expected 15 trained endpoints.")
    series = {row["series_id"] for row in trained}
    expected_series = {
        "a1h_naive_l1",
        "a1h_ol1",
        "a4z_threshold",
        "gelu_control",
        "relu_control",
    }
    if series != expected_series:
        raise ValueError("Analysis 005 trained series changed unexpectedly.")
    if set(data.get("posthoc_control_curves", {})) != {
        "gelu-control",
        "relu-control",
    }:
        raise ValueError("Analysis 005 post-hoc control curves are incomplete.")
    display = data.get("display", {})
    if display.get("y_min") != Y_MIN or display.get("y_max") != Y_MAX:
        raise ValueError("Analysis 005 display cap changed unexpectedly.")


def _logical_counts(path: Path, expected_loss: float, expected_r_model: float) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    coverage = payload["coverage"]
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
        raise ValueError("Run 012 logical-product coverage mismatch: " + ", ".join(mismatches))

    measured = payload["measured"]
    operation_zero_count = sum(
        int(row["zero_product_count"])
        for row in measured["per_operation"].values()
    )
    block_zero_count = int(measured["block_zero_product_count"])
    model_product_count = int(measured["model_product_count"])
    if operation_zero_count != block_zero_count:
        raise ValueError("Run 012 logical-product operation counts do not reconcile.")
    r_model = block_zero_count / model_product_count
    _close(r_model, float(measured["R_model"]), tolerance=1e-16)
    _close(r_model, expected_r_model, tolerance=1e-16)
    # The diagnostic and manifest losses can differ at the 1e-6 level because the
    # logical pass is independently accumulated. Both must still describe the same run.
    if abs(float(coverage["loss"]) - expected_loss) > 5e-5:
        raise ValueError("Run 012 logical-product loss does not reproduce validation loss.")
    return {
        "zero_product_count": block_zero_count,
        "model_product_count": model_product_count,
    }


def _site_counts(
    path: Path,
    expected_fractions: Mapping[str, Any],
) -> dict[str, dict[str, int | float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pooled = {row["name"]: row for row in payload["pooled_by_site"]}
    result: dict[str, dict[str, int | float]] = {}
    for site in SELECTED_SITES:
        row = pooled.get(site)
        if row is None:
            raise ValueError(f"Run 012 activation statistics omit site {site!r}.")
        zero_count = int(row["exact_zero_count"])
        total_count = int(row["total"])
        if int(row["finite"]) != total_count or int(row["nonfinite"]) != 0:
            raise ValueError(f"Run 012 site {site!r} contains non-finite activations.")
        if not 0 <= zero_count <= total_count or total_count <= 0:
            raise ValueError(f"Run 012 site {site!r} has invalid integer counts.")
        fraction = zero_count / total_count
        _close(fraction, float(row["exact_zero_fraction"]))
        _close(fraction, float(expected_fractions[site]))
        result[site] = {
            "exact_zero_count": zero_count,
            "total_count": total_count,
            "exact_zero_fraction": fraction,
        }
    return result


def _build_a4_ol1(verification: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if (
        verification.get("schema_version") != 1
        or verification.get("status") != "verified"
        or verification.get("evidence_label") != "valid"
        or verification.get("condition_count") != 5
        or verification.get("completed_optimizer_steps") != 3_560
        or verification.get("complete_validation_passes") != 20
    ):
        raise ValueError("Run 012 verification is not the expected complete five-condition evidence.")
    expected_kappas = [0.0, 0.01, 0.05, 0.1, 0.5]
    conditions = sorted(verification["conditions"], key=lambda row: row["condition"]["order"])
    if [float(row["condition"]["gate_threshold"]) for row in conditions] != expected_kappas:
        raise ValueError("Run 012 kappa grid changed unexpectedly.")

    sources: dict[str, Any] = {}
    endpoints: list[dict[str, Any]] = []
    for row in conditions:
        condition = row["condition"]
        if (
            condition["topology_id"] != "A4-Z"
            or condition["gate_operator"] != "one_sided_threshold"
            or condition["pressure_method"] != "orthogonal_l1"
            or float(condition["pressure_weight"]) != 1.0
            or tuple(condition["active_sites"]) != SELECTED_SITES
            or tuple(condition["pressure_sites"]) != SELECTED_SITES
            or int(row["completed_steps"]) != 712
        ):
            raise ValueError(f"Run 012 condition {condition['id']} violates the matched design.")
        attempt_dir = RUN_012 / "artifacts" / "attempts" / row["attempt_id"]
        manifest_path = attempt_dir / "manifest.json"
        activation_path = attempt_dir / "diagnostics" / "activation_statistics.json"
        logical_path = attempt_dir / "diagnostics" / "logical_products.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation = manifest["validation_coverage"]
        expected_coverage = {
            "sequences": 338,
            "input_tokens": 692_224,
            "excluded_tail_tokens": 1_444,
            "complete_block_coverage": True,
        }
        mismatches = [
            key for key, expected in expected_coverage.items() if validation.get(key) != expected
        ]
        if mismatches:
            raise ValueError("Run 012 manifest coverage mismatch: " + ", ".join(mismatches))
        loss = float(row["final_validation_loss"])
        _close(float(validation["loss"]), loss)
        r_model = float(row["R_model"])
        logical_counts = _logical_counts(logical_path, loss, r_model)
        site_counts = _site_counts(
            activation_path,
            row["selected_site_exact_zero_fractions"],
        )
        endpoint = {
            "series_id": "a4z_ol1",
            "series_label": "A4-Z plus four-site OL1",
            "run": "run012",
            "condition_id": condition["id"],
            "attempt_id": row["attempt_id"],
            "dose_name": "kappa",
            "dose": float(condition["gate_threshold"]),
            "lambda": float(condition["pressure_weight"]),
            "topology": condition["topology_id"],
            "activation": condition["gate_operator"],
            "pressure_method": condition["pressure_method"],
            "final_validation_loss": loss,
            "R_block": float(row["R_block"]),
            "R_model": r_model,
            "R_model_max": float(row["R_model_max"]),
            "logical_product_counts": logical_counts,
            "site_exact_zero": site_counts,
            "site_exact_zero_fraction": {
                site: float(site_counts[site]["exact_zero_fraction"])
                for site in SELECTED_SITES
            },
        }
        endpoints.append(endpoint)
        sources[condition["id"]] = {
            "attempt_id": row["attempt_id"],
            "manifest": {"path": _repo_path(manifest_path), "sha256": _sha256(manifest_path)},
            "activation_statistics": {
                "path": _repo_path(activation_path),
                "sha256": _sha256(activation_path),
            },
            "logical_products": {
                "path": _repo_path(logical_path),
                "sha256": _sha256(logical_path),
            },
        }
    return endpoints, sources


def build_figure_data() -> dict[str, Any]:
    base = json.loads(SOURCE_005.read_text(encoding="utf-8"))
    verification = json.loads(RUN_012_VERIFICATION.read_text(encoding="utf-8"))
    _validate_base(base)
    a4_ol1, attempt_sources = _build_a4_ol1(verification)
    comparisons = verification["comparison"]["matched_conditions"]
    if len(comparisons) != 5:
        raise ValueError("Run 012 matched A4 comparison is incomplete.")
    return {
        "schema_version": 1,
        "status": "complete_verified_figure_data",
        "question": "Where do the five full-pass A4-OL1 endpoints lie relative to the trained and post-hoc frontiers in Analysis 005?",
        "display": dict(base["display"]),
        "sources": {
            "analysis_005_figure_data": {
                "path": _repo_path(SOURCE_005),
                "sha256": _sha256(SOURCE_005),
            },
            "run_012_verification": {
                "path": _repo_path(RUN_012_VERIFICATION),
                "sha256": _sha256(RUN_012_VERIFICATION),
            },
            "run_012_attempts": attempt_sources,
        },
        "coverage": dict(base["coverage"]),
        "trained_endpoints": [dict(row) for row in base["trained_endpoints"]] + a4_ol1,
        "posthoc_control_curves": base["posthoc_control_curves"],
        "a4_ol1_matched_a4_comparison": comparisons,
        "interpretation": {
            "logical_products": "Exact-zero logical-product opportunities, not removed FLOPs or measured speedup.",
            "site_reduction": "Pool integer exact-zero counts and totals over all layers and validation batches before dividing.",
            "scope": "One seed, one Pythia-14M scale, one MiniPile training pass, and all 338 complete validation blocks.",
        },
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.3f}"


def table_markdown(data: Mapping[str, Any]) -> str:
    rows = sorted(
        (row for row in data["trained_endpoints"] if row["series_id"] == "a4z_ol1"),
        key=lambda row: float(row["dose"]),
    )
    lines = [
        "# A4-OL1 full-validation results",
        "",
        "| kappa | Validation loss | R_model (%) | a zero (%) | m zero (%) | h zero (%) | z zero (%) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        site = row["site_exact_zero_fraction"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{float(row['dose']):g}",
                    f"{float(row['final_validation_loss']):.6f}",
                    _percent(float(row["R_model"])),
                    _percent(float(site["a"])),
                    _percent(float(site["m"])),
                    _percent(float(site["h"])),
                    _percent(float(site["z"])),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "All percentages are count-pooled over the complete 338-block validation pass (692,224 input tokens; 1,444-token excluded tail).",
            "Per-site zero mass is exact-zero activation mass. `R_model` is measured exact-zero logical-product opportunity, not runtime speedup.",
            "The five endpoints share seed 1, 712 optimizer steps, A4-Z topology, one-sided threshold gates, and four-site OL1 pressure with lambda=1.",
            "",
        ]
    )
    return "\n".join(lines)


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


def _series_rows(
    trained: Sequence[Mapping[str, Any]], series_id: str
) -> list[Mapping[str, Any]]:
    return sorted(
        (row for row in trained if row["series_id"] == series_id),
        key=lambda row: float(row.get("dose", 0.0)),
    )


def plot(data: Mapping[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.3,
            "axes.labelsize": 10.3,
            "legend.fontsize": 8.6,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "pdf.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(11.4, 6.3))
    trained = data["trained_endpoints"]
    plotted_series = BASE_TRAINED_SERIES + ("a4z_ol1",)
    series_rows = {series_id: _series_rows(trained, series_id) for series_id in plotted_series}
    for series_id in plotted_series:
        rows = series_rows[series_id]
        style = TRAINED_STYLES[series_id]
        axis.plot(
            [100.0 * float(row["R_model"]) for row in rows],
            [float(row["final_validation_loss"]) for row in rows],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0 if series_id == "a4z_ol1" else 1.8,
            markersize=7.1 if series_id == "a4z_ol1" else 6.6,
            markeredgecolor="white",
            markeredgewidth=0.8,
            alpha=0.94,
            zorder=6 if series_id == "a4z_ol1" else 4,
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
        axis.scatter(
            [100.0 * float(control["R_model"])],
            [float(control["final_validation_loss"])],
            color=CONTROL_STYLES[control_series]["color"],
            marker=CONTROL_STYLES[control_series]["marker"],
            s=69,
            edgecolor="white",
            linewidth=0.9,
            zorder=9,
        )

    a4_offsets = {
        0.0: (-18, 12),
        0.01: (-17, -13),
        0.05: (7, 10),
        0.1: (7, -11),
        0.5: (-54, -13),
    }
    for row in series_rows["a4z_threshold"]:
        dose = float(row["dose"])
        _annotate(
            axis,
            row,
            rf"$\kappa={dose:g}$",
            a4_offsets[dose],
            TRAINED_STYLES["a4z_threshold"]["color"],
        )

    ol1_offsets = {
        0.0: (-43, -15),
        0.01: (-29, 12),
        0.05: (7, -13),
        0.1: (7, 12),
        0.5: (-54, 14),
    }
    for row in series_rows["a4z_ol1"]:
        dose = float(row["dose"])
        _annotate(
            axis,
            row,
            rf"$\kappa={dose:g}$",
            ol1_offsets[dose],
            TRAINED_STYLES["a4z_ol1"]["color"],
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
            linewidth=2.0 if series_id == "a4z_ol1" else 1.8,
            markersize=6.6,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=TRAINED_STYLES[series_id]["label"],
        )
        for series_id in plotted_series
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
        "Pythia-14M: trained frontiers and post-hoc clipping on controls",
        x=0.5,
        y=0.978,
        fontsize=14.0,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.938,
        "A4-OL1 adds four-site orthogonal L1 pressure (lambda=1); displayed validation loss is capped at 6",
        ha="center",
        va="center",
        fontsize=9.4,
        color="#444444",
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=3,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.15,
        labelspacing=0.75,
    )
    figure.text(
        0.5,
        0.024,
        "Post-hoc points above loss 6 are omitted by the explicit display cap (GeLU p >= 0.5; ReLU p >= 0.6).\n"
        r"Lines connect dose/target order only. $R_{\mathrm{model}}$ is exact-zero logical-product opportunity, not measured speedup; one seed.",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#444444",
        linespacing=1.32,
    )
    figure.subplots_adjust(left=0.082, right=0.988, top=0.755, bottom=0.158)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".pdf.tmp")
    figure.savefig(
        temporary,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": "Pythia-14M trained frontiers with A4-OL1 and post-hoc controls",
            "Author": "sparsity-spillover analysis 007",
            "Subject": "Measured R_model versus final validation loss capped at 6",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    temporary.replace(OUTPUT)
    return OUTPUT


def main() -> Path:
    data = build_figure_data()
    _write_json(FIGURE_DATA, data)
    _write_text(TABLES, table_markdown(data))
    output = plot(data)
    print(output)
    return output


if __name__ == "__main__":
    main()
