"""Compare A4, historical h-only, and corrected four-site A4-OL1 endpoints."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent.parent
RUN_011 = REPO_ROOT / "runs" / "011-2026-08-30-pythia14m-full-pass-a4z"
RUN_012 = REPO_ROOT / "runs" / "012-2026-08-30-pythia14m-full-pass-a4-ol1"
RUN_015 = REPO_ROOT / "runs" / "015-2026-08-31-pythia14m-corrected-a4-ol1"
RUN_004_TRAINING = (
    REPO_ROOT / "runs" / "004-2026-08-29-pythia14m-full-pass-l1n" / "training.py"
)
FIGURE_DATA = ANALYSIS_DIR / "figure_data.json"
OUTPUT = ANALYSIS_DIR / "figures" / "01-rmodel-vs-validation-loss.pdf"

EXPECTED_KAPPAS = (0.0, 0.01, 0.05, 0.1, 0.5)
ACTIVE_SITES = ("a", "m", "h", "z")
ALL_SITES = (
    "a",
    "m",
    "h",
    "z",
    "q_post",
    "k_post",
    "v",
    "attention_output",
)
EXPECTED_CAPTURE_HASH = (
    "8277a447cefd5c4a91533b9fe0dd59fddd0c7103e8b8cf984b76a16e0252add1"
)
MAX_DIAGNOSTIC_LOSS_DELTA = 2e-4
MATCHED_MANIFEST_FIELDS = (
    "model",
    "data",
    "recipe",
    "seeds",
    "topology",
    "condition",
    "activation_pressure",
)
MATCHED_GATE_FIELDS = ("model", "data", "recipe", "seeds", "topology")
EXPECTED_COVERAGE = {
    "documents": 500,
    "sequences": 338,
    "input_tokens": 692_224,
    "excluded_tail_tokens": 1_444,
    "seed_count": 1,
}
SERIES = {
    "run011_a4": {
        "label": "A4 without OL1 (Run 011)",
        "short_label": "No OL1",
        "color": "#009E73",
        "marker": "D",
        "linestyle": "-.",
        "realized_pressure_sites": (),
    },
    "run012_h_only": {
        "label": "A4 gates + OL1@h (Run 012)",
        "short_label": "OL1@h",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
        "realized_pressure_sites": ("h",),
    },
    "run015_four_site": {
        "label": "A4 gates + four-site OL1 (Run 015)",
        "short_label": "OL1@{a,m,h,z}",
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
        "realized_pressure_sites": ACTIVE_SITES,
    },
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _close(
    actual: float,
    expected: float,
    *,
    tolerance: float = 1e-15,
    label: str = "value",
) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}.")


def _validate_run012_realization() -> dict[str, Any]:
    wrapper_path = RUN_012 / "training.py"
    wrapper = wrapper_path.read_text(encoding="utf-8")
    expected_wrapper = (
        'load_run004_module("_run012_frozen_run004_training", "training.py")'
    )
    if wrapper.count(expected_wrapper) != 1:
        raise ValueError("Run 012 no longer has the audited frozen Run 004 training link.")

    source = RUN_004_TRAINING.read_text(encoding="utf-8")
    h_only_capture = (
        'capture_context = ActivationCapture(model, ["h"], torch=torch) '
        "if pressure.enabled else nullcontext(None)"
    )
    if source.count(h_only_capture) != 1:
        raise ValueError("Run 012's inherited h-only capture line changed unexpectedly.")
    return {
        "declared_pressure_sites": list(ACTIVE_SITES),
        "realized_pressure_sites": ["h"],
        "wrapper": _repo_path(wrapper_path),
        "wrapper_sha256": _sha256(wrapper_path),
        "inherited_source": _repo_path(RUN_004_TRAINING),
        "inherited_source_sha256": _sha256(RUN_004_TRAINING),
        "capture_expression": h_only_capture,
    }


def _validate_run015_realization(verification: Mapping[str, Any]) -> dict[str, Any]:
    correction = verification.get("correction", {})
    expected = {
        "corrected_pressure_sites": list(ACTIVE_SITES),
        "historical_realized_objective": "A4-Z gates plus OL1 pressure on h only",
        "pressure_capture_names_sha256": EXPECTED_CAPTURE_HASH,
        "pressure_capture_tensor_count_per_microbatch": 24,
        "supersedes_declared_objective_of": (
            "runs/012-2026-08-30-pythia14m-full-pass-a4-ol1"
        ),
    }
    if correction != expected:
        raise ValueError("Run 015 correction metadata changed unexpectedly.")
    for row in verification.get("conditions", []):
        ol1 = row.get("ol1", {})
        if ol1.get("pressure_capture_tensor_count") != 24:
            raise ValueError("Run 015 did not verify 24 pressure tensors per boundary.")
        if ol1.get("pressure_capture_names_sha256") != EXPECTED_CAPTURE_HASH:
            raise ValueError("Run 015 pressure-capture identity does not match four sites.")
        if ol1.get("boundary_count") != 712:
            raise ValueError("Run 015 pressure-capture evidence is not boundary-complete.")
    return {
        "declared_pressure_sites": list(ACTIVE_SITES),
        "realized_pressure_sites": list(ACTIVE_SITES),
        "pressure_capture_tensor_count_per_microbatch": 24,
        "pressure_capture_names_sha256": EXPECTED_CAPTURE_HASH,
    }


def _logical_counts(
    path: Path,
    *,
    expected_loss: float,
    expected_r_model: float,
) -> dict[str, int | float]:
    payload = _read_json(path)
    coverage = payload["coverage"]
    expected_coverage = {
        "sequences": 338,
        "input_tokens": 692_224,
        "excluded_tail_tokens": 1_444,
        "complete_block_coverage": True,
    }
    changed = [
        key for key, expected in expected_coverage.items() if coverage.get(key) != expected
    ]
    if changed:
        raise ValueError("Logical-product coverage mismatch: " + ", ".join(changed))
    diagnostic_loss = float(coverage["loss"])
    if abs(diagnostic_loss - expected_loss) > MAX_DIAGNOSTIC_LOSS_DELTA:
        raise ValueError("Logical-product loss does not reproduce validation loss.")

    measured = payload["measured"]
    operation_zero_count = sum(
        int(row["zero_product_count"])
        for row in measured["per_operation"].values()
    )
    block_zero_count = int(measured["block_zero_product_count"])
    model_product_count = int(measured["model_product_count"])
    if operation_zero_count != block_zero_count:
        raise ValueError("Logical-product operation counts do not reconcile.")
    r_model = block_zero_count / model_product_count
    _close(r_model, float(measured["R_model"]), tolerance=1e-16, label="R_model")
    _close(r_model, expected_r_model, tolerance=1e-16, label="verified R_model")
    return {
        "zero_product_count": block_zero_count,
        "model_product_count": model_product_count,
        "R_model": r_model,
        "diagnostic_validation_loss": diagnostic_loss,
        "diagnostic_minus_terminal_validation_loss": diagnostic_loss - expected_loss,
    }


def _site_counts(path: Path) -> dict[str, dict[str, int | float]]:
    payload = _read_json(path)
    pooled = {row["name"]: row for row in payload["pooled_by_site"]}
    result: dict[str, dict[str, int | float]] = {}
    for site in ALL_SITES:
        row = pooled.get(site)
        if row is None:
            raise ValueError(f"Activation statistics omit site {site!r}.")
        zero_count = int(row["exact_zero_count"])
        total_count = int(row["total"])
        if total_count <= 0 or not 0 <= zero_count <= total_count:
            raise ValueError(f"Site {site!r} has invalid integer counts.")
        if int(row["finite"]) != total_count or int(row["nonfinite"]) != 0:
            raise ValueError(f"Site {site!r} contains non-finite activations.")
        fraction = zero_count / total_count
        _close(
            fraction,
            float(row["exact_zero_fraction"]),
            label=f"{site} exact-zero fraction",
        )
        result[site] = {
            "exact_zero_count": zero_count,
            "total_count": total_count,
            "exact_zero_fraction": fraction,
        }
    return result


def _validate_verification(verification: Mapping[str, Any], *, run_name: str) -> None:
    expected = {
        "schema_version": 1,
        "status": "verified",
        "evidence_label": "valid",
        "condition_count": 5,
        "completed_optimizer_steps": 3_560,
        "training_input_tokens": 7_465_861_120,
        "complete_validation_passes": 20,
    }
    changed = [key for key, value in expected.items() if verification.get(key) != value]
    if changed:
        raise ValueError(f"{run_name} verification mismatch: " + ", ".join(changed))


def _reduce_run(
    run_dir: Path,
    *,
    series_id: str,
    verification: Mapping[str, Any],
    pressure_expected: bool,
) -> list[dict[str, Any]]:
    style = SERIES[series_id]
    conditions = sorted(
        verification["conditions"], key=lambda row: float(row["condition"]["gate_threshold"])
    )
    kappas = tuple(float(row["condition"]["gate_threshold"]) for row in conditions)
    if kappas != EXPECTED_KAPPAS:
        raise ValueError(f"{run_dir.name} does not contain the matched kappa grid.")

    rows = []
    for summary in conditions:
        condition = summary["condition"]
        if condition.get("topology_id") != "A4-Z":
            raise ValueError("Expected A4-Z topology.")
        if tuple(condition.get("active_sites", ())) != ACTIVE_SITES:
            raise ValueError("Expected A4 active gate sites.")
        if condition.get("gate_operator") != "one_sided_threshold":
            raise ValueError("Expected the one-sided A4 gate.")
        if pressure_expected:
            if condition.get("pressure_method") != "orthogonal_l1":
                raise ValueError("Expected orthogonal_l1 pressure.")
            if tuple(condition.get("pressure_sites", ())) != ACTIVE_SITES:
                raise ValueError("The historical declared pressure contract changed.")
            if float(condition.get("pressure_weight", -1.0)) != 1.0:
                raise ValueError("Expected lambda=1.")
            if float(condition.get("step_budget", -1.0)) != 1.0:
                raise ValueError("Expected OL1 trust budget 1.")
        else:
            if condition.get("pressure_method") != "none":
                raise ValueError("Expected the A4 baseline to have no pressure method.")
            if tuple(condition.get("pressure_sites", ())) != ():
                raise ValueError("Expected the A4 baseline to have no pressure sites.")
            if float(condition.get("pressure_weight", -1.0)) != 0.0:
                raise ValueError("Expected the A4 baseline pressure weight to be zero.")
            if condition.get("step_budget") is not None:
                raise ValueError("Expected no OL1 trust budget for the A4 baseline.")
        if int(summary["completed_steps"]) != 712:
            raise ValueError("Condition did not complete 712 optimizer boundaries.")
        if int(summary["input_tokens"]) != 1_493_172_224:
            raise ValueError("Condition training-token count changed.")

        attempt_id = str(summary["attempt_id"])
        attempt_dir = run_dir / "artifacts" / "attempts" / attempt_id
        manifest_path = attempt_dir / "manifest.json"
        logical_path = attempt_dir / "diagnostics" / "logical_products.json"
        activation_path = attempt_dir / "diagnostics" / "activation_statistics.json"
        manifest = _read_json(manifest_path)
        if manifest.get("attempt_id") != attempt_id or manifest.get("status") != "completed":
            raise ValueError("Attempt manifest is not a matching completed artifact.")
        if manifest.get("condition") != condition:
            raise ValueError("Attempt and cohort condition metadata disagree.")
        coverage = manifest.get("validation_coverage", {})
        expected_manifest_coverage = {
            "sequences": 338,
            "input_tokens": 692_224,
            "excluded_tail_tokens": 1_444,
            "complete_block_coverage": True,
        }
        changed = [
            key
            for key, expected in expected_manifest_coverage.items()
            if coverage.get(key) != expected
        ]
        if changed:
            raise ValueError("Attempt validation coverage mismatch: " + ", ".join(changed))

        loss = float(summary["final_validation_loss"])
        r_model = float(summary["R_model"])
        logical = _logical_counts(
            logical_path,
            expected_loss=loss,
            expected_r_model=r_model,
        )
        sites = _site_counts(activation_path)
        matched_contract = {key: manifest[key] for key in MATCHED_MANIFEST_FIELDS}
        gate_contract = {key: manifest[key] for key in MATCHED_GATE_FIELDS}
        rows.append(
            {
                "series_id": series_id,
                "variant": style["short_label"],
                "run": run_dir.name,
                "attempt_id": attempt_id,
                "kappa": float(condition["gate_threshold"]),
                "final_validation_loss": loss,
                "R_model": r_model,
                "declared_pressure_sites": (
                    list(ACTIVE_SITES) if pressure_expected else []
                ),
                "realized_pressure_sites": list(style["realized_pressure_sites"]),
                "logical_product_counts": logical,
                "site_exact_zero": sites,
                "matched_contract_sha256": _canonical_sha256(matched_contract),
                "matched_gate_contract_sha256": _canonical_sha256(gate_contract),
                "source_files": {
                    "manifest": _repo_path(manifest_path),
                    "manifest_sha256": _sha256(manifest_path),
                    "logical_products": _repo_path(logical_path),
                    "logical_products_sha256": _sha256(logical_path),
                    "activation_statistics": _repo_path(activation_path),
                    "activation_statistics_sha256": _sha256(activation_path),
                },
            }
        )
    return rows


def build_figure_data() -> dict[str, Any]:
    verification_paths = {
        "run011_a4": RUN_011 / "artifacts" / "verification.json",
        "run012_h_only": RUN_012 / "artifacts" / "verification.json",
        "run015_four_site": RUN_015 / "artifacts" / "verification.json",
    }
    run011_verification = _read_json(verification_paths["run011_a4"])
    run012_verification = _read_json(verification_paths["run012_h_only"])
    run015_verification = _read_json(verification_paths["run015_four_site"])
    _validate_verification(run011_verification, run_name="Run 011")
    _validate_verification(run012_verification, run_name="Run 012")
    _validate_verification(run015_verification, run_name="Run 015")
    initializations = {
        run011_verification["initial_parameter_sha256"],
        run012_verification["initial_parameter_sha256"],
        run015_verification["initial_parameter_sha256"],
    }
    if len(initializations) != 1:
        raise ValueError("The compared runs do not share initialization.")
    schedules = {
        run011_verification["training_schedule_sha256"],
        run012_verification["training_schedule_sha256"],
        run015_verification["training_schedule_sha256"],
    }
    if len(schedules) != 1:
        raise ValueError("The compared runs do not share the training schedule.")

    realization = {
        "run011_a4": {
            "declared_pressure_sites": [],
            "realized_pressure_sites": [],
            "pressure_method": "none",
        },
        "run012_h_only": _validate_run012_realization(),
        "run015_four_site": _validate_run015_realization(run015_verification),
    }
    run011_rows = _reduce_run(
        RUN_011,
        series_id="run011_a4",
        verification=run011_verification,
        pressure_expected=False,
    )
    run012_rows = _reduce_run(
        RUN_012,
        series_id="run012_h_only",
        verification=run012_verification,
        pressure_expected=True,
    )
    run015_rows = _reduce_run(
        RUN_015,
        series_id="run015_four_site",
        verification=run015_verification,
        pressure_expected=True,
    )

    comparisons = []
    for baseline, historical, corrected in zip(
        run011_rows, run012_rows, run015_rows, strict=True
    ):
        if not baseline["kappa"] == historical["kappa"] == corrected["kappa"]:
            raise ValueError("Matched rows have different kappa values.")
        gate_contracts = {
            baseline["matched_gate_contract_sha256"],
            historical["matched_gate_contract_sha256"],
            corrected["matched_gate_contract_sha256"],
        }
        if len(gate_contracts) != 1:
            raise ValueError("Matched A4 gate contracts differ.")
        if historical["matched_contract_sha256"] != corrected["matched_contract_sha256"]:
            raise ValueError("Matched manifest contracts differ beyond realized capture.")
        comparisons.append(
            {
                "kappa": historical["kappa"],
                "h_only_minus_a4_validation_loss": (
                    historical["final_validation_loss"]
                    - baseline["final_validation_loss"]
                ),
                "h_only_minus_a4_R_model_percentage_points": 100.0
                * (historical["R_model"] - baseline["R_model"]),
                "four_site_minus_a4_validation_loss": (
                    corrected["final_validation_loss"]
                    - baseline["final_validation_loss"]
                ),
                "four_site_minus_a4_R_model_percentage_points": 100.0
                * (corrected["R_model"] - baseline["R_model"]),
                "corrected_minus_h_only_validation_loss": (
                    corrected["final_validation_loss"]
                    - historical["final_validation_loss"]
                ),
                "corrected_minus_h_only_R_model_percentage_points": 100.0
                * (corrected["R_model"] - historical["R_model"]),
            }
        )

    return {
        "schema_version": 1,
        "status": "complete_verified_analysis",
        "question": (
            "Where do A4 without OL1, corrected four-site A4-OL1, and the "
            "historical h-only pressure realization lie at matched kappa?"
        ),
        "coverage": dict(EXPECTED_COVERAGE),
        "matched_identity": {
            "model": "random-initialized Pythia-14M",
            "initial_parameter_sha256": run012_verification[
                "initial_parameter_sha256"
            ],
            "training_schedule_sha256": run012_verification[
                "training_schedule_sha256"
            ],
            "optimizer_steps_per_condition": 712,
            "training_input_tokens_per_condition": 1_493_172_224,
            "gate_topology": "A4-Z",
            "gate_sites": list(ACTIVE_SITES),
            "gate_operator": "one_sided_threshold",
            "baseline_pressure_method": "none",
            "pressure_variant_method": "orthogonal_l1",
            "pressure_weight": 1.0,
            "step_budget": 1.0,
            "matched_manifest_fields": list(MATCHED_MANIFEST_FIELDS),
        },
        "realization_audit": realization,
        "series": run011_rows + run012_rows + run015_rows,
        "matched_comparison": comparisons,
        "source_verification": {
            series_id: {
                "path": _repo_path(path),
                "sha256": _sha256(path),
            }
            for series_id, path in verification_paths.items()
        },
        "interpretation": {
            "zero_mass": "pooled exact-zero activation count divided by pooled activation count",
            "R_model": "exact-zero logical-product opportunity, not measured speedup",
            "comparison": (
                "Run 012's declared four-site metadata is not used as its realized "
                "pressure identity; the audited inherited capture is h only."
            ),
        },
    }


def _series_rows(data: Mapping[str, Any], series_id: str) -> list[Mapping[str, Any]]:
    return sorted(
        (row for row in data["series"] if row["series_id"] == series_id),
        key=lambda row: float(row["kappa"]),
    )


def table_markdown(data: Mapping[str, Any]) -> str:
    pressure_rows = (
        row
        for row in data["series"]
        if row["series_id"] in {"run012_h_only", "run015_four_site"}
    )
    rows = sorted(
        pressure_rows,
        key=lambda row: (
            float(row["kappa"]),
            0 if row["series_id"] == "run012_h_only" else 1,
        ),
    )
    headers = [
        "kappa",
        "Realized pressure",
        *[f"{site} zero (%)" for site in ALL_SITES],
    ]
    lines = [
        "## Pooled exact-zero mass",
        "",
        "| " + " | ".join(headers) + " |",
        "|---:|:---|" + "---:|" * len(ALL_SITES),
    ]
    for row in rows:
        values = [
            f"{float(row['kappa']):g}",
            str(row["variant"]),
            *[
                f"{100.0 * float(row['site_exact_zero'][site]['exact_zero_fraction']):.6f}"
                for site in ALL_SITES
            ],
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Percentages are computed from pooled integer exact-zero counts over all six "
            "layers and the complete 338-block validation pass; layer or batch fractions "
            "are not averaged. All sites except `h` contain 531,628,032 activation "
            "elements per row; `h` contains 2,126,512,128 because of the four-times "
            "intermediate width. `attention_output` is the post-`W_o` diagnostic and is "
            "not an A4 gate or pressure site.",
        ]
    )
    return "\n".join(lines) + "\n"


def _style_axis(axis: Any) -> None:
    axis.grid(True, color="#D8D8D8", linewidth=0.65, alpha=0.75)
    axis.set_axisbelow(True)
    axis.tick_params(direction="out", length=3.5, width=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")


def plot(data: Mapping[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "pdf.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(10.6, 6.6))
    baseline = _series_rows(data, "run011_a4")
    historical = _series_rows(data, "run012_h_only")
    corrected = _series_rows(data, "run015_four_site")

    for left, right in zip(historical, corrected, strict=True):
        xs = [100.0 * float(left["R_model"]), 100.0 * float(right["R_model"])]
        ys = [float(left["final_validation_loss"]), float(right["final_validation_loss"])]
        axis.plot(
            xs,
            ys,
            color="#8A8A8A",
            linestyle=":",
            linewidth=1.25,
            alpha=0.8,
            zorder=2,
        )
        axis.annotate(
            rf"$\kappa={float(left['kappa']):g}$",
            xy=(sum(xs) / 2.0, sum(ys) / 2.0),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=8.0,
            color="#555555",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
            zorder=8,
        )

    for series_id, rows in (
        ("run011_a4", baseline),
        ("run012_h_only", historical),
        ("run015_four_site", corrected),
    ):
        style = SERIES[series_id]
        axis.plot(
            [100.0 * float(row["R_model"]) for row in rows],
            [float(row["final_validation_loss"]) for row in rows],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.35,
            markersize=8.0,
            markeredgecolor="white",
            markeredgewidth=0.95,
            label=style["label"],
            zorder=4 if series_id == "run011_a4" else 5,
        )

    axis.set_xlim(6.85, 13.10)
    axis.set_ylim(5.10, 6.12)
    axis.set_xlabel(r"Measured $R_{\mathrm{model}}$ (%)")
    axis.set_ylabel("Final validation loss (lower is better)")
    _style_axis(axis)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=SERIES[series_id]["color"],
            marker=SERIES[series_id]["marker"],
            linestyle=SERIES[series_id]["linestyle"],
            linewidth=2.35,
            markersize=7.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=SERIES[series_id]["label"],
        )
        for series_id in ("run011_a4", "run012_h_only", "run015_four_site")
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#8A8A8A",
            linestyle=":",
            linewidth=1.25,
            label="Matched kappa pair",
        )
    )
    figure.suptitle(
        "Pythia-14M A4 gates: no OL1 and realized OL1 pressure targets",
        x=0.5,
        y=0.975,
        fontsize=14.0,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.935,
        "A4 without pressure, historical h-only capture, and corrected four-site capture at matched gate threshold",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#444444",
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=4,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.15,
    )
    figure.text(
        0.5,
        0.025,
        "All points use one seed and all 338 complete validation blocks (692,224 tokens). "
        "Colored lines connect kappa order only.\n"
        r"$R_{\mathrm{model}}$ is exact-zero logical-product opportunity, not measured speedup.",
        ha="center",
        va="bottom",
        fontsize=8.1,
        color="#444444",
        linespacing=1.35,
    )
    figure.subplots_adjust(left=0.095, right=0.985, top=0.79, bottom=0.16)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".pdf.tmp")
    figure.savefig(
        temporary,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": "Pythia-14M A4 and OL1 pressure-target comparison",
            "Author": "sparsity-spillover analysis 009",
            "Subject": "Measured R_model versus final validation loss",
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
    output = plot(data)
    print(output)
    return output


if __name__ == "__main__":
    main()
