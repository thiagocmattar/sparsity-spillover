"""Build the verified endpoint comparison for Analysis 004."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ANALYSIS_DIR = Path(__file__).resolve().parent
ROOT = ANALYSIS_DIR.parents[1]
ANALYSIS_003 = (
    ROOT
    / "analyses"
    / "003-2026-08-30-run004-vs-run009-full-pass-l1-ol1"
    / "comparison.json"
)
RUN_011_DIR = ROOT / "runs" / "011-2026-08-30-pythia14m-full-pass-a4z"
RUN_011_VERIFICATION = RUN_011_DIR / "artifacts" / "verification.json"
OUTPUT = ANALYSIS_DIR / "comparison.json"
TABLES = ANALYSIS_DIR / "tables.md"

EXPECTED_KAPPAS = (0.0, 0.01, 0.05, 0.1, 0.5)
EXPECTED_SITES = ("a", "m", "h", "z", "q_post", "k_post", "v", "attention_output")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"{label}: {actual} != {expected}")


def _analysis_003_rows(value: dict[str, Any]) -> list[dict[str, Any]]:
    if value.get("schema_version") != 1:
        raise RuntimeError("Unsupported Analysis 003 comparison schema")
    if value.get("evidence_status") != "complete_verified_matched_cohorts":
        raise RuntimeError("Analysis 003 is not complete verified evidence")
    if len(value.get("conditions", [])) != 10:
        raise RuntimeError("Analysis 003 must contain ten unique endpoints")

    rows: list[dict[str, Any]] = []
    for source in value["conditions"]:
        row = dict(source)
        method = row["pressure_method"]
        if method == "l1_naive":
            row.update(
                series_id="a1h_naive_l1",
                series_label="A1-H naive L1",
                dose_name="lambda",
                dose=float(row["lambda"]),
            )
        elif method == "orthogonal_l1":
            row.update(
                series_id="a1h_ol1",
                series_label="A1-H OL1",
                dose_name="lambda",
                dose=float(row["lambda"]),
            )
        elif row["condition_id"] == "gelu-control":
            row.update(
                series_id="gelu_control",
                series_label="GeLU control",
                dose_name=None,
                dose=None,
            )
        elif row["condition_id"] == "relu-control":
            row.update(
                series_id="relu_control",
                series_label="A1-H ReLU control",
                dose_name=None,
                dose=None,
            )
        else:
            raise RuntimeError(f"Unexpected Analysis 003 condition: {row['condition_id']}")
        rows.append(row)
    return rows


def _run_011_rows(
    verification: dict[str, Any], matched_identity: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if verification.get("status") != "verified":
        raise RuntimeError("Run 011 status is not verified")
    if verification.get("evidence_label") != "valid":
        raise RuntimeError("Run 011 evidence is not valid")
    if verification.get("condition_count") != len(EXPECTED_KAPPAS):
        raise RuntimeError("Run 011 condition count does not match the approved grid")
    if verification.get("initial_parameter_sha256") != matched_identity.get(
        "initial_parameter_sha256"
    ):
        raise RuntimeError("Run 011 initialization does not match Analysis 003")
    if verification.get("training_schedule_sha256") != matched_identity.get(
        "training_schedule_sha256"
    ):
        raise RuntimeError("Run 011 training schedule does not match Analysis 003")

    verification_rows = sorted(
        verification["conditions"],
        key=lambda row: float(row["condition"]["gate_threshold"]),
    )
    observed_kappas = tuple(
        float(row["condition"]["gate_threshold"]) for row in verification_rows
    )
    if observed_kappas != EXPECTED_KAPPAS:
        raise RuntimeError(f"Unexpected Run 011 kappa grid: {observed_kappas}")

    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    expected_steps = int(matched_identity["optimizer_boundaries_per_condition"])
    expected_tokens = int(matched_identity["training_input_tokens_per_condition"])
    expected_validation = matched_identity["validation"]

    for verified in verification_rows:
        attempt_id = verified["attempt_id"]
        attempt_dir = RUN_011_DIR / "artifacts" / "attempts" / attempt_id
        paths = {
            "manifest": attempt_dir / "manifest.json",
            "metrics": attempt_dir / "metrics.json",
            "activation_statistics": attempt_dir / "diagnostics" / "activation_statistics.json",
            "logical_products": attempt_dir / "diagnostics" / "logical_products.json",
        }
        for label, path in paths.items():
            if not path.is_file():
                raise RuntimeError(f"Missing Run 011 {label}: {path}")

        manifest = _load_json(paths["manifest"])
        metrics = _load_json(paths["metrics"])
        activation = _load_json(paths["activation_statistics"])
        logical = _load_json(paths["logical_products"])
        condition = verified["condition"]
        kappa = float(condition["gate_threshold"])

        if manifest.get("status") != "completed":
            raise RuntimeError(f"Attempt {attempt_id} is not completed")
        if manifest.get("condition", {}).get("id") != condition["id"]:
            raise RuntimeError(f"Condition mismatch for {attempt_id}")
        if int(manifest.get("completed_steps", -1)) != expected_steps:
            raise RuntimeError(f"Step coverage mismatch for {attempt_id}")
        if int(manifest.get("input_tokens", -1)) != expected_tokens:
            raise RuntimeError(f"Training-token coverage mismatch for {attempt_id}")
        coverage = manifest["validation_coverage"]
        for field in ("sequences", "input_tokens", "excluded_tail_tokens"):
            if int(coverage[field]) != int(expected_validation[field]):
                raise RuntimeError(f"Validation {field} mismatch for {attempt_id}")
        if not coverage.get("complete_block_coverage"):
            raise RuntimeError(f"Incomplete validation coverage for {attempt_id}")

        final_loss = float(metrics["validation"]["final"]["loss"])
        _require_close(final_loss, float(verified["final_validation_loss"]), "final loss")

        measured = logical["measured"]
        zero_products = int(measured["block_zero_product_count"])
        model_products = int(measured["model_product_count"])
        r_model = zero_products / model_products
        _require_close(r_model, float(measured["R_model"]), "logical R_model")
        _require_close(r_model, float(verified["R_model"]), "verified R_model")

        pooled = {row["name"]: row for row in activation["pooled_by_site"]}
        if not set(EXPECTED_SITES).issubset(pooled):
            raise RuntimeError(f"Missing activation sites for {attempt_id}")
        site_fractions: dict[str, float] = {}
        for site in EXPECTED_SITES:
            site_row = pooled[site]
            exact_fraction = int(site_row["exact_zero_count"]) / int(site_row["total"])
            _require_close(
                exact_fraction,
                float(site_row["exact_zero_fraction"]),
                f"{attempt_id} {site} exact-zero fraction",
            )
            site_fractions[site] = exact_fraction
        for site, expected_fraction in verified["selected_site_exact_zero_fractions"].items():
            _require_close(site_fractions[site], float(expected_fraction), f"verified {site}")

        rows.append(
            {
                "R_model": r_model,
                "R_model_max": float(verified["R_model_max"]),
                "activation": "one_sided_threshold",
                "attempt_id": attempt_id,
                "condition_id": condition["id"],
                "dose": kappa,
                "dose_name": "kappa",
                "final_validation_loss": final_loss,
                "lambda": None,
                "pressure_method": "none",
                "run": "run011",
                "series_id": "a4z_threshold",
                "series_label": "A4-Z threshold",
                "site_exact_zero_fraction": site_fractions,
                "topology": "A4-Z",
            }
        )
        sources.append(
            {
                "attempt_id": attempt_id,
                "files": {
                    label: {"path": _relative(path), "sha256": _sha256(path)}
                    for label, path in paths.items()
                },
            }
        )
    return rows, sources


def _format_dose(row: dict[str, Any]) -> str:
    if row["dose"] is None:
        return "control"
    return f"{row['dose_name']}={float(row['dose']):g}"


def _table_text(conditions: list[dict[str, Any]]) -> str:
    order = {
        "gelu_control": 0,
        "relu_control": 1,
        "a1h_naive_l1": 2,
        "a1h_ol1": 3,
        "a4z_threshold": 4,
    }
    sorted_rows = sorted(
        conditions,
        key=lambda row: (
            order[row["series_id"]],
            -1.0 if row["dose"] is None else float(row["dose"]),
        ),
    )
    lines = [
        "# Analysis 004 tables",
        "",
        "Fractions were recalculated from stored integer counts before display.",
        "",
        "## Combined endpoints",
        "",
        "| Series | Dose | Final validation loss | `R_model` (%) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in sorted_rows:
        lines.append(
            f"| {row['series_label']} | {_format_dose(row)} | "
            f"{float(row['final_validation_loss']):.6f} | "
            f"{100.0 * float(row['R_model']):.6f} |"
        )

    run_011 = sorted(
        (row for row in conditions if row["series_id"] == "a4z_threshold"),
        key=lambda row: float(row["dose"]),
    )
    baseline = run_011[0]
    lines.extend(
        [
            "",
            "## Run 011 threshold effects",
            "",
            "Deltas are relative to the within-A4-Z `kappa=0` reference.",
            "",
            "| `kappa` | Final validation loss | Delta loss | `R_model` (%) | Delta `R_model` (pp) |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in run_011:
        lines.append(
            f"| {float(row['dose']):g} | {float(row['final_validation_loss']):.6f} | "
            f"{float(row['final_validation_loss']) - float(baseline['final_validation_loss']):+.6f} | "
            f"{100.0 * float(row['R_model']):.6f} | "
            f"{100.0 * (float(row['R_model']) - float(baseline['R_model'])):+.6f} |"
        )

    lines.extend(
        [
            "",
            "## Run 011 selected-site exact-zero mass",
            "",
            "| `kappa` | `a` (%) | `m` (%) | `h` (%) | `z` (%) |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in run_011:
        site = row["site_exact_zero_fraction"]
        lines.append(
            f"| {float(row['dose']):g} | {100.0 * site['a']:.6f} | "
            f"{100.0 * site['m']:.6f} | {100.0 * site['h']:.6f} | "
            f"{100.0 * site['z']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Run 011 untargeted-site exact-zero mass",
            "",
            "| `kappa` | `q_post` (%) | `k_post` (%) | `v` (%) | Attention output (%) |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in run_011:
        site = row["site_exact_zero_fraction"]
        lines.append(
            f"| {float(row['dose']):g} | {100.0 * site['q_post']:.9f} | "
            f"{100.0 * site['k_post']:.9f} | {100.0 * site['v']:.9f} | "
            f"{100.0 * site['attention_output']:.9f} |"
        )
    lines.extend(
        [
            "",
            "`attention_output` is post-`W_o`; it is distinct from the selected pre-`W_o` site `z`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> tuple[Path, Path]:
    analysis_003 = _load_json(ANALYSIS_003)
    run_011_verification = _load_json(RUN_011_VERIFICATION)
    matched_identity = analysis_003["matched_identity"]
    conditions = _analysis_003_rows(analysis_003)
    run_011_rows, attempt_sources = _run_011_rows(run_011_verification, matched_identity)
    conditions.extend(run_011_rows)
    if len(conditions) != 15:
        raise RuntimeError("Expected fifteen unique combined endpoints")

    value = {
        "schema_version": 1,
        "question": (
            "Where do full-pass A1-H naive-L1, A1-H OL1, and A4-Z threshold "
            "endpoints lie in final validation loss versus measured R_model?"
        ),
        "evidence_status": "complete_verified_matched_cohorts",
        "conditions": conditions,
        "matched_identity": {
            "initial_parameter_sha256": matched_identity["initial_parameter_sha256"],
            "training_schedule_sha256": matched_identity["training_schedule_sha256"],
            "optimizer_boundaries_per_condition": matched_identity[
                "optimizer_boundaries_per_condition"
            ],
            "training_input_tokens_per_condition": matched_identity[
                "training_input_tokens_per_condition"
            ],
            "validation": matched_identity["validation"],
        },
        "source_artifacts": {
            "analysis_003_comparison": {
                "path": _relative(ANALYSIS_003),
                "sha256": _sha256(ANALYSIS_003),
            },
            "run_011_verification": {
                "path": _relative(RUN_011_VERIFICATION),
                "sha256": _sha256(RUN_011_VERIFICATION),
            },
            "run_011_attempts": attempt_sources,
        },
        "limits": {
            "replicates": "one seed at one Pythia-14M scale",
            "comparison": (
                "A1-H pressure and A4-Z thresholding differ in topology, operator, "
                "and pressure mechanism; their cross-series contrast is descriptive"
            ),
            "logical_products": "R_model is logical opportunity, not measured speedup",
            "line_segments": "connect dose-ordered endpoints only; no interpolation claim",
        },
    }

    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(OUTPUT)
    table_temporary = TABLES.with_suffix(".md.tmp")
    table_temporary.write_text(_table_text(conditions), encoding="utf-8")
    table_temporary.replace(TABLES)
    print(OUTPUT)
    print(TABLES)
    return OUTPUT, TABLES


if __name__ == "__main__":
    main()
