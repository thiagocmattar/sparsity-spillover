"""Partial matched comparison of Run 006 and completed Run 007 conditions."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from sparsity_research.artifacts import build_transfer_inventory, verify_transfer_inventory


ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
RUN006_DIR = REPO_ROOT / "runs/006-2026-08-29-pythia14m-a4z-threshold-local"
RUN007_DIR = REPO_ROOT / "runs/007-2026-08-29-pythia14m-a4z-threshold-ol1-local"
RUN006_VERIFICATION = RUN006_DIR / "artifacts/verification.json"
SITES = ("a", "m", "h", "z")
SELECTED_ATTEMPTS = {
    0.0: "001-20260829-222842-c76ef7c7",
    0.01: "003-20260829-230845-3860f017",
    0.05: "004-20260829-232134-f38ab534",
    0.1: "005-20260829-233403-b12a6654",
}
FAILED_ATTEMPTS = (
    "002-20260829-224130-e2a24f24",
    "006-20260829-234710-5872533a",
    "007-20260830-000653-eba06685",
)
EXPECTED_STEPS = 581
EXPECTED_TOKENS = 76_152_832
EXPECTED_VALIDATION = {
    "sequences": 338,
    "input_tokens": 692_224,
    "excluded_tail_tokens": 1_444,
    "complete_block_coverage": True,
}
EXPECTED_PRESSURE = {
    "method": "orthogonal_l1",
    "sites": ["a", "m", "h", "z"],
    "weight": 1.0,
    "step_budget": 1.0,
    "eps": 1.0e-12,
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a YAML mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory_content_sha256(inventory: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for row in inventory["files"]:
        digest.update(str(row["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _require_validation(record: Any, label: str) -> None:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"Missing validation record: {label}")
    for key, expected in EXPECTED_VALIDATION.items():
        if record.get(key) != expected:
            raise RuntimeError(f"Validation coverage mismatch at {label}/{key}.")
    if not math.isfinite(float(record.get("loss", math.nan))):
        raise RuntimeError(f"Non-finite validation loss at {label}.")


def _require_matched_configs() -> None:
    baseline = _yaml(RUN006_DIR / "config.yaml")
    ol1 = _yaml(RUN007_DIR / "config.yaml")
    if baseline["model"] != ol1["model"] or baseline["data"] != ol1["data"]:
        raise RuntimeError("Model or data configuration differs beyond OL1.")
    if baseline["seeds"] != ol1["seeds"] or baseline["validation"] != ol1["validation"]:
        raise RuntimeError("Seed or validation configuration differs beyond OL1.")
    if baseline["artifacts"] != ol1["artifacts"]:
        raise RuntimeError("Artifact-retention configuration differs beyond OL1.")

    baseline_training = dict(baseline["training"])
    ol1_training = dict(ol1["training"])
    for key in ("target_cohort_seconds", "planning_cohort_seconds"):
        baseline_training.pop(key)
        ol1_training.pop(key)
    if baseline_training != ol1_training:
        raise RuntimeError("Scientific training configuration differs beyond OL1.")

    for key in ("active_sites", "gate_operator", "gate_thresholds"):
        if baseline["conditions"].get(key) != ol1["conditions"].get(key):
            raise RuntimeError(f"Threshold condition differs at {key}.")
    if baseline["conditions"].get("pressure_method") != "none":
        raise RuntimeError("Run 006 is not the no-pressure baseline.")
    expected_conditions = {
        "pressure_method": "orthogonal_l1",
        "pressure_sites": ["a", "m", "h", "z"],
        "pressure_weight": 1.0,
        "step_budget": 1.0,
    }
    for key, expected in expected_conditions.items():
        if ol1["conditions"].get(key) != expected:
            raise RuntimeError(f"Run 007 OL1 configuration differs at {key}.")

    baseline_diagnostics = dict(baseline["diagnostics"])
    ol1_diagnostics = dict(ol1["diagnostics"])
    if baseline_diagnostics.pop("gradient_interaction") is not False:
        raise RuntimeError("Run 006 gradient-interaction contract changed.")
    if ol1_diagnostics.pop("gradient_interaction") is not True:
        raise RuntimeError("Run 007 lacks required OL1 gradient diagnostics.")
    if baseline_diagnostics != ol1_diagnostics:
        raise RuntimeError("Endpoint diagnostics differ beyond OL1 gradient interaction.")


def _load_run006() -> dict[str, Any]:
    value = _json(RUN006_VERIFICATION)
    if value.get("status") != "verified" or value.get("evidence_label") != "valid":
        raise RuntimeError("Run 006 is not terminally verified valid evidence.")
    return value


def _load_completed_run007(
    attempt_id: str,
    kappa: float,
    *,
    initial_sha256: str,
    schedule_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    attempt_dir = RUN007_DIR / "artifacts/attempts" / attempt_id
    manifest = _json(attempt_dir / "manifest.json")
    metrics = _json(attempt_dir / "metrics.json")
    resolved = _yaml(attempt_dir / "config.yaml")
    condition = manifest.get("condition")
    if manifest.get("status") != "completed" or manifest.get("attempt_id") != attempt_id:
        raise RuntimeError(f"Run 007 attempt is not completed: {attempt_id}")
    if condition != metrics.get("condition") or condition != resolved.get("condition"):
        raise RuntimeError(f"Condition identity mismatch: {attempt_id}")
    if float(condition.get("gate_threshold", math.nan)) != kappa:
        raise RuntimeError(f"Kappa mismatch: {attempt_id}")
    expected_condition_fields = {
        "topology_id": "A4-Z",
        "active_sites": ["a", "m", "h", "z"],
        "gate_operator": "one_sided_threshold",
        "pressure_method": "orthogonal_l1",
        "pressure_sites": ["a", "m", "h", "z"],
        "pressure_weight": 1.0,
        "step_budget": 1.0,
    }
    for key, expected in expected_condition_fields.items():
        if condition.get(key) != expected:
            raise RuntimeError(f"Condition mismatch at {attempt_id}/{key}.")
    expected_gate = {"operator": "one_sided_threshold", "kappa": kappa}
    if resolved.get("model", {}).get("site_gate") != expected_gate:
        raise RuntimeError(f"Resolved gate mismatch: {attempt_id}")
    if resolved.get("activation_pressure") != EXPECTED_PRESSURE:
        raise RuntimeError(f"Resolved OL1 mismatch: {attempt_id}")
    if manifest.get("activation_pressure") != EXPECTED_PRESSURE:
        raise RuntimeError(f"Manifest OL1 mismatch: {attempt_id}")
    if manifest.get("initial_parameter_sha256") != initial_sha256:
        raise RuntimeError(f"Initialization mismatch: {attempt_id}")
    if manifest.get("training_schedule_hash") != schedule_sha256:
        raise RuntimeError(f"Schedule mismatch: {attempt_id}")
    if int(manifest.get("completed_steps", -1)) != EXPECTED_STEPS:
        raise RuntimeError(f"Step count mismatch: {attempt_id}")
    if int(manifest.get("input_tokens", -1)) != EXPECTED_TOKENS:
        raise RuntimeError(f"Training-token mismatch: {attempt_id}")

    validation = metrics.get("validation", {})
    for name in ("step_one", "final", "logical_product_diagnostic_eager"):
        _require_validation(validation.get(name), f"{attempt_id}/{name}")

    event_rows = [
        json.loads(line)
        for line in (attempt_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_events = [row for row in event_rows if row.get("event") == "train"]
    if [int(row["step"]) for row in train_events] != list(range(1, EXPECTED_STEPS + 1)):
        raise RuntimeError(f"Training-event coverage mismatch: {attempt_id}")
    for row in train_events:
        if row.get("adamw_gradient_clipping_enabled") is not True:
            raise RuntimeError(f"Task-gradient clipping mismatch: {attempt_id}")
        for key in (
            "pressure_loss",
            "task_gradient_norm",
            "pressure_gradient_norm",
            "task_pressure_gradient_dot",
            "trust_scale",
            "pressure_to_task_ratio_final",
        ):
            if not math.isfinite(float(row.get(key, math.nan))):
                raise RuntimeError(f"Invalid OL1 diagnostic at {attempt_id}/{key}.")
        if not 0.0 <= float(row["trust_scale"]) <= 1.0:
            raise RuntimeError(f"Invalid OL1 trust scale: {attempt_id}")
        if float(row["pressure_to_task_ratio_final"]) > 1.0 + 1.0e-6:
            raise RuntimeError(f"OL1 trust budget exceeded: {attempt_id}")

    activation = _json(attempt_dir / "diagnostics/activation_statistics.json")
    pooled = activation.get("pooled_by_site")
    if not isinstance(pooled, list):
        raise RuntimeError(f"Missing pooled activation statistics: {attempt_id}")
    pooled_by_name = {row["name"]: row for row in pooled}
    if not set(SITES).issubset(pooled_by_name):
        raise RuntimeError(f"Missing threshold-site statistics: {attempt_id}")
    if any(int(row.get("nonfinite", -1)) != 0 for row in pooled):
        raise RuntimeError(f"Non-finite pooled activations: {attempt_id}")

    logical = _json(attempt_dir / "diagnostics/logical_products.json")
    measured = logical.get("measured")
    maximum = logical.get("architecture_maximum")
    if not isinstance(measured, Mapping) or not isinstance(maximum, Mapping):
        raise RuntimeError(f"Missing logical-product evidence: {attempt_id}")
    for key in ("R_block", "R_model"):
        value = float(measured.get(key, math.nan))
        if not 0.0 <= value <= 1.0:
            raise RuntimeError(f"Invalid logical fraction at {attempt_id}/{key}.")

    transfer = _json(attempt_dir / "transfer_inventory.json")
    verify_transfer_inventory(attempt_dir, transfer)
    if build_transfer_inventory(attempt_dir) != transfer:
        raise RuntimeError(f"Transfer inventory is incomplete: {attempt_id}")
    checkpoint = metrics.get("checkpoint", {})
    checkpoint_inventory = build_transfer_inventory(attempt_dir / str(checkpoint.get("path")))
    if _inventory_content_sha256(checkpoint_inventory) != checkpoint.get("content_sha256"):
        raise RuntimeError(f"Checkpoint content mismatch: {attempt_id}")

    row = {
        "attempt_id": attempt_id,
        "final_validation_loss": float(validation["final"]["loss"]),
        "R_block": float(measured["R_block"]),
        "R_model": float(measured["R_model"]),
        "R_model_max": float(maximum["R_model_max_fraction"]),
        "exact_zero_fraction": {
            site: float(pooled_by_name[site]["exact_zero_fraction"]) for site in SITES
        },
    }
    source = {
        "attempt_id": attempt_id,
        "run_code_content_sha256": manifest["run_code"]["content_sha256"],
        "checkpoint_content_sha256": checkpoint["content_sha256"],
        "transfer_inventory": {
            "path": str((attempt_dir / "transfer_inventory.json").relative_to(REPO_ROOT)).replace(
                "\\", "/"
            ),
            "sha256": _sha256(attempt_dir / "transfer_inventory.json"),
            "total_bytes": int(transfer["total_bytes"]),
        },
        "endpoint_files": {
            name: _sha256(attempt_dir / relative)
            for name, relative in {
                "manifest": "manifest.json",
                "metrics": "metrics.json",
                "activation_statistics": "diagnostics/activation_statistics.json",
                "logical_products": "diagnostics/logical_products.json",
            }.items()
        },
    }
    return row, source


def _metrics_run006(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "attempt_id": row["attempt_id"],
        "final_validation_loss": float(row["final_validation_loss"]),
        "R_block": float(row["R_block"]),
        "R_model": float(row["R_model"]),
        "R_model_max": float(row["R_model_max"]),
        "exact_zero_fraction": {
            site: float(row["selected_site_exact_zero_fractions"][site]) for site in SITES
        },
    }


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    no_worse = (
        left["final_validation_loss"] <= right["final_validation_loss"]
        and left["R_model"] >= right["R_model"]
    )
    strict = (
        left["final_validation_loss"] < right["final_validation_loss"]
        or left["R_model"] > right["R_model"]
    )
    return bool(no_worse and strict)


def main() -> Path:
    _require_matched_configs()
    run006 = _load_run006()
    initial_sha256 = str(run006["initial_parameter_sha256"])
    schedule_sha256 = str(run006["training_schedule_sha256"])
    baseline_by_kappa = {
        float(row["condition"]["gate_threshold"]): row for row in run006["conditions"]
    }
    if set(baseline_by_kappa) != {0.0, 0.01, 0.05, 0.1, 0.5}:
        raise RuntimeError("Run 006 kappa grid changed.")

    rows = []
    all_points = []
    source_attempts = []
    run007_code_hashes = set()
    for kappa, attempt_id in SELECTED_ATTEMPTS.items():
        baseline_source = baseline_by_kappa[kappa]
        if int(baseline_source["completed_steps"]) != EXPECTED_STEPS:
            raise RuntimeError(f"Run 006 step mismatch at kappa={kappa}.")
        if int(baseline_source["input_tokens"]) != EXPECTED_TOKENS:
            raise RuntimeError(f"Run 006 token mismatch at kappa={kappa}.")
        baseline = _metrics_run006(baseline_source)
        ol1, source = _load_completed_run007(
            attempt_id,
            kappa,
            initial_sha256=initial_sha256,
            schedule_sha256=schedule_sha256,
        )
        if not math.isclose(baseline["R_model_max"], ol1["R_model_max"], abs_tol=1e-15):
            raise RuntimeError(f"Architecture ceiling mismatch at kappa={kappa}.")
        source_attempts.append(source)
        run007_code_hashes.add(source["run_code_content_sha256"])
        for run_name, point in (("run006", baseline), ("run007", ol1)):
            all_points.append(
                {
                    "run": run_name,
                    "kappa": kappa,
                    "final_validation_loss": point["final_validation_loss"],
                    "R_model": point["R_model"],
                }
            )
        relation = "tradeoff"
        if _dominates(ol1, baseline):
            relation = "run007_dominates"
        elif _dominates(baseline, ol1):
            relation = "run006_dominates"
        rows.append(
            {
                "kappa": kappa,
                "run006": baseline,
                "run007": ol1,
                "delta_run007_minus_run006": {
                    "final_validation_loss": ol1["final_validation_loss"]
                    - baseline["final_validation_loss"],
                    "R_block": ol1["R_block"] - baseline["R_block"],
                    "R_model": ol1["R_model"] - baseline["R_model"],
                    "exact_zero_fraction": {
                        site: ol1["exact_zero_fraction"][site]
                        - baseline["exact_zero_fraction"][site]
                        for site in SITES
                    },
                },
                "paired_quality_R_model_relation": relation,
            }
        )
    if len(run007_code_hashes) != 1:
        raise RuntimeError("Selected Run 007 attempts used different run-code identities.")

    failed = []
    for attempt_id in FAILED_ATTEMPTS:
        path = RUN007_DIR / "artifacts/attempts" / attempt_id / "manifest.json"
        manifest = _json(path)
        if manifest.get("status") != "failed":
            raise RuntimeError(f"Expected failed attempt status: {attempt_id}")
        failure = manifest.get("failure", {})
        failed.append(
            {
                "attempt_id": attempt_id,
                "kappa": float(manifest["condition"]["gate_threshold"]),
                "status": "failed",
                "failure_type": failure.get("type"),
                "failure_first_line": str(failure.get("message", "")).splitlines()[0],
                "manifest_sha256": _sha256(path),
            }
        )

    frontier = [
        point
        for point in all_points
        if not any(_dominates(other, point) for other in all_points if other is not point)
    ]
    result = {
        "schema_version": 1,
        "evidence_status": "partial_completed_attempts",
        "question": "How does A4-Z all-site OL1 change Run 006 at each completed matched kappa?",
        "delta_sign": "run007 minus run006; lower validation loss and higher logical fractions are favorable",
        "available_pair_count": len(rows),
        "planned_pair_count": 5,
        "matched_identity_per_condition": {
            "initial_parameter_sha256": initial_sha256,
            "training_schedule_sha256": schedule_sha256,
            "optimizer_steps": EXPECTED_STEPS,
            "training_input_tokens": EXPECTED_TOKENS,
            "validation_passes": 3,
            **EXPECTED_VALIDATION,
        },
        "intervention": {
            "run006": "no activation pressure",
            "run007": "orthogonal_l1 at a,m,h,z with lambda=1 and trust budget=1",
            "gate": "one-sided threshold at a,m,h,z with matched kappa",
        },
        "source_files": {
            "run006_verification": {
                "path": str(RUN006_VERIFICATION.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": _sha256(RUN006_VERIFICATION),
            },
            "run006_config": {
                "path": str((RUN006_DIR / "config.yaml").relative_to(REPO_ROOT)).replace(
                    "\\", "/"
                ),
                "sha256": _sha256(RUN006_DIR / "config.yaml"),
            },
            "run007_config": {
                "path": str((RUN007_DIR / "config.yaml").relative_to(REPO_ROOT)).replace(
                    "\\", "/"
                ),
                "sha256": _sha256(RUN007_DIR / "config.yaml"),
            },
        },
        "run007_completed_attempt_sources": source_attempts,
        "failed_attempts": failed,
        "unpaired_conditions": [
            {
                "kappa": 0.5,
                "run006": _metrics_run006(baseline_by_kappa[0.5]),
                "run007": None,
                "reason": "Run 007 attempts 006 and 007 failed with accelerator out-of-memory errors.",
            }
        ],
        "rows": rows,
        "joint_quality_R_model_frontier_over_available_pairs": sorted(
            frontier, key=lambda row: (row["R_model"], row["final_validation_loss"])
        ),
        "limits": [
            "one seed and one model scale",
            "Run 007 is not a terminally verified complete cohort; each selected completed attempt is independently revalidated here",
            "kappa=0.5 has no completed Run 007 endpoint and is neither imputed nor included in trend or frontier calculations",
            "failed kappa=0.5 attempts make the missing endpoint potentially non-random with respect to resource demand",
            "the user-directed interpretation treats first- and second-decimal validation-loss differences as prone to realization noise",
            "R_block and R_model are logical-product opportunities, not measured runtime speedups",
            "joint-site OL1 effects are not individually attributable",
        ],
    }
    output = ANALYSIS_DIR / "comparison.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(output)
    return output


if __name__ == "__main__":
    main()
