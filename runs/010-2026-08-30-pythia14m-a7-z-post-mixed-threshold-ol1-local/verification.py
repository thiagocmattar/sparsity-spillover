"""Terminal scientific and artifact verification for Run 010."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from sparsity_research.artifacts import (
    build_transfer_inventory,
    config_sha256,
    verify_transfer_inventory,
)
from sparsity_research.data import file_sha256
from sparsity_research.metrics import LOGICAL_OPERATIONS

from run_config import (
    EXPECTED_ACTIVE_SITES,
    EXPECTED_SCHEDULE_SHA256,
    RUN_DIR,
    condition_specs,
    inventory_content_sha256,
    load_config,
    mapping,
    run_code_identity,
    site_gates,
    write_json,
)


EXPECTED_CEILING_NUMERATOR = 5_638_717_440
EXPECTED_CEILING_DENOMINATOR = 18_825_609_216
EXPECTED_CEILING_FRACTION = EXPECTED_CEILING_NUMERATOR / EXPECTED_CEILING_DENOMINATOR


def verify_cohort() -> Path:
    config = load_config()
    conditions = condition_specs(config)
    max_steps = int(mapping(config, "training")["max_steps"])
    tokens_per_update = int(mapping(config, "training")["global_batch_size"]) * int(
        mapping(config, "data")["sequence_length"]
    )
    expected_code = run_code_identity()["content_sha256"]
    provenance = _json(RUN_DIR / "artifacts" / "launch-provenance.json")
    if provenance.get("config_sha256") != config_sha256(config):
        raise RuntimeError("Launch provenance config identity changed.")
    if provenance.get("run_code_content_sha256") != expected_code:
        raise RuntimeError("Launch provenance run-code identity changed.")
    if provenance.get("training_schedule_sha256") != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError("Launch provenance schedule identity changed.")
    calibration = provenance.get("calibration")
    if not isinstance(calibration, Mapping):
        raise RuntimeError("Launch provenance calibration identity is missing.")
    if file_sha256(RUN_DIR / str(calibration.get("path"))) != calibration.get("sha256"):
        raise RuntimeError("Launch provenance calibration artifact changed.")

    attempts_root = RUN_DIR / "artifacts" / "attempts"
    rows = []
    initial_hashes = set()
    schedule_hashes = set()
    code_hashes = set()
    git_commits = set()
    git_dirty_states = set()
    checkpoint_bytes = 0
    condition_seconds = 0.0

    for condition in conditions:
        attempt_dir = _one_attempt(attempts_root, int(condition["order"]))
        manifest = _json(attempt_dir / "manifest.json")
        metrics = _json(attempt_dir / "metrics.json")
        resolved = _yaml(attempt_dir / "config.yaml")
        if manifest.get("status") != "completed" or manifest.get("attempt_id") != attempt_dir.name:
            raise RuntimeError(f"Attempt {attempt_dir.name} is not terminally completed.")
        if manifest.get("condition") != condition or metrics.get("condition") != condition:
            raise RuntimeError(f"Condition identity mismatch for {attempt_dir.name}.")
        if resolved.get("condition") != condition:
            raise RuntimeError(f"Resolved config condition mismatch for {attempt_dir.name}.")
        expected_gates = site_gates(float(condition["gate_threshold"]))
        if resolved.get("model", {}).get("site_gate") is not None:
            raise RuntimeError(f"Uniform gate unexpectedly configured for {attempt_dir.name}.")
        if resolved.get("model", {}).get("site_gates") != expected_gates:
            raise RuntimeError(f"Resolved mixed gates mismatch for {attempt_dir.name}.")
        expected_pressure = {
            "method": "orthogonal_l1",
            "sites": list(EXPECTED_ACTIVE_SITES),
            "weight": 1.0,
            "step_budget": 1.0,
            "eps": 1.0e-12,
        }
        if resolved.get("activation_pressure") != expected_pressure:
            raise RuntimeError(f"All-site OL1 pressure mismatch for {attempt_dir.name}.")
        if manifest.get("activation_pressure") != expected_pressure:
            raise RuntimeError(f"Manifest OL1 pressure mismatch for {attempt_dir.name}.")
        if int(manifest.get("completed_steps", -1)) != max_steps:
            raise RuntimeError(f"Completed-step mismatch for {attempt_dir.name}.")
        if int(manifest.get("input_tokens", -1)) != max_steps * tokens_per_update:
            raise RuntimeError(f"Training-token mismatch for {attempt_dir.name}.")
        topology = manifest.get("topology", {})
        if topology.get("topology_id") != "A7-Z-POST":
            raise RuntimeError(f"Topology mismatch for {attempt_dir.name}.")
        if topology.get("active_sites") != list(EXPECTED_ACTIVE_SITES):
            raise RuntimeError(f"Active-site mismatch for {attempt_dir.name}.")
        if topology.get("qk_placement") != "post_rope":
            raise RuntimeError(f"QK placement mismatch for {attempt_dir.name}.")
        if topology.get("site_gate") is not None or topology.get("site_gates") != expected_gates:
            raise RuntimeError(f"Realized mixed gates mismatch for {attempt_dir.name}.")

        events = _jsonl(attempt_dir / "events.jsonl")
        train_events = [row for row in events if row.get("event") == "train"]
        if [int(row["step"]) for row in train_events] != list(range(1, max_steps + 1)):
            raise RuntimeError(f"Training events are incomplete for {attempt_dir.name}.")
        for event in train_events:
            _finite(event.get("task_loss"), f"{attempt_dir.name} task loss")
            _finite(event.get("tokens_per_second"), f"{attempt_dir.name} throughput")
            _finite(event.get("adamw_gradient_norm_pre_clip"), f"{attempt_dir.name} gradient")
            if event.get("adamw_gradient_clipping_enabled") is not True:
                raise RuntimeError(f"Task-gradient clipping disabled in {attempt_dir.name}.")
        _require_ol1_events(train_events, attempt_dir.name)

        validation = metrics.get("validation")
        if not isinstance(validation, Mapping):
            raise RuntimeError(f"Validation metrics missing for {attempt_dir.name}.")
        for name in ("step_one", "final", "logical_product_diagnostic_eager"):
            _require_validation(validation.get(name), f"{attempt_dir.name}/{name}")

        activation = _json(attempt_dir / "diagnostics" / "activation_statistics.json")
        requested = list(mapping(config, "diagnostics")["activation_sites"])
        activation_rows = activation.get("rows")
        pooled = activation.get("pooled_by_site")
        if not isinstance(activation_rows, list) or len(activation_rows) != 6 * len(requested):
            raise RuntimeError(f"Activation rows are incomplete for {attempt_dir.name}.")
        if not isinstance(pooled, list) or {row["name"] for row in pooled} != set(requested):
            raise RuntimeError(f"Pooled activation sites mismatch for {attempt_dir.name}.")
        if any(int(row.get("nonfinite", -1)) != 0 for row in activation_rows):
            raise RuntimeError(f"Non-finite activation statistics in {attempt_dir.name}.")
        if activation.get("site_definition", {}).get("z") != (
            "concatenated PV context immediately before attention.dense (W_o)"
        ):
            raise RuntimeError(f"z boundary definition missing for {attempt_dir.name}.")
        pooled_by_name = {row["name"]: row for row in pooled}

        weights = _json(attempt_dir / "diagnostics" / "weight_statistics.json")
        if weights.get("inclusion") != "all named parameters, including bias and normalization":
            raise RuntimeError(f"Weight inclusion rule missing for {attempt_dir.name}.")
        weight_rows = weights.get("rows")
        if not isinstance(weight_rows, list) or not weight_rows:
            raise RuntimeError(f"Weight statistics missing for {attempt_dir.name}.")
        if any(int(row.get("nonfinite", -1)) != 0 for row in weight_rows):
            raise RuntimeError(f"Non-finite weight statistics in {attempt_dir.name}.")

        logical = _json(attempt_dir / "diagnostics" / "logical_products.json")
        measured = logical.get("measured")
        maximum = logical.get("architecture_maximum")
        if not isinstance(measured, Mapping) or not isinstance(maximum, Mapping):
            raise RuntimeError(f"Logical-product artifacts missing for {attempt_dir.name}.")
        if set(measured.get("per_operation", {})) != set(LOGICAL_OPERATIONS):
            raise RuntimeError(f"Logical operations are incomplete for {attempt_dir.name}.")
        for key in (
            "block_zero_product_count", "block_product_count", "lm_head_product_count",
            "model_product_count",
        ):
            if int(measured.get(key, -1)) < 0:
                raise RuntimeError(f"Logical counter {key} missing for {attempt_dir.name}.")
        for key in ("R_block", "R_model"):
            value = float(measured.get(key, math.nan))
            if not 0.0 <= value <= 1.0:
                raise RuntimeError(f"Logical fraction {key} invalid for {attempt_dir.name}.")
        if maximum.get("topology_id") != "A7-Z-POST":
            raise RuntimeError(f"Architecture ceiling topology mismatch for {attempt_dir.name}.")
        if int(maximum.get("reachable_product_count", -1)) != EXPECTED_CEILING_NUMERATOR:
            raise RuntimeError(f"Architecture ceiling numerator mismatch for {attempt_dir.name}.")
        if int(maximum.get("model_product_count", -1)) != EXPECTED_CEILING_DENOMINATOR:
            raise RuntimeError(f"Architecture ceiling denominator mismatch for {attempt_dir.name}.")
        if not math.isclose(
            float(maximum.get("R_model_max_fraction", math.nan)),
            EXPECTED_CEILING_FRACTION,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError(f"Architecture ceiling fraction mismatch for {attempt_dir.name}.")

        checkpoint = metrics.get("checkpoint")
        if not isinstance(checkpoint, Mapping) or checkpoint.get("optimizer_saved") is not False:
            raise RuntimeError(f"Checkpoint retention mismatch for {attempt_dir.name}.")
        rebuilt_checkpoint = build_transfer_inventory(attempt_dir / str(checkpoint["path"]))
        if inventory_content_sha256(rebuilt_checkpoint) != checkpoint.get("content_sha256"):
            raise RuntimeError(f"Checkpoint hash mismatch for {attempt_dir.name}.")
        transfer = _json(attempt_dir / "transfer_inventory.json")
        verify_transfer_inventory(attempt_dir, transfer)

        initial_hashes.add(str(manifest.get("initial_parameter_sha256")))
        schedule_hashes.add(str(manifest.get("training_schedule_hash")))
        code_hashes.add(str(manifest.get("run_code", {}).get("content_sha256")))
        git_commits.add(manifest.get("code", {}).get("git_commit"))
        git_dirty_states.add(manifest.get("code", {}).get("git_dirty"))
        checkpoint_bytes += int(checkpoint["bytes"])
        condition_seconds += float(metrics["timing"]["total_seconds"])
        site_exact = {
            site: float(pooled_by_name[site]["exact_zero_fraction"])
            for site in EXPECTED_ACTIVE_SITES
        }
        rows.append(
            {
                "attempt_id": attempt_dir.name,
                "condition": condition,
                "completed_steps": max_steps,
                "input_tokens": max_steps * tokens_per_update,
                "final_train_loss": float(metrics["training"]["task_loss_final"]),
                "final_validation_loss": float(validation["final"]["loss"]),
                "median_step_seconds": float(metrics["training"]["median_step_seconds"]),
                "median_tokens_per_second": float(metrics["training"]["median_tokens_per_second"]),
                "peak_gpu_memory_allocated_bytes": int(metrics["training"]["peak_gpu_memory_allocated_bytes"]),
                "peak_gpu_memory_reserved_bytes": int(metrics["training"]["peak_gpu_memory_reserved_bytes"]),
                "selected_site_exact_zero_fractions": site_exact,
                "R_block": float(measured["R_block"]),
                "R_model": float(measured["R_model"]),
                "R_model_max": float(maximum["R_model_max_fraction"]),
                "checkpoint_content_sha256": str(checkpoint["content_sha256"]),
                "checkpoint_bytes": int(checkpoint["bytes"]),
                "condition_seconds": float(metrics["timing"]["total_seconds"]),
                "gradient_conflict_rate": sum(
                    bool(event["gradient_conflict"]) for event in train_events
                ) / len(train_events),
                "projection_rate": sum(
                    bool(event["projection_applied"]) for event in train_events
                ) / len(train_events),
                "trust_budget_saturation_rate": sum(
                    float(event["trust_scale"]) < 1.0 - 1.0e-12 for event in train_events
                ) / len(train_events),
                "final_pressure_loss": float(train_events[-1]["pressure_loss"]),
                "final_pressure_to_task_ratio": float(
                    train_events[-1]["pressure_to_task_ratio_final"]
                ),
            }
        )

    expected_git_commit = provenance.get("git_commit")
    expected_git_dirty = provenance.get("git_dirty")
    if len(initial_hashes) != 1 or "None" in initial_hashes:
        raise RuntimeError("Conditions did not share one identified initialization.")
    if schedule_hashes != {EXPECTED_SCHEDULE_SHA256}:
        raise RuntimeError("Conditions did not share Run 006's training schedule.")
    if code_hashes != {expected_code}:
        raise RuntimeError("Executed run-code identities differ from terminal verification code.")
    if git_commits != {expected_git_commit} or expected_git_commit is None:
        raise RuntimeError("Git commit identity differs from launch provenance.")
    if git_dirty_states != {expected_git_dirty} or expected_git_dirty is None:
        raise RuntimeError("Git dirty-state identity differs from launch provenance.")

    output = RUN_DIR / "artifacts" / "verification.json"
    summary = {
        "schema_version": 1,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "verifier": {"path": Path(__file__).name, "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "status": "verified",
        "evidence_label": "valid",
        "question": "joint A7-Z-POST mixed threshold plus all-site post-threshold OL1 dose response",
        "condition_count": len(rows),
        "total_optimizer_steps": max_steps * len(rows),
        "total_training_input_tokens": max_steps * tokens_per_update * len(rows),
        "complete_validation_passes": 3 * len(rows),
        "summed_condition_seconds": condition_seconds,
        "initial_parameter_sha256": next(iter(initial_hashes)),
        "training_schedule_sha256": next(iter(schedule_hashes)),
        "run_code_content_sha256": next(iter(code_hashes)),
        "git_commit": next(iter(git_commits)),
        "git_dirty": next(iter(git_dirty_states)),
        "checkpoint_total_bytes": checkpoint_bytes,
        "conditions": rows,
        "interpretation": {
            "logical_products": "opportunities, not removed FLOPs or speedup",
            "pressure": "unweighted mean across 42 post-threshold site-layer tensors",
            "scope": "one seed, one scale, local horizon; joint sites are not individually attributable",
        },
    }
    write_json(output, summary)
    write_json(
        RUN_DIR / "artifacts" / "progress.json",
        {
            "status": "verified",
            "condition_count": len(rows),
            "completed_conditions": len(rows),
            "completed_optimizer_steps": max_steps * len(rows),
            "total_optimizer_steps": max_steps * len(rows),
            "evidence_label": "valid",
        },
    )
    return output


def _one_attempt(root: Path, order: int) -> Path:
    matches = sorted(path for path in root.glob(f"{order:03d}-*") if path.is_dir())
    if len(matches) != 1:
        raise RuntimeError(f"Expected one attempt for condition order {order}, found {len(matches)}.")
    return matches[0]


def _require_ol1_events(events: list[Mapping[str, Any]], attempt_id: str) -> None:
    required = {
        "pressure_loss",
        "task_gradient_norm",
        "pressure_gradient_norm",
        "task_pressure_gradient_dot",
        "task_pressure_gradient_cosine",
        "gradient_conflict",
        "task_direction_norm",
        "pressure_direction_norm_raw",
        "task_pressure_dot_before",
        "task_pressure_dot_after",
        "projection_applied",
        "pressure_to_task_ratio_raw",
        "trust_scale",
        "pressure_to_task_ratio_final",
    }
    for event in events:
        missing = required.difference(event)
        if missing:
            raise RuntimeError(f"OL1 fields missing in {attempt_id}: {sorted(missing)}")
        for key in required.difference({"gradient_conflict", "projection_applied"}):
            _finite(event[key], f"{attempt_id} {key}")
        if not 0.0 <= float(event["trust_scale"]) <= 1.0:
            raise RuntimeError(f"Invalid OL1 trust scale in {attempt_id}.")
        if float(event["pressure_to_task_ratio_final"]) > 1.0 + 1.0e-6:
            raise RuntimeError(f"OL1 trust budget exceeded in {attempt_id}.")


def _require_validation(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label}: validation record missing.")
    expected = {
        "sequences": 338,
        "input_tokens": 692_224,
        "excluded_tail_tokens": 1_444,
        "complete_block_coverage": True,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise RuntimeError(f"{label}: validation coverage mismatch.")
    _finite(value.get("loss"), f"{label} loss")


def _finite(value: Any, label: str) -> None:
    if not math.isfinite(float(value)):
        raise RuntimeError(f"{label}: non-finite value.")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected YAML mapping: {path}")
    return value

