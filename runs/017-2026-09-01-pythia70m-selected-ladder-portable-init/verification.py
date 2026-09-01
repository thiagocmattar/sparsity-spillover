"""Per-condition and terminal cohort verification for Run 017."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from sparsity_research.artifacts import build_transfer_inventory, verify_transfer_inventory

from initialization import EXPECTED_ARCHITECTURE
from run_config import (
    A4_SITES,
    A7_SITES,
    DIAGNOSTIC_SITES,
    EXPECTED_INITIAL_PARAMETER_SHA256,
    EXPECTED_SCHEDULE_SHA256,
    RUN_DIR,
    condition_specs,
    expected_ceiling,
    inventory_content_sha256,
    load_config,
    mapping,
    write_json,
)


OUTPUT = RUN_DIR / "artifacts" / "verification.json"


def verify_attempt(condition_id: str) -> dict[str, Any]:
    config = load_config()
    condition = next(row for row in condition_specs(config) if row["id"] == condition_id)
    attempt_dir = _one_attempt(RUN_DIR / "artifacts" / "attempts", int(condition["order"]))
    manifest = _json(attempt_dir / "manifest.json")
    metrics = _json(attempt_dir / "metrics.json")
    if manifest.get("status") != "completed" or manifest.get("attempt_id") != attempt_dir.name:
        raise ValueError(f"Attempt {attempt_dir.name} is not terminally completed.")
    if manifest.get("condition") != condition or metrics.get("condition") != condition:
        raise ValueError(f"Condition identity mismatch for {attempt_dir.name}.")
    if int(manifest.get("completed_steps", -1)) != 712:
        raise ValueError(f"Incomplete training steps for {attempt_dir.name}.")
    if int(manifest.get("input_tokens", -1)) != 1_493_172_224:
        raise ValueError(f"Training token count mismatch for {attempt_dir.name}.")
    overflow = list(manifest.get("gradient_overflow_steps", []))
    if overflow:
        raise ValueError(f"FP16 overflow skipped optimizer steps in {attempt_dir.name}: {overflow}")
    if manifest.get("initial_parameter_sha256") != EXPECTED_INITIAL_PARAMETER_SHA256:
        raise ValueError(f"Pinned portable CPU initialization mismatch for {attempt_dir.name}.")
    if manifest.get("training_schedule_hash") != EXPECTED_SCHEDULE_SHA256:
        raise ValueError(f"Pinned training schedule mismatch for {attempt_dir.name}.")
    _require_validation(metrics, attempt_dir.name)
    _require_diagnostics(attempt_dir, metrics, config, condition)
    _require_boundaries(attempt_dir, condition)
    _require_checkpoint(attempt_dir, metrics)
    transfer = _json(attempt_dir / "transfer_inventory.json")
    verify_transfer_inventory(attempt_dir, transfer)
    return {
        "status": "verified",
        "attempt_id": attempt_dir.name,
        "condition": condition,
        "worker_id": manifest["worker_id"],
        "initial_parameter_sha256": str(manifest["initial_parameter_sha256"]),
        "training_schedule_sha256": str(manifest["training_schedule_hash"]),
        "run_code_sha256": str(manifest["run_code"]["content_sha256"]),
        "final_validation_loss": float(metrics["validation"]["final"]["loss"]),
        "R_model": float(
            _json(attempt_dir / "diagnostics" / "logical_products.json")["measured"]["R_model"]
        ),
        "R_model_max": float(
            _json(attempt_dir / "diagnostics" / "logical_products.json")["architecture_maximum"]["R_model_max_fraction"]
        ),
        "median_tokens_per_second": float(metrics["training"]["median_tokens_per_second"]),
        "peak_gpu_memory_allocated_bytes": int(metrics["training"]["peak_gpu_memory_allocated_bytes"]),
        "peak_gpu_memory_reserved_bytes": int(metrics["training"]["peak_gpu_memory_reserved_bytes"]),
        "checkpoint_content_sha256": metrics["checkpoints"]["final"]["content_sha256"],
        "transfer_bytes": int(transfer["total_bytes"]),
    }


def verify_run() -> dict[str, Any]:
    config = load_config()
    rows = [verify_attempt(condition["id"]) for condition in condition_specs(config)]
    initial_hashes = {row["initial_parameter_sha256"] for row in rows}
    schedule_hashes = {row["training_schedule_sha256"] for row in rows}
    code_hashes = {row["run_code_sha256"] for row in rows}
    if len(initial_hashes) != 1:
        raise ValueError("Conditions did not share one random initial parameter hash.")
    if initial_hashes != {EXPECTED_INITIAL_PARAMETER_SHA256}:
        raise ValueError("The cohort does not match the pinned portable CPU initialization hash.")
    if len(schedule_hashes) != 1:
        raise ValueError("Conditions did not share one full-pass data schedule hash.")
    if schedule_hashes != {EXPECTED_SCHEDULE_SHA256}:
        raise ValueError("The cohort does not match the pinned Run 017 schedule hash.")
    if len(code_hashes) != 1:
        raise ValueError("Conditions did not share one run-code identity.")
    result = {
        "schema_version": 1,
        "status": "verified",
        "evidence_label": "valid",
        "condition_count": len(rows),
        "completed_optimizer_steps": 712 * len(rows),
        "training_input_tokens": 1_493_172_224 * len(rows),
        "initial_parameter_sha256": next(iter(initial_hashes)),
        "training_schedule_sha256": next(iter(schedule_hashes)),
        "run_code_sha256": next(iter(code_hashes)),
        "conditions": rows,
        "interpretation": {
            "engine": "Transformers/PyTorch mapping of Pythia-70M recipe; not GPT-NeoX-bitwise",
            "logical_products": "R_block/R_model are opportunities, not removed FLOPs or speedup",
            "scope": "one seed, one scale, one MiniPile pass",
            "teal": "A0/A1 post-hoc TEAL is verified separately after this training cohort",
        },
    }
    write_json(OUTPUT, result)
    return result


def _one_attempt(root: Path, order: int) -> Path:
    matches = sorted(path for path in root.glob(f"{order:03d}-*") if path.is_dir())
    if len(matches) != 1:
        raise ValueError(f"Expected one attempt for condition order {order}, found {len(matches)}.")
    return matches[0]


def _require_validation(metrics: Mapping[str, Any], attempt_id: str) -> None:
    expected = {
        "sequences": 338,
        "input_tokens": 692_224,
        "excluded_tail_tokens": 1_444,
        "complete_block_coverage": True,
    }
    validation = metrics.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError(f"Missing validation metrics for {attempt_id}.")
    for name in ("step_one", "final", "activation_diagnostic", "logical_product_diagnostic_eager"):
        row = validation.get(name)
        if not isinstance(row, Mapping) or any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Validation coverage mismatch for {attempt_id}/{name}.")
        if not math.isfinite(float(row.get("loss", math.nan))):
            raise ValueError(f"Non-finite validation loss for {attempt_id}/{name}.")


def _require_diagnostics(
    attempt_dir: Path,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    condition: Mapping[str, Any],
) -> None:
    activation = _json(attempt_dir / "diagnostics" / "activation_statistics.json")
    rows = activation.get("rows")
    pooled = activation.get("pooled_by_site")
    requested = list(mapping(config, "diagnostics")["activation_sites"])
    if requested != list(DIAGNOSTIC_SITES):
        raise ValueError("Diagnostic site identity changed during verification.")
    layer_count = int(EXPECTED_ARCHITECTURE["num_hidden_layers"])
    if not isinstance(rows, list) or len(rows) != layer_count * len(requested):
        raise ValueError(f"Incomplete activation rows for {attempt_dir.name}.")
    if {row["name"].split(".layer_", 1)[0] for row in rows} != set(requested):
        raise ValueError(f"Activation site mismatch for {attempt_dir.name}.")
    if any(int(row.get("nonfinite", -1)) != 0 for row in rows):
        raise ValueError(f"Non-finite activations for {attempt_dir.name}.")
    if not isinstance(pooled, list) or {row.get("name") for row in pooled} != set(requested):
        raise ValueError(f"Per-site pooled zero-mass rows are incomplete for {attempt_dir.name}.")
    for site_row in pooled:
        selected = [row for row in rows if row["name"].startswith(f"{site_row['name']}.layer_")]
        if int(site_row["total"]) != sum(int(row["total"]) for row in selected):
            raise ValueError(f"Per-site activation denominator does not reconcile for {attempt_dir.name}.")
        if int(site_row["exact_zero_count"]) != sum(int(row["exact_zero_count"]) for row in selected):
            raise ValueError(f"Per-site exact-zero count does not reconcile for {attempt_dir.name}.")

    weights = _json(attempt_dir / "diagnostics" / "weight_statistics.json")
    if weights.get("inclusion") != "all named parameters, including bias and normalization":
        raise ValueError(f"Weight inclusion rule missing for {attempt_dir.name}.")
    if not weights.get("rows") or not isinstance(weights.get("pooled"), Mapping):
        raise ValueError(f"Weight diagnostics are incomplete for {attempt_dir.name}.")

    logical = _json(attempt_dir / "diagnostics" / "logical_products.json")
    measured = logical.get("measured", {})
    ceiling = logical.get("architecture_maximum", {})
    expected = expected_ceiling(str(condition["topology_id"]))
    for key in (
        "block_zero_product_count", "block_product_count", "lm_head_product_count", "model_product_count"
    ):
        if int(measured.get(key, -1)) < 0:
            raise ValueError(f"Logical counter {key} missing for {attempt_dir.name}.")
    operations = measured.get("per_operation", {})
    if sum(int(row["zero_product_count"]) for row in operations.values()) != int(measured["block_zero_product_count"]):
        raise ValueError(f"Logical zero-product counts do not reconcile for {attempt_dir.name}.")
    if sum(int(row["product_count"]) for row in operations.values()) != int(measured["block_product_count"]):
        raise ValueError(f"Logical product denominators do not reconcile for {attempt_dir.name}.")
    for key in (
        "topology_id", "reachable_product_count", "block_product_count", "lm_head_product_count",
        "model_product_count", "R_model_max_fraction",
    ):
        if ceiling.get(key) != expected.get(key):
            raise ValueError(f"R_model_max field {key} changed for {attempt_dir.name}.")
    if not 0.0 <= float(measured.get("R_model", math.nan)) <= float(ceiling["R_model_max_fraction"]) + 1e-12:
        raise ValueError(f"Measured R_model exceeds its declared reach ceiling for {attempt_dir.name}.")
    if metrics.get("diagnostics", {}).get("logical_products_path") != "diagnostics/logical_products.json":
        raise ValueError(f"Diagnostic manifest mismatch for {attempt_dir.name}.")


def _require_boundaries(attempt_dir: Path, condition: Mapping[str, Any]) -> None:
    events = [json.loads(line) for line in (attempt_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    training = [row for row in events if row.get("event") == "train"]
    if len(training) != 712 or [int(row["step"]) for row in training] != list(range(1, 713)):
        raise ValueError(f"Training boundary coverage is incomplete for {attempt_dir.name}.")
    if condition["pressure_method"] == "none":
        if any("ol1_correction_applied" in row for row in training):
            raise ValueError(f"Control unexpectedly records OL1 for {attempt_dir.name}.")
        return
    sites = tuple(condition["pressure_sites"])
    if sites not in {A4_SITES, A7_SITES}:
        raise ValueError(f"Unexpected OL1 sites for {attempt_dir.name}.")
    layer_count = int(EXPECTED_ARCHITECTURE["num_hidden_layers"])
    expected_names = tuple(
        sorted(f"{site}.layer_{layer}" for site in sites for layer in range(layer_count))
    )
    expected_hash = hashlib.sha256("\n".join(expected_names).encode("utf-8")).hexdigest()
    finite_keys = (
        "task_gradient_norm", "pressure_gradient_norm", "pressure_to_task_gradient_norm_ratio",
        "task_pressure_gradient_dot", "task_pressure_gradient_cosine",
    )
    for row in training:
        if (
            row.get("ol1_correction_applied") is not True
            or row.get("pressure_sites") != list(sites)
            or int(row.get("pressure_capture_tensor_count", -1)) != len(expected_names)
            or row.get("pressure_capture_names_sha256") != expected_hash
            or any(not math.isfinite(float(row.get(key, math.nan))) for key in finite_keys)
        ):
            raise ValueError(f"Incomplete OL1 boundary diagnostics at {attempt_dir.name}/step {row.get('step')}.")


def _require_checkpoint(attempt_dir: Path, metrics: Mapping[str, Any]) -> None:
    checkpoints = metrics.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise ValueError(f"Checkpoint metrics missing for {attempt_dir.name}.")
    if checkpoints.get("model_steps") != [712] or checkpoints.get("optimizer_steps") != [712]:
        raise ValueError(f"Final-only checkpoint policy changed for {attempt_dir.name}.")
    checkpoint_root = attempt_dir / "checkpoints"
    children = sorted(path.name for path in checkpoint_root.iterdir() if path.is_dir())
    if children != ["step_000712"]:
        raise ValueError(f"Unexpected retained checkpoints for {attempt_dir.name}: {children}")
    rebuilt = build_transfer_inventory(checkpoint_root)
    if inventory_content_sha256(rebuilt) != checkpoints.get("inventory_content_sha256"):
        raise ValueError(f"Checkpoint inventory hash mismatch for {attempt_dir.name}.")
    final = checkpoints["final"]
    final_dir = attempt_dir / final["path"]
    rebuilt_final = build_transfer_inventory(final_dir)
    if inventory_content_sha256(rebuilt_final) != final["content_sha256"]:
        raise ValueError(f"Final checkpoint hash mismatch for {attempt_dir.name}.")
    state_path = final_dir / "training_state.pt"
    metadata = _json(final_dir / "checkpoint_metadata.json")
    if not state_path.is_file() or metadata.get("rng_saved") != ["python", "numpy", "torch_cpu", "torch_cuda_all"]:
        raise ValueError(f"Complete optimizer/scaler/RNG recovery state is missing for {attempt_dir.name}.")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value
