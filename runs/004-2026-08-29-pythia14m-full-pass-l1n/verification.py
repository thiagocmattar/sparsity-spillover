"""Terminal cross-Pod verification for Run 004."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from sparsity_research.artifacts import build_transfer_inventory, verify_transfer_inventory

from run_config import (
    RUN_DIR,
    condition_specs,
    inventory_content_sha256,
    load_config,
    mapping,
    write_json,
)


OUTPUT = RUN_DIR / "artifacts" / "verification.json"


def verify_run() -> dict[str, Any]:
    config = load_config()
    attempts_root = RUN_DIR / "artifacts" / "attempts"
    rows = []
    initial_hashes = set()
    schedule_hashes = set()
    code_hashes = set()
    for condition in condition_specs(config):
        attempt_dir = _one_attempt(attempts_root, int(condition["order"]))
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
        _require_validation(metrics, attempt_dir.name)
        _require_diagnostics(attempt_dir, metrics, config)
        _require_checkpoints(attempt_dir, metrics)
        transfer = _json(attempt_dir / "transfer_inventory.json")
        verify_transfer_inventory(attempt_dir, transfer)
        initial_hashes.add(str(manifest["initial_parameter_sha256"]))
        schedule_hashes.add(str(manifest["training_schedule_hash"]))
        code_hashes.add(str(manifest["run_code"]["content_sha256"]))
        rows.append(
            {
                "attempt_id": attempt_dir.name,
                "condition": condition,
                "worker_id": manifest["worker_id"],
                "final_validation_loss": float(metrics["validation"]["final"]["loss"]),
                "median_tokens_per_second": float(metrics["training"]["median_tokens_per_second"]),
                "peak_gpu_memory_allocated_bytes": int(metrics["training"]["peak_gpu_memory_allocated_bytes"]),
                "peak_gpu_memory_reserved_bytes": int(metrics["training"]["peak_gpu_memory_reserved_bytes"]),
                "checkpoint_content_sha256": metrics["checkpoints"]["final"]["content_sha256"],
                "transfer_bytes": int(transfer["total_bytes"]),
            }
        )
    if len(initial_hashes) != 1:
        raise ValueError("Conditions did not share one initial parameter hash.")
    if len(schedule_hashes) != 1:
        raise ValueError("Conditions did not share one full-pass schedule hash.")
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
            "engine": "Transformers/PyTorch mapping of Pythia-14M recipe; not GPT-NeoX-bitwise",
            "logical_products": "opportunities, not removed FLOPs or speedup",
            "scope": "one seed, one scale, one MiniPile pass",
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
    expected = {"sequences": 338, "input_tokens": 692_224, "excluded_tail_tokens": 1_444}
    validation = metrics.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError(f"Missing validation metrics for {attempt_id}.")
    for name in ("step_one", "final", "activation_diagnostic", "logical_product_diagnostic_eager"):
        row = validation.get(name)
        if not isinstance(row, Mapping) or any(int(row.get(key, -1)) != value for key, value in expected.items()):
            raise ValueError(f"Validation coverage mismatch for {attempt_id}/{name}.")
        if not math.isfinite(float(row.get("loss", math.nan))):
            raise ValueError(f"Non-finite validation loss for {attempt_id}/{name}.")


def _require_diagnostics(attempt_dir: Path, metrics: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    activation = _json(attempt_dir / "diagnostics" / "activation_statistics.json")
    rows = activation.get("rows")
    requested = list(mapping(config, "diagnostics")["activation_sites"])
    if not isinstance(rows, list) or len(rows) != 6 * len(requested):
        raise ValueError(f"Incomplete activation rows for {attempt_dir.name}.")
    if {row["name"].split(".layer_", 1)[0] for row in rows} != set(requested):
        raise ValueError(f"Activation site mismatch for {attempt_dir.name}.")
    if any(int(row.get("nonfinite", -1)) != 0 for row in rows):
        raise ValueError(f"Non-finite activations for {attempt_dir.name}.")
    weights = _json(attempt_dir / "diagnostics" / "weight_statistics.json")
    if weights.get("inclusion") != "all named parameters, including bias and normalization":
        raise ValueError(f"Weight inclusion rule missing for {attempt_dir.name}.")
    logical = _json(attempt_dir / "diagnostics" / "logical_products.json")
    measured = logical.get("measured", {})
    ceiling = logical.get("architecture_maximum", {})
    for key in ("block_zero_product_count", "block_product_count", "lm_head_product_count", "model_product_count"):
        if int(measured.get(key, -1)) < 0:
            raise ValueError(f"Logical counter {key} missing for {attempt_dir.name}.")
    if ceiling.get("topology_id") not in {"A0", "A1-H"}:
        raise ValueError(f"Architecture ceiling topology mismatch for {attempt_dir.name}.")
    if metrics.get("diagnostics", {}).get("logical_products_path") != "diagnostics/logical_products.json":
        raise ValueError(f"Diagnostic manifest mismatch for {attempt_dir.name}.")


def _require_checkpoints(attempt_dir: Path, metrics: Mapping[str, Any]) -> None:
    checkpoints = metrics.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise ValueError(f"Checkpoint metrics missing for {attempt_dir.name}.")
    expected_model = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 712]
    if checkpoints.get("model_steps") != expected_model or checkpoints.get("optimizer_steps") != [256, 512, 712]:
        raise ValueError(f"Checkpoint cadence mismatch for {attempt_dir.name}.")
    rebuilt = build_transfer_inventory(attempt_dir / "checkpoints")
    if inventory_content_sha256(rebuilt) != checkpoints.get("inventory_content_sha256"):
        raise ValueError(f"Checkpoint inventory hash mismatch for {attempt_dir.name}.")
    final = checkpoints["final"]
    rebuilt_final = build_transfer_inventory(attempt_dir / final["path"])
    if inventory_content_sha256(rebuilt_final) != final["content_sha256"]:
        raise ValueError(f"Final checkpoint hash mismatch for {attempt_dir.name}.")
    if not (attempt_dir / final["path"] / "training_state.pt").is_file():
        raise ValueError(f"Final optimizer recovery state missing for {attempt_dir.name}.")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value

