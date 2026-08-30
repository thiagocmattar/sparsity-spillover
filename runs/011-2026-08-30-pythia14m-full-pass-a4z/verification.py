"""Per-worker and cohort verification for paper-scale Run 011 A4-Z evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from sparsity_research.artifacts import build_transfer_inventory, verify_transfer_inventory

from run_config import (
    EXPECTED_ACTIVE_SITES,
    EXPECTED_CEILING_DENOMINATOR,
    EXPECTED_CEILING_FRACTION,
    EXPECTED_CEILING_NUMERATOR,
    EXPECTED_INITIAL_PARAMETER_SHA256,
    EXPECTED_MODEL_CHECKPOINTS,
    EXPECTED_OPTIMIZER_CHECKPOINTS,
    EXPECTED_RUN004_CODE_SHA256,
    EXPECTED_SCHEDULE_SHA256,
    RUN004_VERIFICATION,
    RUN_DIR,
    condition_specs,
    inventory_content_sha256,
    load_config,
    mapping,
    write_json,
)


OUTPUT = RUN_DIR / "artifacts" / "verification.json"
LOGICAL_OPERATIONS = {
    "qkv_projection",
    "qk_scores",
    "probability_value",
    "attention_output_projection",
    "mlp_w1",
    "mlp_w2",
}


def verify_attempt(condition_id: str) -> dict[str, Any]:
    """Verify one independently retrieved worker without requiring peer attempts."""

    config = load_config()
    by_id = {row["id"]: row for row in condition_specs(config)}
    if condition_id not in by_id:
        raise KeyError(f"Unknown Run 011 condition {condition_id!r}.")
    condition = by_id[condition_id]
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
    if list(manifest.get("gradient_overflow_steps", [])):
        raise ValueError(f"FP16 overflow skipped optimizer steps in {attempt_dir.name}.")

    _require_train_events(attempt_dir, condition)
    _require_validation(metrics, attempt_dir.name)
    activation, logical = _require_diagnostics(attempt_dir, metrics, config)
    _require_checkpoints(attempt_dir, metrics)
    transfer = _json(attempt_dir / "transfer_inventory.json")
    verify_transfer_inventory(attempt_dir, transfer)

    pooled_by_name = {row["name"]: row for row in activation["pooled_by_site"]}
    measured = logical["measured"]
    maximum = logical["architecture_maximum"]
    return {
        "status": "verified",
        "evidence_label": "valid",
        "attempt_id": attempt_dir.name,
        "condition": condition,
        "worker_id": manifest["worker_id"],
        "completed_steps": 712,
        "input_tokens": 1_493_172_224,
        "initial_parameter_sha256": str(manifest["initial_parameter_sha256"]),
        "training_schedule_sha256": str(manifest["training_schedule_hash"]),
        "run_code_sha256": str(manifest["run_code"]["content_sha256"]),
        "git_commit": manifest.get("code", {}).get("git_commit"),
        "git_dirty": manifest.get("code", {}).get("git_dirty"),
        "final_train_loss": float(metrics["training"]["task_loss_final"]),
        "final_validation_loss": float(metrics["validation"]["final"]["loss"]),
        "median_step_seconds": float(metrics["training"]["median_step_seconds"]),
        "median_tokens_per_second": float(metrics["training"]["median_tokens_per_second"]),
        "peak_gpu_memory_allocated_bytes": int(
            metrics["training"]["peak_gpu_memory_allocated_bytes"]
        ),
        "peak_gpu_memory_reserved_bytes": int(
            metrics["training"]["peak_gpu_memory_reserved_bytes"]
        ),
        "selected_site_exact_zero_fractions": {
            site: float(pooled_by_name[site]["exact_zero_fraction"])
            for site in EXPECTED_ACTIVE_SITES
        },
        "R_block": float(measured["R_block"]),
        "R_model": float(measured["R_model"]),
        "R_model_max": float(maximum["R_model_max_fraction"]),
        "checkpoint_content_sha256": str(metrics["checkpoints"]["final"]["content_sha256"]),
        "checkpoint_bytes": int(metrics["checkpoints"]["total_bytes"]),
        "transfer_bytes": int(transfer["total_bytes"]),
        "condition_seconds": float(metrics["timing"]["total_seconds"]),
    }


def verify_run() -> dict[str, Any]:
    config = load_config()
    rows = [verify_attempt(condition["id"]) for condition in condition_specs(config)]
    initial_hashes = {row["initial_parameter_sha256"] for row in rows}
    schedule_hashes = {row["training_schedule_sha256"] for row in rows}
    code_hashes = {row["run_code_sha256"] for row in rows}
    git_commits = {row["git_commit"] for row in rows}
    git_dirty_states = {row["git_dirty"] for row in rows}
    if initial_hashes != {EXPECTED_INITIAL_PARAMETER_SHA256}:
        raise ValueError("Run 011 initialization does not match Run 004.")
    if schedule_hashes != {EXPECTED_SCHEDULE_SHA256}:
        raise ValueError("Run 011 schedule does not match Run 004.")
    if len(code_hashes) != 1:
        raise ValueError("Conditions did not share one Run 011 code identity.")
    if len(git_commits) != 1 or len(git_dirty_states) != 1:
        raise ValueError("Conditions did not share one repository identity.")

    source = _load_run004_verification()
    relu_control = next(
        row for row in source["conditions"] if row.get("condition", {}).get("id") == "relu-control"
    )
    public_rows = [
        {key: value for key, value in row.items() if key not in {"status", "evidence_label"}}
        for row in rows
    ]
    result = {
        "schema_version": 1,
        "status": "verified",
        "evidence_label": "valid",
        "question": "paper-scale A4-Z one-sided threshold dose response without pressure",
        "condition_count": len(public_rows),
        "completed_optimizer_steps": 712 * len(public_rows),
        "training_input_tokens": 1_493_172_224 * len(public_rows),
        "complete_validation_passes": 4 * len(public_rows),
        "initial_parameter_sha256": next(iter(initial_hashes)),
        "training_schedule_sha256": next(iter(schedule_hashes)),
        "run_code_sha256": next(iter(code_hashes)),
        "git_commit": next(iter(git_commits)),
        "git_dirty": next(iter(git_dirty_states)),
        "conditions": public_rows,
        "comparison": {
            "source_run": "runs/004-2026-08-29-pythia14m-full-pass-l1n",
            "source_verification": (
                "runs/004-2026-08-29-pythia14m-full-pass-l1n/artifacts/verification.json"
            ),
            "reused_a1_h_condition": {
                "condition_id": "relu-control",
                "attempt_id": relu_control["attempt_id"],
                "final_validation_loss": float(relu_control["final_validation_loss"]),
            },
        },
        "interpretation": {
            "engine": "Transformers/PyTorch mapping of Pythia-14M recipe; not GPT-NeoX-bitwise",
            "comparison": "A4 kappa=0 versus the matched Run 004 A1-H ReLU control",
            "logical_products": "opportunities, not removed FLOPs or speedup",
            "parallelism": "five independent one-GPU conditions; no DDP",
            "scope": "one seed, one scale, one MiniPile pass; joint sites are not individually attributable",
        },
    }
    write_json(OUTPUT, result)
    return result


def _require_train_events(attempt_dir: Path, condition: Mapping[str, Any]) -> None:
    with (attempt_dir / "events.jsonl").open("rb") as handle:
        events = [json.loads(line) for line in handle.read().splitlines() if line]
    train = [row for row in events if row.get("event") == "train"]
    if [int(row.get("step", -1)) for row in train] != list(range(1, 713)):
        raise ValueError(f"Incomplete train event history for {attempt_dir.name}.")
    pressure_fields = {
        "pressure_loss",
        "pressure_weight",
        "weighted_pressure_loss",
        "augmented_loss",
        "task_pressure_dot",
        "task_pressure_cosine",
    }
    finite_fields = (
        "task_loss",
        "step_wall_seconds",
        "tokens_per_second",
        "learning_rate",
        "loss_scale",
        "adamw_gradient_norm_pre_clip",
        "adamw_gradient_norm_post_clip",
    )
    for row in train:
        if row.get("condition_id") != condition["id"]:
            raise ValueError(f"Condition mismatch in {attempt_dir.name} event history.")
        if row.get("optimizer_step_skipped") is not False or row.get("gradient_overflow") is not False:
            raise ValueError(f"Skipped or overflowed boundary in {attempt_dir.name}.")
        if any(not math.isfinite(float(row.get(key, math.nan))) for key in finite_fields):
            raise ValueError(f"Non-finite training metric in {attempt_dir.name}.")
        if float(row["adamw_gradient_norm_post_clip"]) > 1.0 + 1e-5:
            raise ValueError(f"Gradient clip exceeded in {attempt_dir.name}.")
        if pressure_fields.intersection(row):
            raise ValueError(f"Pressure metrics unexpectedly present in {attempt_dir.name}.")


def _require_validation(metrics: Mapping[str, Any], attempt_id: str) -> None:
    expected = {"sequences": 338, "input_tokens": 692_224, "excluded_tail_tokens": 1_444}
    validation = metrics.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError(f"Missing validation metrics for {attempt_id}.")
    for name in ("step_one", "final", "activation_diagnostic", "logical_product_diagnostic_eager"):
        row = validation.get(name)
        if not isinstance(row, Mapping) or any(
            int(row.get(key, -1)) != value for key, value in expected.items()
        ):
            raise ValueError(f"Validation coverage mismatch for {attempt_id}/{name}.")
        if not math.isfinite(float(row.get("loss", math.nan))):
            raise ValueError(f"Non-finite validation loss for {attempt_id}/{name}.")


def _require_diagnostics(
    attempt_dir: Path, metrics: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    activation = _json(attempt_dir / "diagnostics" / "activation_statistics.json")
    rows = activation.get("rows")
    pooled = activation.get("pooled_by_site")
    requested = list(mapping(config, "diagnostics")["activation_sites"])
    if not isinstance(rows, list) or len(rows) != 6 * len(requested):
        raise ValueError(f"Incomplete activation rows for {attempt_dir.name}.")
    if not isinstance(pooled, list) or {row["name"] for row in pooled} != set(requested):
        raise ValueError(f"Activation site mismatch for {attempt_dir.name}.")
    if any(int(row.get("nonfinite", -1)) != 0 for row in rows):
        raise ValueError(f"Non-finite activations for {attempt_dir.name}.")
    if activation.get("site_definition", {}).get("z") != (
        "concatenated PV context immediately before attention.dense (W_o)"
    ):
        raise ValueError(f"z boundary definition missing for {attempt_dir.name}.")

    weights = _json(attempt_dir / "diagnostics" / "weight_statistics.json")
    if weights.get("inclusion") != "all named parameters, including bias and normalization":
        raise ValueError(f"Weight inclusion rule missing for {attempt_dir.name}.")
    weight_rows = weights.get("rows")
    if not isinstance(weight_rows, list) or not weight_rows:
        raise ValueError(f"Weight statistics missing for {attempt_dir.name}.")
    if any(int(row.get("nonfinite", -1)) != 0 for row in weight_rows):
        raise ValueError(f"Non-finite weight statistics for {attempt_dir.name}.")

    logical = _json(attempt_dir / "diagnostics" / "logical_products.json")
    measured = logical.get("measured")
    maximum = logical.get("architecture_maximum")
    if not isinstance(measured, Mapping) or not isinstance(maximum, Mapping):
        raise ValueError(f"Logical-product artifacts missing for {attempt_dir.name}.")
    if set(measured.get("per_operation", {})) != LOGICAL_OPERATIONS:
        raise ValueError(f"Logical operations are incomplete for {attempt_dir.name}.")
    for key in (
        "block_zero_product_count",
        "block_product_count",
        "lm_head_product_count",
        "model_product_count",
    ):
        if int(measured.get(key, -1)) < 0:
            raise ValueError(f"Logical counter {key} missing for {attempt_dir.name}.")
    for key in ("R_block", "R_model"):
        value = float(measured.get(key, math.nan))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Logical fraction {key} invalid for {attempt_dir.name}.")
    if maximum.get("topology_id") != "A4-Z":
        raise ValueError(f"Architecture ceiling topology mismatch for {attempt_dir.name}.")
    if int(maximum.get("reachable_product_count", -1)) != EXPECTED_CEILING_NUMERATOR:
        raise ValueError(f"Architecture ceiling numerator mismatch for {attempt_dir.name}.")
    if int(maximum.get("model_product_count", -1)) != EXPECTED_CEILING_DENOMINATOR:
        raise ValueError(f"Architecture ceiling denominator mismatch for {attempt_dir.name}.")
    if not math.isclose(
        float(maximum.get("R_model_max_fraction", math.nan)),
        EXPECTED_CEILING_FRACTION,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(f"Architecture ceiling fraction mismatch for {attempt_dir.name}.")
    if metrics.get("diagnostics", {}).get("logical_products_path") != (
        "diagnostics/logical_products.json"
    ):
        raise ValueError(f"Diagnostic manifest mismatch for {attempt_dir.name}.")
    return activation, logical


def _require_checkpoints(attempt_dir: Path, metrics: Mapping[str, Any]) -> None:
    checkpoints = metrics.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise ValueError(f"Checkpoint metrics missing for {attempt_dir.name}.")
    if checkpoints.get("model_steps") != list(EXPECTED_MODEL_CHECKPOINTS):
        raise ValueError(f"Model checkpoint cadence mismatch for {attempt_dir.name}.")
    if checkpoints.get("optimizer_steps") != list(EXPECTED_OPTIMIZER_CHECKPOINTS):
        raise ValueError(f"Optimizer checkpoint cadence mismatch for {attempt_dir.name}.")
    rebuilt = build_transfer_inventory(attempt_dir / "checkpoints")
    if inventory_content_sha256(rebuilt) != checkpoints.get("inventory_content_sha256"):
        raise ValueError(f"Checkpoint inventory hash mismatch for {attempt_dir.name}.")
    final = checkpoints.get("final")
    if not isinstance(final, Mapping):
        raise ValueError(f"Final checkpoint metadata missing for {attempt_dir.name}.")
    rebuilt_final = build_transfer_inventory(attempt_dir / str(final["path"]))
    if inventory_content_sha256(rebuilt_final) != final.get("content_sha256"):
        raise ValueError(f"Final checkpoint hash mismatch for {attempt_dir.name}.")
    if not (attempt_dir / str(final["path"]) / "training_state.pt").is_file():
        raise ValueError(f"Final optimizer recovery state missing for {attempt_dir.name}.")


def _load_run004_verification() -> dict[str, Any]:
    source = _json(RUN004_VERIFICATION)
    expected = {
        "status": "verified",
        "evidence_label": "valid",
        "condition_count": 6,
        "initial_parameter_sha256": EXPECTED_INITIAL_PARAMETER_SHA256,
        "training_schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "run_code_sha256": EXPECTED_RUN004_CODE_SHA256,
    }
    if any(source.get(key) != value for key, value in expected.items()):
        raise ValueError("Run 004 comparator identity changed.")
    ids = [row.get("condition", {}).get("id") for row in source.get("conditions", [])]
    if ids != [
        "gelu-control",
        "relu-control",
        "relu-l1n-0p05",
        "relu-l1n-0p1",
        "relu-l1n-0p5",
        "relu-l1n-1",
    ]:
        raise ValueError("Run 004 comparator condition order changed.")
    return source


def _one_attempt(root: Path, order: int) -> Path:
    matches = sorted(path for path in root.glob(f"{order:03d}-*") if path.is_dir())
    if len(matches) != 1:
        raise ValueError(f"Expected one attempt for condition order {order}, found {len(matches)}.")
    return matches[0]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value
