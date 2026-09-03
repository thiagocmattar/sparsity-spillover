"""Predeclared training-only learning-rate selection for Run 021."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from sparsity_research.artifacts import verify_transfer_inventory

from run_config import (
    EXPECTED_INITIAL_PARAMETER_SHA256,
    EXPECTED_SCHEDULE_SHA256,
    REPO_ROOT,
    RUN_DIR,
    condition_specs,
    load_config,
    mapping,
    write_json,
)
from verification import verify_run


OUTPUT = RUN_DIR / "artifacts" / "selection.json"
FINAL_WINDOW = 64


def select_learning_rate() -> dict[str, Any]:
    """Verify all inputs and apply the approved final-64 training-loss rule."""

    config = load_config()
    verified = verify_run()
    rows = [_baseline_row(config)]
    by_id = {row["condition"]["id"]: row for row in verified["conditions"]}
    for condition in condition_specs(config):
        verified_row = by_id[condition["id"]]
        attempt_dir = RUN_DIR / "artifacts" / "attempts" / verified_row["attempt_id"]
        rows.append(
            _training_row(
                attempt_dir,
                condition_id=condition["id"],
                peak=float(condition["peak_learning_rate"]),
                minimum=float(condition["minimum_learning_rate"]),
                validation_loss=float(verified_row["final_validation_loss"]),
                r_model=float(verified_row["R_model"]),
                provenance="run021_new_full_pass",
            )
        )

    status, selected_condition_id, best, next_action = selection_decision(rows)

    result = {
        "schema_version": 1,
        "status": status,
        "selection_metric_name": "mean_task_loss_final_64_optimizer_boundaries",
        "selection_uses_validation": False,
        "selected_condition_id": selected_condition_id,
        "best_observed_condition_id": str(best["condition_id"]),
        "next_action": next_action,
        "arms": rows,
        "shared_initial_parameter_sha256": EXPECTED_INITIAL_PARAMETER_SHA256,
        "shared_training_schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "interpretation": (
            "Validation loss and R_model are reported only after the training-only choice; "
            "they do not enter the selection metric."
        ),
    }
    write_json(OUTPUT, result)
    return result


def selection_decision(
    rows: list[dict[str, Any]],
) -> tuple[str, str | None, dict[str, Any], str]:
    """Return the predeclared choice without reading validation fields."""

    stable = [row for row in rows if row["stable"]]
    if not stable:
        raise ValueError("No stable learning-rate arm is available for selection.")
    best = min(stable, key=lambda row: (row["selection_metric"], row["peak_learning_rate"]))
    if math.isclose(float(best["peak_learning_rate"]), 1e-3, rel_tol=0.0, abs_tol=1e-15):
        status = "deferred_unbracketed"
        selected_condition_id = None
        next_action = "propose fresh A0 peak-LR 1.5e-3 full pass in a new numbered run"
    else:
        status = "selected"
        selected_condition_id = str(best["condition_id"])
        next_action = (
            "reuse the already-complete Run 019 TEAL frontier"
            if best["provenance"] == "run019_pinned_baseline"
            else "run the selected Run 021 checkpoint's ten-point post-hoc TEAL frontier"
        )
    return status, selected_condition_id, best, next_action


def _baseline_row(config: Mapping[str, Any]) -> dict[str, Any]:
    baseline = mapping(config, "baseline")
    attempt_dir = REPO_ROOT / str(baseline["source_run"]) / "artifacts" / "attempts" / str(
        baseline["attempt_id"]
    )
    expected_hashes = {
        "manifest.json": str(baseline["manifest_sha256"]),
        "metrics.json": str(baseline["metrics_sha256"]),
        "events.jsonl": str(baseline["events_sha256"]),
        "transfer_inventory.json": str(baseline["transfer_inventory_sha256"]),
    }
    for name, expected in expected_hashes.items():
        observed = _sha256_file(attempt_dir / name)
        if observed != expected:
            raise ValueError(f"Pinned Run 019 baseline file changed: {name}.")
    manifest = _json(attempt_dir / "manifest.json")
    metrics = _json(attempt_dir / "metrics.json")
    transfer = _json(attempt_dir / "transfer_inventory.json")
    verify_transfer_inventory(attempt_dir, transfer)
    if (
        manifest.get("status") != "completed"
        or manifest.get("condition", {}).get("id") != baseline["condition_id"]
        or int(manifest.get("completed_steps", -1)) != 712
        or int(manifest.get("input_tokens", -1)) != 1_493_172_224
        or manifest.get("initial_parameter_sha256") != EXPECTED_INITIAL_PARAMETER_SHA256
        or manifest.get("training_schedule_hash") != EXPECTED_SCHEDULE_SHA256
        or manifest.get("config_sha256") != baseline["config_sha256"]
        or manifest.get("run_code", {}).get("content_sha256") != baseline["run_code_sha256"]
        or metrics.get("checkpoints", {}).get("final", {}).get("content_sha256")
        != baseline["final_checkpoint_content_sha256"]
    ):
        raise ValueError("Pinned Run 019 baseline scientific identity changed.")
    logical = _json(attempt_dir / "diagnostics" / "logical_products.json")
    return _training_row(
        attempt_dir,
        condition_id=str(baseline["condition_id"]),
        peak=float(baseline["peak_learning_rate"]),
        minimum=float(baseline["minimum_learning_rate"]),
        validation_loss=float(logical["coverage"]["loss"]),
        r_model=float(logical["measured"]["R_model"]),
        provenance="run019_pinned_baseline",
    )


def _training_row(
    attempt_dir: Path,
    *,
    condition_id: str,
    peak: float,
    minimum: float,
    validation_loss: float,
    r_model: float,
    provenance: str,
) -> dict[str, Any]:
    events = [
        json.loads(line)
        for line in (attempt_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    training = [row for row in events if row.get("event") == "train"]
    if len(training) != 712 or [int(row["step"]) for row in training] != list(range(1, 713)):
        raise ValueError(f"Incomplete training events for {condition_id}.")
    finite_keys = (
        "task_loss",
        "learning_rate",
        "adamw_gradient_norm_pre_clip",
        "adamw_gradient_norm_post_clip",
    )
    stable = all(
        all(math.isfinite(float(row.get(key, math.nan))) for key in finite_keys)
        and row.get("gradient_overflow") is False
        and row.get("optimizer_step_skipped") is False
        for row in training
    )
    final_losses = [float(row["task_loss"]) for row in training[-FINAL_WINDOW:]]
    clipped = sum(bool(row.get("adamw_gradient_was_clipped")) for row in training)
    return {
        "condition_id": condition_id,
        "provenance": provenance,
        "peak_learning_rate": peak,
        "minimum_learning_rate": minimum,
        "stable": stable,
        "selection_metric": fmean(final_losses) if stable else None,
        "selection_window_start_step": 712 - FINAL_WINDOW + 1,
        "selection_window_end_step": 712,
        "final_task_loss": float(training[-1]["task_loss"]),
        "gradient_clipped_boundary_count": clipped,
        "validation_loss_not_used_for_selection": validation_loss,
        "R_model_not_used_for_selection": r_model,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value
