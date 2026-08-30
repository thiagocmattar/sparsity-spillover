"""Append-only Run 005 verification for the detached null-Git limitation.

This does not modify immutable attempts. It first runs the original terminal
verifier and requires it to reach only the known Git-identity failure, then
records a valid-with-provenance-limitation summary using the launch sidecar.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_config import RUN_DIR, condition_specs, load_config, write_json  # noqa: E402
from verification import verify_cohort  # noqa: E402


EXPECTED_FAILURE = "Git commit identity is missing or mixed."


def verify_with_provenance_limitation() -> Path:
    try:
        verify_cohort()
    except RuntimeError as error:
        if str(error) != EXPECTED_FAILURE:
            raise
    else:
        raise RuntimeError("Original verifier unexpectedly accepted detached Git identity.")

    config = load_config()
    conditions = condition_specs(config)
    launch_plan = _json(RUN_DIR / "prelaunch" / "launch-plan.json")
    sidecar = _json(RUN_DIR / "artifacts" / "launch-provenance.json")
    if sidecar.get("git_commit") != launch_plan["identity"]["git_commit"]:
        raise RuntimeError("Launch-sidecar Git commit differs from the approved packet.")
    for key in (
        "config_sha256",
        "run_code_content_sha256",
        "training_schedule_sha256",
    ):
        if sidecar.get(key) != launch_plan["identity"][key]:
            raise RuntimeError(f"Launch-sidecar {key} differs from the approved packet.")

    attempts_root = RUN_DIR / "artifacts" / "attempts"
    rows = []
    initial_hashes = set()
    schedule_hashes = set()
    code_hashes = set()
    checkpoint_bytes = 0
    total_condition_seconds = 0.0
    for condition in conditions:
        matches = sorted(
            path
            for path in attempts_root.glob(f"{int(condition['order']):03d}-*")
            if path.is_dir()
        )
        if len(matches) != 1:
            raise RuntimeError(f"Expected one attempt for condition {condition['id']}.")
        attempt = matches[0]
        manifest = _json(attempt / "manifest.json")
        metrics = _json(attempt / "metrics.json")
        if manifest.get("code") != {"git_commit": None, "git_dirty": None}:
            raise RuntimeError(f"Unexpected detached Git state for {attempt.name}.")
        if manifest.get("status") != "completed" or int(manifest.get("completed_steps", -1)) != 581:
            raise RuntimeError(f"Scientific attempt is incomplete: {attempt.name}.")
        events = _jsonl(attempt / "events.jsonl")
        training_events = [event for event in events if event.get("event") == "train"]
        if len(training_events) != 581:
            raise RuntimeError(f"Training event count mismatch for {attempt.name}.")
        activation = _json(attempt / "diagnostics" / "activation_statistics.json")
        pooled = {row["name"]: row for row in activation["pooled_by_site"]}
        logical = _json(attempt / "diagnostics" / "logical_products.json")
        measured = logical["measured"]
        maximum = logical["architecture_maximum"]
        pressure_events = [event for event in training_events if "trust_scale" in event]
        trust_scales = [float(event["trust_scale"]) for event in pressure_events]
        final_ratios = [
            float(event["pressure_to_task_ratio_final"]) for event in pressure_events
        ]
        conflict_rate = (
            sum(bool(event["gradient_conflict"]) for event in pressure_events)
            / len(pressure_events)
            if pressure_events
            else None
        )
        projection_rate = (
            sum(bool(event["projection_applied"]) for event in pressure_events)
            / len(pressure_events)
            if pressure_events
            else None
        )
        row = {
            "attempt_id": attempt.name,
            "condition": condition,
            "completed_steps": int(manifest["completed_steps"]),
            "input_tokens": int(manifest["input_tokens"]),
            "final_train_loss": float(metrics["training"]["task_loss_final"]),
            "final_validation_loss": float(metrics["validation"]["final"]["loss"]),
            "median_step_seconds": float(metrics["training"]["median_step_seconds"]),
            "median_tokens_per_second": float(
                metrics["training"]["median_tokens_per_second"]
            ),
            "condition_seconds": float(metrics["timing"]["total_seconds"]),
            "h_exact_zero_fraction": float(pooled["h"]["exact_zero_fraction"]),
            "h_near_zero_fraction_epsilon_0p001": float(
                pooled["h"]["threshold_fractions"]["0.001"]
            ),
            "R_block": float(measured["R_block"]),
            "R_model": float(measured["R_model"]),
            "R_model_max": float(maximum["R_model_max_fraction"]),
            "gradient_conflict_rate": conflict_rate,
            "projection_rate": projection_rate,
            "trust_scale_minimum": min(trust_scales) if trust_scales else None,
            "trust_budget_saturation_rate": (
                sum(value < 1.0 for value in trust_scales) / len(trust_scales)
                if trust_scales
                else None
            ),
            "final_correction_ratio_maximum": max(final_ratios) if final_ratios else None,
            "checkpoint_bytes": int(metrics["checkpoint"]["bytes"]),
            "checkpoint_content_sha256": metrics["checkpoint"]["content_sha256"],
        }
        for key in (
            "final_train_loss",
            "final_validation_loss",
            "median_step_seconds",
            "median_tokens_per_second",
            "condition_seconds",
            "h_exact_zero_fraction",
            "h_near_zero_fraction_epsilon_0p001",
            "R_block",
            "R_model",
            "R_model_max",
        ):
            if not math.isfinite(float(row[key])):
                raise RuntimeError(f"Non-finite summary field {key} for {attempt.name}.")
        rows.append(row)
        checkpoint_bytes += row["checkpoint_bytes"]
        total_condition_seconds += row["condition_seconds"]
        initial_hashes.add(manifest["initial_parameter_sha256"])
        schedule_hashes.add(manifest["training_schedule_hash"])
        code_hashes.add(manifest["run_code"]["content_sha256"])

    expected_identity = launch_plan["identity"]
    if initial_hashes != {expected_identity["initial_parameter_sha256_from_calibration"]}:
        raise RuntimeError("Initial parameter identities do not match calibration.")
    if schedule_hashes != {expected_identity["training_schedule_sha256"]}:
        raise RuntimeError("Training schedules do not match the launch packet.")
    if code_hashes != {expected_identity["run_code_content_sha256"]}:
        raise RuntimeError("Run-code identities do not match the launch packet.")

    driver = _json(RUN_DIR / "artifacts" / "driver.json")
    started = datetime.fromisoformat(driver["started_at"])
    finished = datetime.fromisoformat(driver["finished_at"])
    wall_seconds = (finished - started).total_seconds()
    if driver.get("failure", {}).get("message") != EXPECTED_FAILURE:
        raise RuntimeError("Driver did not stop only at the known provenance limitation.")
    if wall_seconds > float(launch_plan["calibration"]["approved_etc_ceiling_seconds"]):
        raise RuntimeError("Cohort exceeded the approved local wall-time envelope.")

    output = RUN_DIR / "artifacts" / "verification.json"
    result = {
        "schema_version": 1,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "verifier": {
            "path": Path(__file__).name,
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "original_verifier_expected_failure": EXPECTED_FAILURE,
        },
        "status": "verified",
        "evidence_label": "valid_with_provenance_limitation",
        "provenance_limitation": (
            "All detached attempt manifests have null Git commit/dirty fields. "
            "Exact config, run-code content, schedule, initialization, data, "
            "environment, checkpoint, diagnostics, and launch-sidecar identities pass."
        ),
        "launch_provenance_sidecar": "artifacts/launch-provenance.json",
        "launch_git_commit": sidecar["git_commit"],
        "launch_git_dirty": sidecar["git_dirty"],
        "condition_count": len(rows),
        "total_optimizer_steps": sum(row["completed_steps"] for row in rows),
        "total_training_input_tokens": sum(row["input_tokens"] for row in rows),
        "complete_validation_passes": 3 * len(rows),
        "checkpoint_total_bytes": checkpoint_bytes,
        "cohort_wall_seconds": wall_seconds,
        "summed_condition_seconds": total_condition_seconds,
        "approved_etc_ceiling_seconds": launch_plan["calibration"][
            "approved_etc_ceiling_seconds"
        ],
        "initial_parameter_sha256": next(iter(initial_hashes)),
        "training_schedule_sha256": next(iter(schedule_hashes)),
        "run_code_content_sha256": next(iter(code_hashes)),
        "conditions": rows,
        "interpretation": {
            "logical_products": "opportunities, not removed FLOPs or speedup",
            "scope": "one seed, one scale, sub-hour local horizon",
        },
    }
    write_json(output, result)
    return output


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    print(verify_with_provenance_limitation())
