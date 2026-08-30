"""Terminal Run 009 verification, including exact Run 004 comparator identities."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from _reuse_run004 import load_run004_module
from run_config import (
    EXPECTED_INITIAL_PARAMETER_SHA256,
    EXPECTED_RUN004_CODE_SHA256,
    EXPECTED_SCHEDULE_SHA256,
    RUN004_VERIFICATION,
    RUN_DIR,
    condition_specs,
    load_config,
    write_json,
)


OUTPUT = RUN_DIR / "artifacts" / "verification.json"
_BASE = load_run004_module("_run009_frozen_run004_verification", "verification.py")


def verify_run() -> dict[str, Any]:
    config = load_config()
    temporary = RUN_DIR / "artifacts" / ".run004-generic-verification.json"
    _BASE.OUTPUT = temporary
    try:
        result = _BASE.verify_run()
        _require_expected_run009_identities(result)
        source = _load_run004_verification()
        source_by_id = {row["condition"]["id"]: row for row in source["conditions"]}
        comparisons = []
        attempts_root = RUN_DIR / "artifacts" / "attempts"
        for condition in condition_specs(config):
            attempt_dir = _BASE._one_attempt(attempts_root, int(condition["order"]))
            _require_ol1_events(attempt_dir, condition)
            l1_id = condition["id"].replace("relu-ol1", "relu-l1n")
            comparisons.append({
                "ol1_condition_id": condition["id"],
                "matched_l1_condition_id": l1_id,
                "matched_l1_attempt_id": source_by_id[l1_id]["attempt_id"],
                "matched_l1_final_validation_loss": float(source_by_id[l1_id]["final_validation_loss"]),
            })
        result["comparison"] = {
            "source_run": "runs/004-2026-08-29-pythia14m-full-pass-l1n",
            "source_verification": "runs/004-2026-08-29-pythia14m-full-pass-l1n/artifacts/verification.json",
            "reused_controls": [
                {
                    "condition_id": condition_id,
                    "attempt_id": source_by_id[condition_id]["attempt_id"],
                    "final_validation_loss": float(source_by_id[condition_id]["final_validation_loss"]),
                }
                for condition_id in ("gelu-control", "relu-control")
            ],
            "matched_l1_conditions": comparisons,
        }
        result["interpretation"].update({
            "comparison": "Run 009 OL1 versus matched Run 004 naive L1; Run 004 controls reused",
            "hardware": "pairwise A100 type is requested but numerical execution is not bitwise across Pods",
        })
        write_json(OUTPUT, result)
        return result
    finally:
        temporary.unlink(missing_ok=True)


def _require_expected_run009_identities(result: dict[str, Any]) -> None:
    if result.get("condition_count") != 4:
        raise ValueError("Run 009 must contain exactly four new OL1 conditions.")
    if result.get("initial_parameter_sha256") != EXPECTED_INITIAL_PARAMETER_SHA256:
        raise ValueError("Run 009 initialization does not match Run 004.")
    if result.get("training_schedule_sha256") != EXPECTED_SCHEDULE_SHA256:
        raise ValueError("Run 009 schedule does not match Run 004.")


def _load_run004_verification() -> dict[str, Any]:
    source = _json(RUN004_VERIFICATION)
    if source.get("status") != "verified" or source.get("evidence_label") != "valid":
        raise ValueError("Run 004 comparator verification is not valid and terminal.")
    expected = {
        "initial_parameter_sha256": EXPECTED_INITIAL_PARAMETER_SHA256,
        "training_schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "run_code_sha256": EXPECTED_RUN004_CODE_SHA256,
        "condition_count": 6,
    }
    if any(source.get(key) != value for key, value in expected.items()):
        raise ValueError("Run 004 comparator identity changed.")
    ids = [row.get("condition", {}).get("id") for row in source.get("conditions", [])]
    if ids != [
        "gelu-control", "relu-control", "relu-l1n-0p05", "relu-l1n-0p1",
        "relu-l1n-0p5", "relu-l1n-1",
    ]:
        raise ValueError("Run 004 comparator condition order changed.")
    return source


def _require_ol1_events(attempt_dir: Path, condition: dict[str, Any]) -> None:
    events_path = attempt_dir / "events.jsonl"
    with events_path.open("rb") as handle:
        events = [json.loads(line) for line in handle.read().splitlines() if line]
    train = [row for row in events if row.get("event") == "train"]
    if [int(row.get("step", -1)) for row in train] != list(range(1, 713)):
        raise ValueError(f"Incomplete OL1 event history for {attempt_dir.name}.")
    finite_fields = (
        "task_loss", "pressure_loss", "task_gradient_norm", "pressure_gradient_norm",
        "task_direction_norm", "pressure_direction_norm_raw",
        "task_pressure_dot_before", "task_pressure_dot_after",
        "pressure_to_task_ratio_raw", "trust_scale", "pressure_to_task_ratio_final",
    )
    for row in train:
        if row.get("condition_id") != condition["id"]:
            raise ValueError(f"Condition mismatch in {attempt_dir.name} event history.")
        if row.get("fp16_overflow_policy") != "skip_entire_boundary":
            raise ValueError(f"FP16/OL1 atomicity missing in {attempt_dir.name}.")
        if row.get("ol1_correction_applied") is not True:
            raise ValueError(f"OL1 correction missing in {attempt_dir.name}.")
        if any(not math.isfinite(float(row.get(key, math.nan))) for key in finite_fields):
            raise ValueError(f"Non-finite OL1 metric in {attempt_dir.name}.")
        if float(row["pressure_to_task_ratio_final"]) > 1.0 + 1e-9:
            raise ValueError(f"OL1 trust budget exceeded in {attempt_dir.name}.")
        if float(row["adamw_gradient_norm_post_clip"]) > 1.0 + 1e-5:
            raise ValueError(f"Task gradient clip exceeded in {attempt_dir.name}.")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value
