"""Run 015 verification with mandatory four-site pressure-capture evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from _reuse_run012 import load_run012_module
from run_config import (
    EXPECTED_PRESSURE_CAPTURE_NAMES_SHA256,
    EXPECTED_PRESSURE_CAPTURE_TENSOR_COUNT,
    RUN_DIR,
    write_json,
)


_BASE = load_run012_module("_run015_valid_run012_verification", "verification.py")
_ORIGINAL_REQUIRE_TRAIN_EVENTS = _BASE._require_train_events


def _require_train_events(
    attempt_dir: Path, condition: Mapping[str, Any]
) -> dict[str, Any]:
    result = _ORIGINAL_REQUIRE_TRAIN_EVENTS(attempt_dir, condition)
    with (attempt_dir / "events.jsonl").open("rb") as handle:
        events = [json.loads(line) for line in handle.read().splitlines() if line]
    train = [row for row in events if row.get("event") == "train"]
    for row in train:
        if (
            int(row.get("pressure_capture_tensor_count", -1))
            != EXPECTED_PRESSURE_CAPTURE_TENSOR_COUNT
            or row.get("pressure_capture_names_sha256")
            != EXPECTED_PRESSURE_CAPTURE_NAMES_SHA256
        ):
            raise ValueError(
                f"Four-site pressure-capture identity mismatch in {attempt_dir.name} "
                f"at step {row.get('step')}."
            )
    result.update(
        pressure_capture_tensor_count=EXPECTED_PRESSURE_CAPTURE_TENSOR_COUNT,
        pressure_capture_names_sha256=EXPECTED_PRESSURE_CAPTURE_NAMES_SHA256,
    )
    return result


_BASE._require_train_events = _require_train_events


def verify_attempt(condition_id: str) -> dict[str, Any]:
    return _BASE.verify_attempt(condition_id)


def verify_run() -> dict[str, Any]:
    result = _BASE.verify_run()
    result["question"] = "paper-scale matched A4-Z threshold versus corrected four-site A4-OL1"
    result["interpretation"]["comparison"] = (
        "Run 015 corrected four-site A4-OL1 versus Run 011 A4 at matched kappa"
    )
    result["correction"] = {
        "supersedes_declared_objective_of": (
            "runs/012-2026-08-30-pythia14m-full-pass-a4-ol1"
        ),
        "historical_realized_objective": "A4-Z gates plus OL1 pressure on h only",
        "corrected_pressure_sites": ["a", "m", "h", "z"],
        "pressure_capture_tensor_count_per_microbatch": (
            EXPECTED_PRESSURE_CAPTURE_TENSOR_COUNT
        ),
        "pressure_capture_names_sha256": EXPECTED_PRESSURE_CAPTURE_NAMES_SHA256,
    }
    write_json(RUN_DIR / "artifacts" / "verification.json", result)
    return result
