"""Corrected four-site adapter around the matched Run 012 smoke probe."""

from __future__ import annotations

from typing import Any

from _reuse_run012 import load_run012_module
from run015_capture import FourSitePressureCapture
from run_config import (
    EXPECTED_PRESSURE_CAPTURE_NAMES_SHA256,
    EXPECTED_PRESSURE_CAPTURE_TENSOR_COUNT,
)


_BASE = load_run012_module("_run015_valid_run012_smoke", "smoke.py")
_BASE._BASE.ActivationCapture = FourSitePressureCapture
_BASE._BASE.__dict__["ActivationCapture"] = FourSitePressureCapture
_ORIGINAL_EXACT_TARGET = _BASE._exact_target


def _exact_target(
    samples: list[dict[str, Any]],
    target: dict[str, Any],
    sequence_length: int,
    accumulation_steps: int,
) -> dict[str, Any]:
    result = _ORIGINAL_EXACT_TARGET(samples, target, sequence_length, accumulation_steps)
    if result.get("status") != "sampled":
        return result
    by_id = {str(row.get("condition_id")): row for row in samples}
    for condition_id, condition in result["conditions"].items():
        boundary = by_id.get(condition_id, {}).get("last_boundary", {})
        capture_ok = (
            int(boundary.get("pressure_capture_tensor_count", -1))
            == EXPECTED_PRESSURE_CAPTURE_TENSOR_COUNT
            and boundary.get("pressure_capture_names_sha256")
            == EXPECTED_PRESSURE_CAPTURE_NAMES_SHA256
        )
        condition["pressure_capture_identity_ok"] = capture_ok
        condition["finite_nonoverflowing_boundaries"] = bool(
            condition["finite_nonoverflowing_boundaries"] and capture_ok
        )
        condition["fits_with_10pct_headroom"] = bool(
            condition["fits_with_10pct_headroom"] and capture_ok
        )
    result["all_conditions_fit"] = all(
        row["fits_with_10pct_headroom"] for row in result["conditions"].values()
    )
    return result


_BASE._exact_target = _exact_target
ENDPOINT_CONDITION_IDS = _BASE.ENDPOINT_CONDITION_IDS
run_smoke = _BASE.run_smoke
