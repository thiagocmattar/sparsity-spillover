"""Estimate end-to-end duration from representative measured phase timings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any


def estimate(
    calibration: dict[str, Any],
    *,
    steps: int,
    validation_passes: int,
    gpu_count: int = 1,
    hourly_gpu_price: float | None = None,
    maximum_hours: float | None = None,
    storage_cost: float = 0.0,
) -> dict[str, Any]:
    step_samples = _samples(calibration, "optimizer_step_seconds")
    validation_samples = _samples(calibration, "full_validation_seconds")
    fixed = sum(
        float(calibration.get(key, 0.0))
        for key in (
            "provision_seconds",
            "setup_seconds",
            "diagnostics_seconds",
            "checkpoint_seconds",
            "transfer_seconds",
        )
    )
    median_step = statistics.median(step_samples)
    upper_step = _percentile(step_samples, 0.90)
    median_validation = statistics.median(validation_samples)
    upper_validation = _percentile(validation_samples, 0.90)
    median_seconds = fixed + steps * median_step + validation_passes * median_validation
    upper_seconds = fixed + steps * upper_step + validation_passes * upper_validation
    result: dict[str, Any] = {
        "steps": steps,
        "validation_passes": validation_passes,
        "step_samples": len(step_samples),
        "validation_samples": len(validation_samples),
        "fixed_seconds": fixed,
        "median_step_seconds": median_step,
        "p90_step_seconds": upper_step,
        "median_validation_seconds": median_validation,
        "p90_validation_seconds": upper_validation,
        "median_etc_seconds": median_seconds,
        "p90_etc_seconds": upper_seconds,
        "median_etc_hours": median_seconds / 3600.0,
        "p90_etc_hours": upper_seconds / 3600.0,
    }
    if hourly_gpu_price is not None:
        result["projected_compute_cost_median"] = median_seconds / 3600.0 * gpu_count * hourly_gpu_price
        result["projected_compute_cost_p90"] = upper_seconds / 3600.0 * gpu_count * hourly_gpu_price
    if maximum_hours is not None:
        if hourly_gpu_price is None:
            raise ValueError("Maximum-hours cost requires hourly_gpu_price.")
        result["maximum_hours"] = maximum_hours
        result["maximum_total_cost"] = maximum_hours * gpu_count * hourly_gpu_price + storage_cost
    return result


def _samples(calibration: dict[str, Any], key: str) -> list[float]:
    values = calibration.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"Calibration requires a non-empty {key} list.")
    samples = [float(value) for value in values]
    if any(not math.isfinite(value) or value <= 0.0 for value in samples):
        raise ValueError(f"Every {key} sample must be positive and finite.")
    return samples


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--validation-passes", type=int, required=True)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--hourly-gpu-price", type=float)
    parser.add_argument("--maximum-hours", type=float)
    parser.add_argument("--storage-cost", type=float, default=0.0)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    result = estimate(
        calibration,
        steps=args.steps,
        validation_passes=args.validation_passes,
        gpu_count=args.gpu_count,
        hourly_gpu_price=args.hourly_gpu_price,
        maximum_hours=args.maximum_hours,
        storage_cost=args.storage_cost,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

