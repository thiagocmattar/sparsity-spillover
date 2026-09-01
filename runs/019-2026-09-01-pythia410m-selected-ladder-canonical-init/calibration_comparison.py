#!/usr/bin/env python
"""Compare retrieved GPU calibrations without selecting hardware automatically."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from run_config import EXPECTED_INITIAL_PARAMETER_SHA256, write_json


def compare(paths: list[Path], output: Path) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("At least two candidate calibration artifacts are required.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison output: {output}")
    artifacts = [_json(path) for path in paths]
    if any(row.get("status") != "passed" for row in artifacts):
        raise ValueError("Only passed calibration artifacts may enter the comparison.")
    initial_hashes = {
        sample["initial_parameter_sha256"]
        for artifact in artifacts
        for sample in artifact["samples"]
    }
    if initial_hashes != {EXPECTED_INITIAL_PARAMETER_SHA256}:
        raise ValueError("Candidate calibrations do not share the canonical initialization.")
    schedule_hashes = {artifact["cache"]["schedule_sha256"] for artifact in artifacts}
    code_hashes = {artifact["run_code"]["content_sha256"] for artifact in artifacts}
    if len(schedule_hashes) != 1 or len(code_hashes) != 1:
        raise ValueError("Candidate calibrations differ in schedule or run-code identity.")
    candidates = []
    for path, artifact in zip(paths, artifacts, strict=True):
        projection = artifact["projection"]
        device = artifact["device"]
        peak = max(int(sample["peak_memory_reserved_bytes"]) for sample in artifact["samples"])
        candidates.append(
            {
                "path": path.as_posix(),
                "gpu_type_id": artifact["price_snapshot"]["gpu_type_id"],
                "cloud_type": artifact["price_snapshot"]["cloud_type"],
                "hourly_price_usd": float(artifact["price_snapshot"]["hourly_price_usd"]),
                "price_captured_at": artifact["price_snapshot"]["captured_at"],
                "device_name": device["name"],
                "visible_memory_gib": int(device["total_memory_bytes"]) / 1024**3,
                "peak_reserved_gib": peak / 1024**3,
                "remaining_headroom_fraction": 1.0 - peak / int(device["total_memory_bytes"]),
                "projected_gpu_hours": float(projection["projected_gpu_hours"]),
                "projected_total_gpu_cost_usd": float(
                    projection["projected_total_gpu_cost_usd"]
                ),
                "projected_twelve_way_makespan_hours": float(
                    projection["projected_twelve_way_makespan_seconds"]
                )
                / 3600.0,
                "twelve_gpu_hourly_burn_usd": float(
                    projection["twelve_gpu_hourly_burn_usd"]
                ),
                "condition_projections": projection["conditions"],
            }
        )
    for candidate in candidates:
        candidate["nondominated_cost_makespan"] = not any(
            other is not candidate
            and other["projected_total_gpu_cost_usd"]
            <= candidate["projected_total_gpu_cost_usd"]
            and other["projected_twelve_way_makespan_hours"]
            <= candidate["projected_twelve_way_makespan_hours"]
            and (
                other["projected_total_gpu_cost_usd"]
                < candidate["projected_total_gpu_cost_usd"]
                or other["projected_twelve_way_makespan_hours"]
                < candidate["projected_twelve_way_makespan_hours"]
            )
            for other in candidates
        )
    cheapest = min(candidates, key=lambda row: row["projected_total_gpu_cost_usd"])
    fastest = min(candidates, key=lambda row: row["projected_twelve_way_makespan_hours"])
    result = {
        "schema_version": 1,
        "kind": "run019_gpu_cost_etc_comparison",
        "status": "ready_for_human_selection",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "initial_parameter_sha256": EXPECTED_INITIAL_PARAMETER_SHA256,
        "schedule_sha256": next(iter(schedule_hashes)),
        "run_code_sha256": next(iter(code_hashes)),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "cheapest_candidate": {
            "gpu_type_id": cheapest["gpu_type_id"],
            "cloud_type": cheapest["cloud_type"],
        },
        "fastest_candidate": {
            "gpu_type_id": fastest["gpu_type_id"],
            "cloud_type": fastest["cloud_type"],
        },
        "selection": None,
        "selection_rule": (
            "Human chooses one same-SKU cohort after considering nondominated total cost, "
            "makespan, live capacity, and required memory headroom. Refresh prices before launch."
        ),
        "excluded_overhead": (
            "Provisioning, package installation, input upload, and output download remain external "
            "to the measured workload and must be added to the launch envelope."
        ),
    }
    write_json(output, result)
    return result


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return dict(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibrations", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.calibrations, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
