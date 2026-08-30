#!/usr/bin/env python
"""Exact target-GPU A4-Z endpoint preflight; creates no scientific attempt."""

from __future__ import annotations

import json
import platform
import sys
from time import perf_counter

from run_config import (
    EXPECTED_INITIAL_PARAMETER_SHA256,
    EXPECTED_SCHEDULE_SHA256,
    RUN_DIR,
    build_schedule,
    cache_identity,
    load_config,
    load_verified_caches,
    run_code_identity,
    write_json,
)
from smoke import ENDPOINT_CONDITION_IDS, run_smoke


def main() -> None:
    import numpy as np
    import torch
    import transformers

    config = load_config()
    runtime = config["runtime"]
    realized = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": torch.__version__.split("+", 1)[0],
        "transformers": transformers.__version__,
        "cuda_runtime": str(torch.version.cuda),
    }
    if realized != runtime:
        raise RuntimeError(f"Pinned runtime mismatch: realized={realized}, expected={runtime}")

    cache_started = perf_counter()
    train, validation, train_metadata, validation_metadata, verification_seconds = (
        load_verified_caches(config, np=np)
    )
    _, schedule_hash, schedule = build_schedule(config, train_metadata, np=np)
    cache_result = {
        "train": cache_identity(train_metadata),
        "validation": cache_identity(validation_metadata),
        "verification_seconds": verification_seconds,
        "wall_seconds": perf_counter() - cache_started,
        "schedule": schedule,
        "schedule_sha256": schedule_hash,
    }
    del train, validation
    if schedule_hash != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError("Preflight schedule does not match Run 004.")

    device_properties = torch.cuda.get_device_properties(0)
    smoke = run_smoke(
        batch_sizes=(int(config["training"]["micro_batch_size"]),),
        sequence_length=int(config["data"]["sequence_length"]),
        boundaries=5,
        accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        target_gpu=torch.cuda.get_device_name(0),
        target_memory_bytes=int(device_properties.total_memory),
    )
    samples = smoke.get("samples", [])
    sampled_ids = {row.get("condition_id") for row in samples}
    initial_hash_ok = (
        sampled_ids == set(ENDPOINT_CONDITION_IDS)
        and all(
            row.get("initial_parameter_sha256") == EXPECTED_INITIAL_PARAMETER_SHA256
            for row in samples
        )
    )
    five_boundaries = (
        sampled_ids == set(ENDPOINT_CONDITION_IDS)
        and all(len(row.get("boundary_seconds", [])) == 5 for row in samples)
    )
    exact_ok = bool(smoke.get("exact_target", {}).get("all_conditions_fit"))
    result = {
        "schema_version": 1,
        "kind": "non_evidence_remote_preflight",
        "status": "passed" if exact_ok and initial_hash_ok and five_boundaries else "failed",
        "platform": platform.platform(),
        "runtime": realized,
        "device": smoke.get("device"),
        "cache": cache_result,
        "run_code": run_code_identity(),
        "smoke": smoke,
        "checks": {
            "exact_target_fit": exact_ok,
            "initial_parameter_matches_run004": initial_hash_ok,
            "five_complete_boundaries_at_both_kappa_endpoints": five_boundaries,
        },
    }
    output = RUN_DIR / "prelaunch" / "remote-preflight.json"
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
