#!/usr/bin/env python
"""Exact target-GPU preflight; creates no scientific attempt."""

from __future__ import annotations

import json
import platform
import sys
from time import perf_counter

from run_config import RUN_DIR, cache_identity, load_config, load_verified_caches, run_code_identity, write_json
from smoke import run_smoke


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
    train, validation, train_metadata, validation_metadata, verification_seconds = load_verified_caches(config, np=np)
    cache_result = {
        "train": cache_identity(train_metadata),
        "validation": cache_identity(validation_metadata),
        "verification_seconds": verification_seconds,
        "wall_seconds": perf_counter() - cache_started,
    }
    del train, validation
    device_properties = torch.cuda.get_device_properties(0)
    smoke = run_smoke(
        batch_sizes=(int(config["training"]["micro_batch_size"]),),
        sequence_length=int(config["data"]["sequence_length"]),
        boundaries=2,
        accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        target_gpu=torch.cuda.get_device_name(0),
        target_memory_bytes=int(device_properties.total_memory),
    )
    result = {
        "schema_version": 1,
        "kind": "non_evidence_remote_preflight",
        "status": "passed" if smoke.get("exact_target", {}).get("all_conditions_fit") else "failed",
        "platform": platform.platform(),
        "runtime": realized,
        "device": smoke.get("device"),
        "cache": cache_result,
        "run_code": run_code_identity(),
        "smoke": smoke,
    }
    output = RUN_DIR / "prelaunch" / "remote-preflight.json"
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
