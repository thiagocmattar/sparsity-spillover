#!/usr/bin/env python
"""Verify two independent strict loads of the canonical 410M initialization."""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import random
from typing import Any

from initialization_artifact import load_pinned_initialization, sha256_file
from model_factory import build_pinned_run019_model
from run_config import (
    EXPECTED_INITIAL_PARAMETER_SHA256,
    RUN_DIR,
    condition_specs,
    load_config,
    mapping,
    parameter_sha256,
    resolved_condition_config,
    write_json,
)


def verify_initialization_identity(output: Path | None = None) -> dict[str, Any]:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    config = load_config()
    artifact = mapping(config, "initialization_artifact")
    a0 = next(row for row in condition_specs(config) if row["id"] == "a0-gelu")
    resolved = resolved_condition_config(config, a0)
    draws = []
    for _ in range(2):
        model = build_pinned_run019_model(
            dict(mapping(resolved, "model")),
            device=torch.device("cpu"),
            torch=torch,
            auto_model=AutoModelForCausalLM,
        )
        recipe = load_pinned_initialization(model, torch=torch)
        realized = parameter_sha256(model)
        draws.append(
            {
                "parameter_sha256": realized,
                "python_rng_probe": random.getrandbits(64),
                "numpy_rng_probe": np.random.bytes(16).hex(),
                "torch_cpu_rng_probe": torch.randint(0, 2**31, (8,), dtype=torch.int64).tolist(),
                "recipe": recipe,
            }
        )
        del model
    if {row["parameter_sha256"] for row in draws} != {EXPECTED_INITIAL_PARAMETER_SHA256}:
        raise RuntimeError("Independent strict loads did not reproduce the pinned parameter hash.")
    probes = {
        (row["python_rng_probe"], row["numpy_rng_probe"], tuple(row["torch_cpu_rng_probe"]))
        for row in draws
    }
    if len(probes) != 1:
        raise RuntimeError("Independent strict loads did not restore the same post-init RNG state.")
    model_path = RUN_DIR / str(artifact["model_path"])
    rng_path = RUN_DIR / str(artifact["rng_path"])
    result = {
        "schema_version": 1,
        "kind": "canonical_initialization_identity_check",
        "status": "verified",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": __import__("platform").python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "model_artifact": {
            "path": model_path.relative_to(RUN_DIR).as_posix(),
            "bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
        },
        "rng_artifact": {
            "path": rng_path.relative_to(RUN_DIR).as_posix(),
            "bytes": rng_path.stat().st_size,
            "sha256": sha256_file(rng_path),
        },
        "parameter_sha256": EXPECTED_INITIAL_PARAMETER_SHA256,
        "independent_strict_loads": len(draws),
        "post_initialization_rng_probe": draws[0],
    }
    if output is not None:
        write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_initialization_identity(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
