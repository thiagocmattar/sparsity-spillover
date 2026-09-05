#!/usr/bin/env python3
"""Measure backend/device sensitivity of the final Run-024 source-loss gate."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np


RUN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUN_DIR))

from run_config import checkpoint_path, condition_by_id, repo_path, write_json  # noqa: E402


def load_benchmark():
    path = RUN_DIR / "03_benchmark.py"
    specification = importlib.util.spec_from_file_location("run024_source_backend_diagnostic", path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    benchmark = load_benchmark()
    base = benchmark._impl
    config = benchmark.load_resolved_config()
    condition = condition_by_id(config, "a7-ol1-kappa-0p5")
    tokens = np.memmap(repo_path(config["validation"]["tokens"]), dtype=np.int32, mode="r")
    model = base.configure_model(
        base.load_checkpoint_pythia(
            base.AutoModelForCausalLM,
            checkpoint_path(config, condition),
            torch=base.torch,
        ).to(device="cuda:0", dtype=base.torch.float32)
    )

    records = []
    order = ("sdpa", "eager", "sdpa", "eager", "sdpa")
    progress = args.output.resolve().with_name("diagnostic-progress.json")
    for index, backend in enumerate(order, start=1):
        model.set_attn_implementation(backend)
        record = base.evaluate(
            model,
            tokens,
            config=config,
            condition=condition,
            progress_path=progress,
            stage=f"diagnostic_{backend}_{index}",
            autocast_dtype=base.torch.float16,
            batch_size=config["validation"]["source_reproduction_batch_size"],
        )
        records.append(
            {
                "order": index,
                "requested_backend": backend,
                "resolved_backend": model.config._attn_implementation,
                **record,
            }
        )

    canonical_eager = condition["canonical_eager_validation_loss"]
    archived_sdpa = condition["archived_sdpa_validation_loss"]
    for record in records:
        record["absolute_difference_from_canonical_eager"] = abs(record["loss"] - canonical_eager)
        record["absolute_difference_from_archived_a100_sdpa"] = abs(record["loss"] - archived_sdpa)

    output = {
        "schema_version": 1,
        "condition_id": condition["id"],
        "checkpoint": str(checkpoint_path(config, condition)),
        "gpu": base.gpu_snapshot(),
        "archived": {
            "canonical_eager_loss": canonical_eager,
            "a100_sdpa_loss": archived_sdpa,
            "source_loss_absolute_tolerance": config["validation"]["source_loss_absolute_tolerance"],
        },
        "order": list(order),
        "records": records,
        "interpretation": "Source-only diagnostic; no sparse-kernel or timing result.",
    }
    write_json(args.output.resolve(), output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

