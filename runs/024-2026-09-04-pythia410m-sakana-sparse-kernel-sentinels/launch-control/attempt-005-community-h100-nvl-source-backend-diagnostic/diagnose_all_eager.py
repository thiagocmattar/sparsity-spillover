#!/usr/bin/env python3
"""Preflight the canonical eager source-loss gate for all six sentinels."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np


RUN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUN_DIR))

from run_config import checkpoint_path, conditions_for_phase, repo_path, write_json  # noqa: E402


def load_benchmark():
    path = RUN_DIR / "03_benchmark.py"
    specification = importlib.util.spec_from_file_location("run024_all_eager_diagnostic", path)
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
    tokens = np.memmap(repo_path(config["validation"]["tokens"]), dtype=np.int32, mode="r")
    tolerance = config["validation"]["source_loss_absolute_tolerance"]
    progress = args.output.resolve().with_name("all-eager-progress.json")
    records = []

    for condition in conditions_for_phase(config, "sentinel"):
        model = base.configure_model(
            base.load_checkpoint_pythia(
                base.AutoModelForCausalLM,
                checkpoint_path(config, condition),
                torch=base.torch,
            ).to(device="cuda:0", dtype=base.torch.float32)
        )
        model.set_attn_implementation("eager")
        record = base.evaluate(
            model,
            tokens,
            config=config,
            condition=condition,
            progress_path=progress,
            stage=f"diagnostic_eager_{condition['id']}",
            autocast_dtype=base.torch.float16,
            batch_size=config["validation"]["source_reproduction_batch_size"],
        )
        reference = condition["canonical_eager_validation_loss"]
        difference = abs(record["loss"] - reference)
        records.append(
            {
                "condition_id": condition["id"],
                "requested_backend": "eager",
                "resolved_backend": model.config._attn_implementation,
                "loss": record["loss"],
                "canonical_eager_loss": reference,
                "absolute_difference": difference,
                "passed": difference <= tolerance,
                "coverage": {
                    "sequences": record["sequences"],
                    "input_tokens": record["input_tokens"],
                    "excluded_tail_tokens": record["excluded_tail_tokens"],
                    "complete_block_coverage": record["complete_block_coverage"],
                },
            }
        )
        del model
        base.torch.cuda.empty_cache()

    output = {
        "schema_version": 1,
        "gpu": base.gpu_snapshot(),
        "source_loss_absolute_tolerance": tolerance,
        "conditions": records,
        "passed": all(record["passed"] for record in records),
        "interpretation": "Source-only gate preflight; no sparse-kernel or timing result.",
    }
    write_json(args.output.resolve(), output)
    if not output["passed"]:
        raise RuntimeError("At least one canonical eager source-loss gate failed.")
    print(args.output.resolve())


if __name__ == "__main__":
    main()

