#!/usr/bin/env python3
"""Recompute canonical six-operation logical counts on the timing GPU."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sparsity_research.evaluation import evaluate_complete_blocks
from sparsity_research.logical_capture import LogicalProductAccumulator, capture_logical_products


RUN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUN_DIR))
SPEC = importlib.util.spec_from_file_location("run023_benchmark_logical_forensics", RUN_DIR / "03_benchmark.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load the Run-023 benchmark module.")
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = BENCHMARK.load_config()
    condition = next(row for row in config["conditions"] if row["id"] == "a7-ol1-kappa-0")
    model = BENCHMARK.configure_model(
        BENCHMARK.load_checkpoint_pythia(
            BENCHMARK.AutoModelForCausalLM,
            BENCHMARK.checkpoint_path(config, condition),
            torch=torch,
        ).to(device=torch.device("cuda:0"), dtype=torch.float32)
    )
    tokens = np.memmap(
        BENCHMARK.repo_path(config["validation"]["tokens"]),
        dtype=np.int32,
        mode="r",
    )
    accumulator = LogicalProductAccumulator()
    with capture_logical_products(model, accumulator=accumulator, torch=torch):
        coverage = evaluate_complete_blocks(
            model=model,
            tokens=tokens,
            block_size=config["model"]["architecture"]["sequence_length"],
            batch_size=config["validation"]["logical_opportunity_batch_size"],
            device=torch.device("cuda:0"),
            torch=torch,
            np=np,
            autocast_dtype=torch.float16,
        )
    observed = accumulator.summary(
        model=model,
        total_input_tokens=int(coverage["input_tokens"]),
    )
    archived = json.loads(
        BENCHMARK.logical_products_path(config, condition).read_text(encoding="utf-8")
    )["measured"]
    comparison = {
        "schema_version": 1,
        "condition_id": condition["id"],
        "gpu": BENCHMARK.gpu_snapshot(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "coverage": coverage,
        "observed_h100_nvl": observed,
        "archived_h200": archived,
        "per_operation_zero_count_delta": {
            name: (
                int(observed["per_operation"][name]["zero_product_count"])
                - int(archived["per_operation"][name]["zero_product_count"])
            )
            for name in observed["per_operation"]
        },
        "interpretation": (
            "Diagnostic only: exact-zero logical counts may differ across Hopper SKUs; "
            "do not substitute these counts for the archived Run-018 R_model silently."
        ),
    }
    BENCHMARK.write_json(args.output, comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
