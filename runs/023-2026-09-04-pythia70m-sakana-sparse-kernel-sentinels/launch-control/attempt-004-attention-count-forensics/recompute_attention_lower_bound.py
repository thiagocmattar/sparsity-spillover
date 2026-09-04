#!/usr/bin/env python3
"""Recompute A7 kappa=0 q-only/v-only counts on the timing GPU."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


RUN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUN_DIR))
SPEC = importlib.util.spec_from_file_location("run023_benchmark_forensics", RUN_DIR / "03_benchmark.py")
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
    checkpoint = BENCHMARK.checkpoint_path(config, condition)
    model = BENCHMARK.configure_model(
        BENCHMARK.load_checkpoint_pythia(
            BENCHMARK.AutoModelForCausalLM,
            checkpoint,
            torch=torch,
        ).to(device=torch.device("cuda:0"), dtype=torch.float32)
    )
    tokens = np.memmap(
        BENCHMARK.repo_path(config["validation"]["tokens"]),
        dtype=np.int32,
        mode="r",
    )
    accumulators = {
        layer: BENCHMARK.AttentionOpportunityAccumulator(
            config["model"]["architecture"]["sequence_length"]
        )
        for layer in range(config["model"]["architecture"]["layers"])
    }
    coverage = BENCHMARK.evaluate(
        model,
        tokens,
        config=config,
        condition=condition,
        progress_path=args.output.with_suffix(".progress.json"),
        stage="forensic_source_attention_opportunity",
        autocast_dtype=torch.float16,
        batch_size=config["validation"]["logical_opportunity_batch_size"],
        capture_sites=["q_post", "v"],
        attention_opportunities=accumulators,
    )
    observed = BENCHMARK.combine_attention_opportunities(accumulators)
    canonical = json.loads(
        BENCHMARK.logical_products_path(config, condition).read_text(encoding="utf-8")
    )["measured"]["per_operation"]
    comparison = {
        "schema_version": 1,
        "condition_id": condition["id"],
        "gpu": BENCHMARK.gpu_snapshot(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "coverage": coverage,
        "observed": observed,
        "canonical_run018": {
            "qk_union_zero_products": int(canonical["qk_scores"]["zero_product_count"]),
            "pv_union_zero_products": int(canonical["probability_value"]["zero_product_count"]),
            "source_gpu": "NVIDIA H200",
        },
        "deltas": {
            "q_only_minus_canonical_qk_union": (
                observed["q_only_causal_zero_products"]
                - int(canonical["qk_scores"]["zero_product_count"])
            ),
            "v_only_minus_canonical_pv_union": (
                observed["v_only_causal_zero_products"]
                - int(canonical["probability_value"]["zero_product_count"])
            ),
        },
        "interpretation": (
            "Diagnostic only: compare H100-NVL operand-zero lower bounds with archived "
            "H200 union counts; do not combine across hardware realizations."
        ),
    }
    BENCHMARK.write_json(args.output, comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
