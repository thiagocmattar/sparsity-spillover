#!/usr/bin/env python3
"""Audit H100 Q/V-only opportunity against pinned A100 canonical unions."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


RUN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUN_DIR))

from benchmark_core import AttentionOpportunityAccumulator  # noqa: E402
from run_config import (  # noqa: E402
    checkpoint_path,
    condition_by_id,
    logical_products_path,
    repo_path,
    write_json,
)


def load_benchmark():
    path = RUN_DIR / "03_benchmark.py"
    specification = importlib.util.spec_from_file_location("run024_attention_opportunity_audit", path)
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
    accumulators = {
        layer: AttentionOpportunityAccumulator(config["model"]["architecture"]["sequence_length"])
        for layer in range(config["model"]["architecture"]["layers"])
    }
    validation = benchmark.evaluate(
        model,
        tokens,
        config=config,
        condition=condition,
        progress_path=args.output.resolve().with_name("audit-progress.json"),
        stage="source_attention_opportunity",
        autocast_dtype=base.torch.float16,
        batch_size=config["validation"]["logical_opportunity_batch_size"],
        capture_sites=["q_post", "v"],
        attention_opportunities=accumulators,
    )
    measured = benchmark.combine_attention_opportunities(accumulators)
    canonical = json.loads(logical_products_path(config, condition).read_text(encoding="utf-8"))["measured"]
    canonical_qk = int(canonical["per_operation"]["qk_scores"]["zero_product_count"])
    canonical_pv = int(canonical["per_operation"]["probability_value"]["zero_product_count"])
    current_q = int(measured["q_only_causal_zero_products"])
    current_v = int(measured["v_only_causal_zero_products"])
    output = {
        "schema_version": 1,
        "condition_id": condition["id"],
        "gpu": base.gpu_snapshot(),
        "validation": validation,
        "h100_eager_qv_opportunity": measured,
        "run019_a100_eager_canonical_union": {
            "qk_scores_zero_product_count": canonical_qk,
            "probability_value_zero_product_count": canonical_pv,
            "R_model": canonical["R_model"],
        },
        "comparison": {
            "q_only_minus_canonical_qk_union": current_q - canonical_qk,
            "v_only_minus_canonical_pv_union": current_v - canonical_pv,
            "q_only_within_canonical_union": current_q <= canonical_qk,
            "v_only_within_canonical_union": current_v <= canonical_pv,
            "cross_device_extension_is_valid": current_q <= canonical_qk and current_v <= canonical_pv,
        },
        "interpretation": (
            "H100 Q/V-only counts are valid on their own causal denominators. "
            "They may extend canonical A100 R_model only when each lower-bound component is within its matched A100 union."
        ),
    }
    write_json(args.output.resolve(), output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

