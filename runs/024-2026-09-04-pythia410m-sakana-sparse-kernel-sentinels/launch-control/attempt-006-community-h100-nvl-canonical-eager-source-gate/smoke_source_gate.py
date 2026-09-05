#!/usr/bin/env python3
"""Exercise the patched source-identity gate before the full Attempt-006 run."""

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
    specification = importlib.util.spec_from_file_location("run024_source_gate_smoke", path)
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
    if condition["archived_validation_loss"] != condition["canonical_eager_validation_loss"]:
        raise RuntimeError("Resolved source reference is not the canonical eager endpoint.")
    tokens = np.memmap(repo_path(config["validation"]["tokens"]), dtype=np.int32, mode="r")
    model = base.configure_model(
        base.load_checkpoint_pythia(
            base.AutoModelForCausalLM,
            checkpoint_path(config, condition),
            torch=base.torch,
        ).to(device="cuda:0", dtype=base.torch.float32)
    )
    record = benchmark.evaluate(
        model,
        tokens,
        config=config,
        condition=condition,
        progress_path=args.output.resolve().with_name("smoke-progress.json"),
        stage="source_fp16_validation",
        autocast_dtype=base.torch.float16,
        batch_size=config["validation"]["source_reproduction_batch_size"],
    )
    difference = abs(record["loss"] - condition["archived_validation_loss"])
    expected_labels = (
        record.get("attention_implementation") == "eager"
        and record.get("matched_canonical_run019_protocol") is True
        and record.get("source_identity_reference") == "canonical_eager"
        and record.get("runtime_attention_implementation_before_override") == "sdpa"
    )
    passed = difference <= config["validation"]["source_loss_absolute_tolerance"] and expected_labels
    output = {
        "schema_version": 1,
        "passed": passed,
        "condition_id": condition["id"],
        "loss": record["loss"],
        "canonical_eager_loss": condition["archived_validation_loss"],
        "archived_a100_sdpa_loss": condition["archived_sdpa_validation_loss"],
        "absolute_difference": difference,
        "tolerance": config["validation"]["source_loss_absolute_tolerance"],
        "labels": {
            "attention_implementation": record.get("attention_implementation"),
            "matched_canonical_run019_protocol": record.get("matched_canonical_run019_protocol"),
            "source_identity_reference": record.get("source_identity_reference"),
            "runtime_attention_implementation_before_override": record.get(
                "runtime_attention_implementation_before_override"
            ),
        },
        "coverage": {
            "sequences": record["sequences"],
            "input_tokens": record["input_tokens"],
            "excluded_tail_tokens": record["excluded_tail_tokens"],
            "complete_block_coverage": record["complete_block_coverage"],
        },
    }
    write_json(args.output.resolve(), output)
    if not passed:
        raise RuntimeError(output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

