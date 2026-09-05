#!/usr/bin/env python3
"""Measure whether the exact batch-32 full-model path has safe VRAM headroom."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from _reuse_run023 import load_run023_module
from pythia_sparse import install_sparse_linears, set_adapter_mode
from run_config import checkpoint_path, condition_by_id, load_config, repo_path, write_json
from sparsity_research.pythia import load_checkpoint_pythia


_benchmark = load_run023_module("_run023_batch_calibration_benchmark", "03_benchmark.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config()
    batch_size = config["measurement"]["optional_full_model_batch"]
    minimum_headroom = config["measurement"]["minimum_optional_batch_headroom_fraction"]
    condition = condition_by_id(config, "a4-ol1-kappa-0p5")
    tokens = np.memmap(repo_path(config["validation"]["tokens"]), mode="r", dtype=np.int32)
    output = {
        "schema_version": 1,
        "condition_id": condition["id"],
        "batch_size": batch_size,
        "minimum_headroom_fraction": minimum_headroom,
        "representative_scope": "two BF16 models; native, adapter-dense, and sparse full forwards",
        "eligible": False,
    }
    started = time.perf_counter()
    try:
        _benchmark.load_sparse_ops(args.upstream_dir)
        checkpoint = checkpoint_path(config, condition)
        dense_model = _benchmark.configure_model(
            load_checkpoint_pythia(AutoModelForCausalLM, checkpoint, torch=torch).to(
                device="cuda", dtype=torch.bfloat16
            )
        )
        sparse_model = _benchmark.configure_model(
            load_checkpoint_pythia(AutoModelForCausalLM, checkpoint, torch=torch).to(
                device="cuda", dtype=torch.bfloat16
            )
        )
        adapters = install_sparse_linears(sparse_model, condition["linear_operations"], torch=torch)
        dense_model.eval()
        sparse_model.eval()
        inputs = _benchmark.make_timing_inputs(
            tokens, config["model"]["architecture"]["sequence_length"], batch_size
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        timings = {}
        for name, model, mode in (
            ("native_dense", dense_model, None),
            ("adapter_dense", sparse_model, "dense"),
            ("sparse_linear", sparse_model, "sparse"),
        ):
            if mode is not None:
                set_adapter_mode(adapters, mode)
            torch.cuda.synchronize()
            before = time.perf_counter()
            with torch.inference_mode(), _benchmark.flash_context():
                result = model(input_ids=inputs[0], use_cache=False)
            torch.cuda.synchronize()
            timings[name] = (time.perf_counter() - before) * 1000
            del result
        total = torch.cuda.get_device_properties(0).total_memory
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        headroom = (total - peak_reserved) / total
        output.update(
            {
                "eligible": headroom >= minimum_headroom,
                "gpu": torch.cuda.get_device_name(0),
                "total_memory_bytes": total,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "headroom_fraction": headroom,
                "forward_ms": timings,
            }
        )
    except torch.cuda.OutOfMemoryError as error:
        output.update({"eligible": False, "failure": "cuda_out_of_memory", "detail": str(error)})
    finally:
        output["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

