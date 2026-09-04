#!/usr/bin/env python3
"""Run the Pythia-14M A0 validation, occupancy, and raw-ELL benchmarks."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import transformers
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

from benchmark_core import (
    OccupancyAccumulator,
    pack_exact_ell,
    relative_errors,
    summarize_latency,
    unpack_exact_ell,
    validate_raw_ell_contract,
)
from run_config import load_config, repo_path, write_json
from sparsity_research.capture import ActivationCapture
from sparsity_research.evaluation import evaluate_complete_blocks
from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata


MAX_TIMING_REPETITIONS = 50_000


def gpu_snapshot() -> dict[str, str]:
    queries = {
        "identity": "name,uuid,driver_version,memory.total,power.limit",
        "dynamic": "timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu",
    }
    return {
        name: subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        for name, query in queries.items()
    }


def update_progress(path: Path, stage: str, **values: object) -> None:
    write_json(path, {"stage": stage, "updated_unix": time.time(), **values})
    print(f"[progress] {stage}", flush=True)


def flash_context():
    return sdpa_kernel(SDPBackend.FLASH_ATTENTION)


def timed_average_ms(function: Callable[[], object], repetitions: int) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / repetitions


def adaptive_repetitions(function: Callable[[], object], target_seconds: float) -> tuple[int, float]:
    pilot_ms = timed_average_ms(function, 1)
    repetitions = max(1, min(MAX_TIMING_REPETITIONS, math.ceil(target_seconds * 1000.0 / pilot_ms)))
    return repetitions, pilot_ms


def benchmark_callables_interleaved(
    functions: dict[str, Callable[[], object]],
    *,
    warmups: int,
    blocks: int,
    target_seconds: float,
) -> dict[str, dict]:
    names = list(functions)
    pilots: dict[str, float] = {}
    repetitions: dict[str, int] = {}
    samples = {name: [] for name in names}
    for name, function in functions.items():
        for _ in range(warmups):
            function()
        torch.cuda.synchronize()
        repetitions[name], pilots[name] = adaptive_repetitions(function, target_seconds)
    for block in range(blocks):
        rotated = names[block % len(names) :] + names[: block % len(names)]
        for name in rotated:
            samples[name].append(timed_average_ms(functions[name], repetitions[name]))
    result = {}
    for name in names:
        result[name] = summarize_latency(samples[name], batch_size=1, sequence_length=1)
        result[name]["operations_per_second"] = 1000.0 / result[name]["median_ms"]
        del result[name]["sequences_per_second"]
        del result[name]["tokens_per_second"]
        result[name].update(
            {
                "repetitions_per_block": repetitions[name],
                "pilot_ms": pilots[name],
                "clock": "host_wall_with_cuda_synchronize",
                "block_order": "cyclic_interleave",
            }
        )
    return result


def benchmark_full_model_pair(
    model: object,
    inputs: list[torch.Tensor],
    *,
    batch_size: int,
    sequence_length: int,
    warmups: int,
    blocks: int,
    target_seconds: float,
) -> dict:
    active_input = [inputs[0]]

    def forward() -> object:
        with torch.inference_mode(), flash_context():
            return model(input_ids=active_input[0], use_cache=False)

    def with_identity_hooks(callback: Callable[[], object]) -> object:
        handles = [
            layer.mlp.act.register_forward_hook(lambda _module, _arguments, output: output)
            for layer in model.gpt_neox.layers
        ]
        try:
            return callback()
        finally:
            for handle in handles:
                handle.remove()

    for _ in range(warmups):
        forward()

    def warm_identity() -> None:
        for _ in range(warmups):
            forward()

    with_identity_hooks(warm_identity)
    torch.cuda.synchronize()
    dense_reps, dense_pilot = adaptive_repetitions(forward, target_seconds)
    hooked_pilot = with_identity_hooks(lambda: timed_average_ms(forward, 1))
    hooked_reps = max(1, min(MAX_TIMING_REPETITIONS, math.ceil(target_seconds * 1000.0 / hooked_pilot)))
    repetitions = min(dense_reps, hooked_reps)
    values = {"dense": [], "identity_hook": []}
    for block in range(blocks):
        active_input[0] = inputs[block % len(inputs)]
        order = ("dense", "identity_hook") if block % 2 == 0 else ("identity_hook", "dense")
        for variant in order:
            if variant == "dense":
                values[variant].append(timed_average_ms(forward, repetitions))
            else:
                values[variant].append(with_identity_hooks(lambda: timed_average_ms(forward, repetitions)))
    result = {
        name: summarize_latency(sample, batch_size=batch_size, sequence_length=sequence_length)
        for name, sample in values.items()
    }
    result["dense"]["pilot_ms"] = dense_pilot
    result["identity_hook"]["pilot_ms"] = hooked_pilot
    for value in result.values():
        value["repetitions_per_block"] = repetitions
        value["clock"] = "host_wall_with_cuda_synchronize"
    result["identity_hook_overhead_ratio"] = (
        result["identity_hook"]["median_ms"] / result["dense"]["median_ms"] - 1.0
    )
    return result


def make_timing_inputs(tokens: np.memmap, block_size: int, batch_size: int, device: torch.device) -> list[torch.Tensor]:
    if batch_size == 1:
        starts = [index * block_size for index in range(5)]
        arrays = [np.array(tokens[start : start + block_size], dtype=np.int64, copy=True)[None, :] for start in starts]
    else:
        arrays = [
            np.stack(
                [np.array(tokens[index * block_size : (index + 1) * block_size], dtype=np.int64, copy=True) for index in range(batch_size)]
            )
        ]
    return [torch.as_tensor(array, dtype=torch.long, device=device) for array in arrays]


def benchmark_raw_ell(
    model: object,
    captured: dict[int, torch.Tensor],
    config: dict,
) -> list[dict]:
    settings = config["measurement"]
    tolerance = settings["correctness_relative_l2_tolerance"]
    records = []
    for layer_index, activation in sorted(captured.items()):
        layer = model.gpt_neox.layers[layer_index]
        rhs = layer.mlp.dense_4h_to_h.weight.detach().transpose(0, 1).contiguous()
        bias = layer.mlp.dense_4h_to_h.bias.detach().contiguous()
        for rows in settings["raw_ell_rows"]:
            matrix = activation[:rows].contiguous()
            values, indices, counts, stride = pack_exact_ell(
                matrix, torch=torch, alignment=settings["raw_ell_alignment"]
            )
            contract = validate_raw_ell_contract(
                values,
                indices,
                counts,
                rhs,
                overflow_threshold=stride,
                torch=torch,
            )
            reconstructed = unpack_exact_ell(values, indices, counts, columns=matrix.shape[1], torch=torch)
            if not bool(torch.equal(matrix, reconstructed)):
                raise RuntimeError(f"ELL round trip failed at layer {layer_index}, M={rows}.")

            def dense_operation() -> torch.Tensor:
                return matrix @ rhs + bias

            def sparse_operation() -> torch.Tensor:
                return torch.ops.sparse_ops.ell_spmm_raw(
                    values,
                    indices,
                    counts,
                    rhs,
                    contract["M"],
                    contract["K"],
                    contract["N"],
                    stride,
                    stride,
                ) + bias

            def pack_only() -> tuple:
                return pack_exact_ell(matrix, torch=torch, alignment=settings["raw_ell_alignment"])

            def pack_and_sparse() -> torch.Tensor:
                packed_values, packed_indices, packed_counts, packed_stride = pack_exact_ell(
                    matrix, torch=torch, alignment=settings["raw_ell_alignment"]
                )
                packed_contract = validate_raw_ell_contract(
                    packed_values,
                    packed_indices,
                    packed_counts,
                    rhs,
                    overflow_threshold=packed_stride,
                    torch=torch,
                )
                return torch.ops.sparse_ops.ell_spmm_raw(
                    packed_values,
                    packed_indices,
                    packed_counts,
                    rhs,
                    packed_contract["M"],
                    packed_contract["K"],
                    packed_contract["N"],
                    packed_stride,
                    packed_stride,
                ) + bias

            dense = dense_operation()
            sparse = sparse_operation()
            reference = matrix.float() @ rhs.float() + bias.float()
            torch.cuda.synchronize()
            errors = {
                "sparse_vs_dense_bf16": relative_errors(sparse, dense, torch=torch),
                "sparse_vs_fp32": relative_errors(sparse, reference, torch=torch),
                "dense_bf16_vs_fp32": relative_errors(dense, reference, torch=torch),
            }
            if errors["sparse_vs_fp32"]["relative_l2"] > tolerance:
                raise RuntimeError(f"ELL error exceeded tolerance at layer {layer_index}, M={rows}: {errors}")
            timings = benchmark_callables_interleaved(
                {
                    "dense_bf16": dense_operation,
                    "sparse_kernel_plus_bias": sparse_operation,
                    "exact_pack": pack_only,
                    "exact_pack_plus_sparse_kernel_plus_bias": pack_and_sparse,
                },
                warmups=settings["raw_ell_warmups"],
                blocks=settings["raw_ell_blocks"],
                target_seconds=settings["raw_ell_target_seconds_per_block"],
            )
            dense_ms = timings["dense_bf16"]["median_ms"]
            for timing in timings.values():
                timing["speedup_vs_dense"] = dense_ms / timing["median_ms"]
            records.append(
                {
                    "layer": layer_index,
                    "rows": rows,
                    "contract": contract,
                    "ell_stride": stride,
                    "overflow_threshold": stride,
                    "skipped_rows": 0,
                    "exact_pack_round_trip": True,
                    "exact_zero_count": int((matrix == 0).sum().item()),
                    "elements": matrix.numel(),
                    "errors": errors,
                    "timings": timings,
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    update_progress(progress_path, "starting")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if torch.cuda.get_device_capability(0) != tuple(config["runtime"]["required_compute_capability"]):
        raise RuntimeError("Run 022 requires exact Hopper compute capability 9.0.")
    if torch.__version__.split("+")[0] != config["runtime"]["pythia_torch"]:
        raise RuntimeError(f"PyTorch mismatch: {torch.__version__}")
    if transformers.__version__ != config["runtime"]["pythia_transformers"]:
        raise RuntimeError(f"Transformers mismatch: {transformers.__version__}")
    if np.__version__ != config["runtime"]["pythia_numpy"]:
        raise RuntimeError(f"NumPy mismatch: {np.__version__}")

    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    loader_path = args.upstream_dir.resolve() / "custom_models" / "hybrid_modules" / "sparse_ops_loader.py"
    specification = importlib.util.spec_from_file_location("run022_sparse_ops_loader", loader_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load upstream operator loader: {loader_path}")
    loader = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(loader)
    loader.load_sparse_ops(verbose=False)
    device = torch.device("cuda:0")
    checkpoint = repo_path(config["model"]["checkpoint"])
    tokens = np.memmap(repo_path(config["validation"]["tokens"]), mode="r", dtype=np.int32)
    model = load_checkpoint_pythia(AutoModelForCausalLM, checkpoint, torch=torch).to(device=device, dtype=torch.float32)
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("sdpa")
    topology = topology_metadata(model)
    if topology["topology_id"] != "A0" or topology["active_sites"]:
        raise RuntimeError(f"Unexpected topology: {topology}")
    weight_norms = [
        {
            "layer": index,
            "w2_frobenius_l2": float(layer.mlp.dense_4h_to_h.weight.detach().float().norm().item()),
            "w2_bias_l2": float(layer.mlp.dense_4h_to_h.bias.detach().float().norm().item()),
        }
        for index, layer in enumerate(model.gpt_neox.layers)
    ]

    update_progress(progress_path, "source_validation")
    with flash_context():
        source_validation = evaluate_complete_blocks(
            model=model,
            tokens=tokens,
            block_size=config["model"]["architecture"]["sequence_length"],
            batch_size=config["validation"]["batch_size"],
            device=device,
            torch=torch,
            np=np,
            autocast_dtype=torch.float16,
        )
    archived = config["model"]["archived_validation"]
    source_validation["archived_loss"] = archived["loss"]
    source_validation["absolute_loss_difference"] = abs(source_validation["loss"] - archived["loss"])
    if source_validation["absolute_loss_difference"] > archived["tolerance"]:
        raise RuntimeError(f"Archived validation loss was not reproduced: {source_validation}")

    model.to(dtype=torch.bfloat16)
    accumulators = {
        layer: OccupancyAccumulator(
            width=config["model"]["architecture"]["intermediate_size"],
            tile_width=config["measurement"]["tile_width"],
            near_zero_thresholds=tuple(config["measurement"]["near_zero_thresholds"]),
            payload_capacities=config["measurement"]["tile_payload_capacity"],
        )
        for layer in range(config["model"]["architecture"]["layers"])
    }
    captured_samples: dict[int, torch.Tensor] = {}
    update_progress(progress_path, "bf16_validation_and_h_occupancy")
    with ActivationCapture(model, ["h"], torch=torch) as capture:
        def after_batch(_output: object, _sequences: int) -> None:
            for layer in accumulators:
                value = capture.activations[f"h.layer_{layer}"]
                accumulators[layer].update(value, torch=torch)
                if layer not in captured_samples:
                    captured_samples[layer] = value.detach().reshape(-1, value.shape[-1])[:2048].clone().contiguous()
            capture.clear()

        with flash_context():
            bf16_validation = evaluate_complete_blocks(
                model=model,
                tokens=tokens,
                block_size=config["model"]["architecture"]["sequence_length"],
                batch_size=config["validation"]["batch_size"],
                device=device,
                torch=torch,
                np=np,
                autocast_dtype=None,
                after_batch=after_batch,
            )
    occupancy = [
        {"layer": layer, **accumulator.finalize()} for layer, accumulator in sorted(accumulators.items())
    ]

    update_progress(progress_path, "full_model_timing")
    full_model = {}
    for batch_size in config["measurement"]["full_model_batches"]:
        inputs = make_timing_inputs(
            tokens,
            config["model"]["architecture"]["sequence_length"],
            batch_size,
            device,
        )
        full_model[str(batch_size)] = benchmark_full_model_pair(
            model,
            inputs,
            batch_size=batch_size,
            sequence_length=config["model"]["architecture"]["sequence_length"],
            warmups=config["measurement"]["full_model_warmups"],
            blocks=config["measurement"]["full_model_blocks"],
            target_seconds=config["measurement"]["full_model_target_seconds_per_block"],
        )
        del inputs
        torch.cuda.empty_cache()

    update_progress(progress_path, "raw_ell_timing")
    raw_ell = benchmark_raw_ell(model, captured_samples, config)
    result = {
        "schema_version": 1,
        "run": config["name"],
        "completed_unix": time.time(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu": gpu_snapshot(),
            "upstream_directory": str(args.upstream_dir.resolve()),
        },
        "model": {
            "checkpoint": config["model"]["checkpoint"],
            "condition_id": config["model"]["condition_id"],
            "topology": topology,
            "parameter_dtype_source": "float32",
            "runtime_dtype": "bfloat16",
        },
        "validation": {
            "source_fp32_parameters_fp16_autocast": source_validation,
            "bf16_parameters": bf16_validation,
        },
        "weight_norms": weight_norms,
        "h_occupancy": occupancy,
        "full_model": full_model,
        "raw_ell": raw_ell,
        "interpretation_contract": {
            "a0_is_negative_control": True,
            "raw_ell_is_not_end_to_end_model_replacement": True,
            "r_model_is_not_runtime_speedup": True,
            "attention_not_tested": True,
        },
    }
    write_json(output_dir / "a0-results.json", result)
    update_progress(progress_path, "benchmark_complete", raw_ell_records=len(raw_ell))
    print(f"PASS: benchmark -> {output_dir / 'a0-results.json'}")


if __name__ == "__main__":
    main()
