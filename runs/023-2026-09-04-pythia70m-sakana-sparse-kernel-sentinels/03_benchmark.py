#!/usr/bin/env python3
"""Benchmark the Run-023 Pythia-70M phase on the pinned Sakana-derived operators."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import transformers
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

from benchmark_core import (
    AttentionOpportunityAccumulator,
    LINEAR_OPERATION_SPECS,
    OccupancyAccumulator,
    covered_opportunity,
    paired_speedup,
    relative_errors,
    summarize_latency,
    validate_exact_ell,
)
from pythia_sparse import (
    ExactEllWorkspace,
    clear_adapter_workspaces,
    dense_attention_composition,
    exact_sparse_mm,
    install_sparse_linears,
    set_adapter_mode,
    sparse_attention_composition,
)
from run_config import (
    checkpoint_path,
    conditions_for_phase,
    load_config,
    logical_products_path,
    repo_path,
    write_json,
)
from sparsity_research.capture import ActivationCapture
from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata


MAX_REPETITIONS = 10_000


def flash_context():
    return sdpa_kernel(SDPBackend.FLASH_ATTENTION)


def gpu_snapshot() -> dict[str, str]:
    queries = {
        "identity": "name,uuid,driver_version,memory.total,power.limit",
        "dynamic": "timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
    }
    return {
        key: subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        for key, query in queries.items()
    }


def update_progress(path: Path, stage: str, **values: Any) -> None:
    previous = {}
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = {}
    now = time.time()
    payload = {**previous, "stage": stage, "updated_unix": now, **values}
    if "phase_etc_seconds" in values:
        payload["phase_projected_completion_unix"] = now + float(values["phase_etc_seconds"])
        payload.pop("stage_etc_seconds", None)
    write_json(path, payload)
    print(f"[progress] {stage} {values.get('condition_id', '')}", flush=True)


def load_sparse_ops(upstream_dir: Path) -> None:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    loader_path = upstream_dir.resolve() / "custom_models" / "hybrid_modules" / "sparse_ops_loader.py"
    specification = importlib.util.spec_from_file_location("run023_sparse_ops_loader", loader_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load derived sparse operator: {loader_path}")
    loader = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(loader)
    loader.load_sparse_ops(verbose=False)


def configure_model(model: Any) -> Any:
    model.eval()
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("sdpa")
    model.config.use_cache = False
    return model


def make_accumulators(condition: dict, config: dict) -> dict[str, OccupancyAccumulator]:
    sites = {LINEAR_OPERATION_SPECS[operation]["site"] for operation in condition["linear_operations"]}
    if condition["attention_microbenchmark"]:
        sites.update({"q_post", "k_post", "v"})
    widths = {"a": 512, "m": 512, "h": 2048, "z": 512, "q_post": 64, "k_post": 64, "v": 64}
    result = {}
    for layer in range(config["model"]["architecture"]["layers"]):
        for site in sorted(sites):
            width = widths[site]
            tile_width = min(config["measurement"]["tile_width"], width)
            capacities = (
                config["measurement"]["tile_payload_capacity"]
                if tile_width == 256
                else {f"factor_{factor}": tile_width // factor - 1 for factor in (8, 4, 2)}
            )
            result[f"{site}.layer_{layer}"] = OccupancyAccumulator(
                width=width,
                tile_width=tile_width,
                near_zero_thresholds=tuple(config["measurement"]["near_zero_thresholds"]),
                payload_capacities=capacities,
            )
    return result


def evaluate(
    model: Any,
    tokens: np.memmap,
    *,
    config: dict,
    condition: dict,
    progress_path: Path,
    stage: str,
    autocast_dtype: torch.dtype | None,
    batch_size: int,
    capture_sites: list[str] | None = None,
    accumulators: dict[str, OccupancyAccumulator] | None = None,
    attention_opportunities: dict[int, AttentionOpportunityAccumulator] | None = None,
    samples: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    block_size = config["model"]["architecture"]["sequence_length"]
    complete_blocks = config["validation"]["complete_blocks"]
    weighted_loss = 0.0
    sequences = 0
    batches = 0
    started = time.perf_counter()
    capture_context = (
        ActivationCapture(model, capture_sites or [], torch=torch)
        if capture_sites
        else nullcontext(None)
    )
    with torch.inference_mode(), capture_context as capture:
        for offset in range(0, complete_blocks, batch_size):
            selected = range(offset, min(offset + batch_size, complete_blocks))
            array = np.stack(
                [np.array(tokens[index * block_size : (index + 1) * block_size], dtype=np.int64, copy=True) for index in selected]
            )
            input_ids = torch.as_tensor(array, dtype=torch.long, device="cuda")
            autocast = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if autocast_dtype is not None
                else nullcontext()
            )
            with autocast, flash_context():
                output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
            batch_sequences = len(array)
            loss = float(output.loss.detach().cpu())
            if not math.isfinite(loss):
                raise RuntimeError(f"{stage}: non-finite validation loss.")
            weighted_loss += loss * batch_sequences
            sequences += batch_sequences
            batches += 1
            if capture is not None:
                if accumulators is not None:
                    for name, accumulator in accumulators.items():
                        accumulator.update(capture.activations[name], torch=torch)
                if attention_opportunities is not None:
                    for layer, accumulator in attention_opportunities.items():
                        accumulator.update(
                            capture.activations[f"q_post.layer_{layer}"],
                            capture.activations[f"v.layer_{layer}"],
                            torch=torch,
                        )
                if samples is not None and not samples:
                    for name, value in capture.activations.items():
                        if name.startswith(("q_post.", "k_post.", "v.")):
                            samples[name] = value.detach()[:1].to(torch.bfloat16).clone().contiguous()
                        else:
                            samples[name] = (
                                value.detach().reshape(-1, value.shape[-1])[: config["measurement"]["primitive_rows"]]
                                .to(torch.bfloat16)
                                .clone()
                                .contiguous()
                            )
                capture.clear()
            elapsed = time.perf_counter() - started
            rate = sequences * block_size / elapsed
            remaining = (complete_blocks - sequences) * block_size / rate
            update_progress(
                progress_path,
                stage,
                condition_id=condition["id"],
                current_loss=weighted_loss / sequences,
                completed_sequences=sequences,
                total_sequences=complete_blocks,
                tokens_per_second=rate,
                stage_etc_seconds=remaining,
            )
            del output, input_ids
    tail = len(tokens) - complete_blocks * block_size
    return {
        "loss": weighted_loss / sequences,
        "batch_size": batch_size,
        "batches": batches,
        "sequences": sequences,
        "input_tokens": sequences * block_size,
        "source_tokens": len(tokens),
        "excluded_tail_tokens": tail,
        "complete_block_coverage": sequences == complete_blocks,
        "elapsed_seconds": time.perf_counter() - started,
    }


def timed_average_ms(function: Callable[[], Any], repetitions: int) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / repetitions


def benchmark_interleaved(
    variants: dict[str, Callable[[], Any]],
    *,
    prepare: dict[str, Callable[[], None]] | None,
    warmups: int,
    blocks: int,
    target_seconds: float,
    seed: int,
    batch_size: int,
    sequence_length: int,
) -> dict[str, Any]:
    pilots = {}
    for name, function in variants.items():
        if prepare:
            prepare[name]()
        for _ in range(warmups):
            function()
        pilots[name] = timed_average_ms(function, 1)
    repetitions = max(
        1,
        min(MAX_REPETITIONS, math.ceil(target_seconds * 1000.0 / max(pilots.values()))),
    )
    samples = {name: [] for name in variants}
    orders = []
    generator = random.Random(seed)
    names = list(variants)
    for _ in range(blocks):
        order = names.copy()
        generator.shuffle(order)
        orders.append(order)
        for name in order:
            if prepare:
                prepare[name]()
            samples[name].append(timed_average_ms(variants[name], repetitions))
    result = {
        name: summarize_latency(values, batch_size=batch_size, sequence_length=sequence_length)
        for name, values in samples.items()
    }
    for name in variants:
        result[name]["pilot_ms"] = pilots[name]
        result[name]["repetitions_per_block"] = repetitions
        result[name]["clock"] = "host_wall_with_cuda_synchronize"
    result["block_orders"] = orders
    if "native_dense" in samples:
        result["paired_speedup_vs_native"] = {
            name: paired_speedup(samples["native_dense"], values)
            for name, values in samples.items()
            if name != "native_dense"
        }
    elif "dense_bf16" in samples:
        result["paired_speedup_vs_dense"] = {
            name: paired_speedup(samples["dense_bf16"], values)
            for name, values in samples.items()
            if name != "dense_bf16" and name != "pack_only"
        }
    return result


def make_timing_inputs(tokens: np.memmap, block_size: int, batch_size: int) -> list[torch.Tensor]:
    groups = []
    group_count = 5 if batch_size == 1 else 2
    for group in range(group_count):
        start_block = group * batch_size
        array = np.stack(
            [
                np.array(
                    tokens[(start_block + index) * block_size : (start_block + index + 1) * block_size],
                    dtype=np.int64,
                    copy=True,
                )
                for index in range(batch_size)
            ]
        )
        groups.append(torch.as_tensor(array, dtype=torch.long, device="cuda"))
    return groups


def benchmark_full_model(
    dense_model: Any,
    adapter_model: Any,
    adapters: dict[str, Any],
    tokens: np.memmap,
    *,
    config: dict,
    condition_index: int,
) -> dict[str, Any]:
    measurement = config["measurement"]
    result = {}
    for batch_size in measurement["full_model_batches"]:
        inputs = make_timing_inputs(tokens, config["model"]["architecture"]["sequence_length"], batch_size)
        active = [inputs[0]]

        def dense_forward() -> Any:
            with torch.inference_mode(), flash_context():
                return dense_model(input_ids=active[0], use_cache=False)

        def adapter_forward() -> Any:
            with torch.inference_mode(), flash_context():
                return adapter_model(input_ids=active[0], use_cache=False)

        variants = {
            "native_dense": dense_forward,
            "adapter_dense": adapter_forward,
            "sparse_linear": adapter_forward,
        }
        prepare = {
            "native_dense": lambda: None,
            "adapter_dense": lambda: set_adapter_mode(adapters, "dense"),
            "sparse_linear": lambda: set_adapter_mode(adapters, "sparse"),
        }
        result[str(batch_size)] = benchmark_interleaved(
            variants,
            prepare=prepare,
            warmups=measurement["full_model_warmups"],
            blocks=measurement["full_model_blocks"],
            target_seconds=measurement["full_model_target_seconds_per_block"],
            seed=config["source_run"]["seed"] + condition_index * 100 + batch_size,
            batch_size=batch_size,
            sequence_length=config["model"]["architecture"]["sequence_length"],
        )
        active[0] = inputs[-1]
        del inputs
        clear_adapter_workspaces(adapters)
        torch.cuda.empty_cache()
    return result


def linear_for_operation(model: Any, layer: int, operation: str) -> Any:
    block = model.gpt_neox.layers[layer]
    return {
        "qkv_projection": block.attention.query_key_value,
        "mlp_w1": block.mlp.dense_h_to_4h,
        "mlp_w2": block.mlp.dense_4h_to_h,
        "attention_output_projection": block.attention.dense,
    }[operation]


def benchmark_linear_primitives(
    model: Any,
    samples: dict[str, torch.Tensor],
    condition: dict,
    config: dict,
    *,
    condition_index: int,
) -> list[dict[str, Any]]:
    measurement = config["measurement"]
    tolerance = measurement["correctness_relative_l2_tolerance"]
    records = []
    for layer in range(config["model"]["architecture"]["layers"]):
        for operation in condition["linear_operations"]:
            spec = LINEAR_OPERATION_SPECS[operation]
            matrix = samples[f"{spec['site']}.layer_{layer}"].contiguous()
            linear = linear_for_operation(model, layer, operation)
            rhs = linear.weight.detach().transpose(0, 1).contiguous()
            bias = linear.bias.detach().contiguous() if linear.bias is not None else None
            workspace = ExactEllWorkspace()
            workspace.ensure(matrix.shape[0], matrix.shape[1], rhs.shape[1], device=matrix.device, torch=torch)

            def dense() -> torch.Tensor:
                return torch.nn.functional.linear(matrix, linear.weight, bias)

            def pack() -> None:
                torch.ops.sparse_ops.dense_to_ell_exact_out(
                    matrix, workspace.values, workspace.columns, workspace.counts
                )

            def kernel() -> torch.Tensor:
                torch.ops.sparse_ops.ell_spmm_raw_out(
                    workspace.values,
                    workspace.columns,
                    workspace.counts,
                    rhs,
                    matrix.shape[0],
                    matrix.shape[1],
                    rhs.shape[1],
                    matrix.shape[1],
                    matrix.shape[1],
                    workspace.output,
                )
                return workspace.output if bias is None else workspace.output + bias

            def pack_and_kernel() -> torch.Tensor:
                value = exact_sparse_mm(matrix, rhs, workspace, torch=torch)
                return value if bias is None else value + bias

            pack()
            contract = validate_exact_ell(
                matrix, workspace.values, workspace.columns, workspace.counts, torch=torch
            )
            sparse = pack_and_kernel()
            dense_value = dense()
            reference = torch.nn.functional.linear(
                matrix.float(), linear.weight.detach().float(), None if bias is None else bias.float()
            )
            errors = {
                "sparse_vs_fp32": relative_errors(sparse, reference, torch=torch),
                "dense_bf16_vs_fp32": relative_errors(dense_value, reference, torch=torch),
                "sparse_vs_dense_bf16": relative_errors(sparse, dense_value, torch=torch),
            }
            if errors["sparse_vs_fp32"]["relative_l2"] > tolerance:
                raise RuntimeError(f"{condition['id']} layer {layer} {operation} failed correctness: {errors}")
            timings = benchmark_interleaved(
                {
                    "dense_bf16": dense,
                    "pack_only": pack,
                    "kernel_only_plus_bias": kernel,
                    "pack_plus_kernel_plus_bias": pack_and_kernel,
                },
                prepare=None,
                warmups=measurement["primitive_warmups"],
                blocks=measurement["primitive_blocks"],
                target_seconds=measurement["primitive_target_seconds_per_block"],
                seed=config["source_run"]["seed"] + condition_index * 1000 + layer * 10 + list(LINEAR_OPERATION_SPECS).index(operation),
                batch_size=1,
                sequence_length=1,
            )
            records.append(
                {
                    "layer": layer,
                    "operation": operation,
                    "site": spec["site"],
                    "shape": {"M": matrix.shape[0], "K": matrix.shape[1], "N": rhs.shape[1]},
                    "contract": contract,
                    "exact_zero_count": int((matrix == 0).sum().item()),
                    "elements": matrix.numel(),
                    "errors": errors,
                    "timings": timings,
                    "workspace_allocations": workspace.allocations,
                }
            )
    return records


def benchmark_attention(
    samples: dict[str, torch.Tensor],
    config: dict,
    *,
    condition_id: str,
    condition_index: int,
) -> list[dict[str, Any]]:
    measurement = config["measurement"]
    tolerance = measurement["correctness_relative_l2_tolerance"]
    records = []
    for layer in range(config["model"]["architecture"]["layers"]):
        query = samples[f"q_post.layer_{layer}"]
        key = samples[f"k_post.layer_{layer}"]
        value = samples[f"v.layer_{layer}"]
        q_workspace = ExactEllWorkspace()
        v_workspace = ExactEllWorkspace()

        def dense() -> torch.Tensor:
            return dense_attention_composition(query, key, value, torch=torch)

        def sparse() -> torch.Tensor:
            return sparse_attention_composition(
                query,
                key,
                value,
                q_workspace=q_workspace,
                v_workspace=v_workspace,
                torch=torch,
            )

        dense_value = dense()
        sparse_value = sparse()
        errors = {"sparse_vs_dense_bf16": relative_errors(sparse_value, dense_value, torch=torch)}
        if errors["sparse_vs_dense_bf16"]["relative_l2"] > tolerance:
            raise RuntimeError(f"{condition_id} layer {layer} attention composition failed: {errors}")
        timings = benchmark_interleaved(
            {"dense_bf16": dense, "sparse_q_and_v": sparse},
            prepare=None,
            warmups=max(2, measurement["primitive_warmups"] // 4),
            blocks=measurement["primitive_blocks"],
            target_seconds=measurement["primitive_target_seconds_per_block"],
            seed=config["source_run"]["seed"] + condition_index * 1000 + 500 + layer,
            batch_size=1,
            sequence_length=config["model"]["architecture"]["sequence_length"],
        )
        speedup = timings["paired_speedup_vs_dense"]["sparse_q_and_v"]["median"]

        # Head-zero component timings isolate packing and raw kernel costs.
        q = query[0, 0].contiguous()
        k_t = key[0, 0].transpose(0, 1).contiguous()
        q_component = benchmark_mm_components(
            q,
            k_t,
            config,
            seed=config["source_run"]["seed"] + condition_index * 1000 + 600 + layer,
        )
        head_size = query.shape[-1]
        scores = (q @ k_t) / math.sqrt(head_size)
        causal = torch.ones(scores.shape, dtype=torch.bool, device=scores.device).tril()
        probabilities = torch.softmax(scores.masked_fill(~causal, -torch.inf).float(), dim=-1).to(query.dtype)
        v_t = value[0, 0].transpose(0, 1).contiguous()
        pv_component = benchmark_mm_components(
            v_t,
            probabilities.transpose(0, 1).contiguous(),
            config,
            seed=config["source_run"]["seed"] + condition_index * 1000 + 700 + layer,
        )
        records.append(
            {
                "layer": layer,
                "shape": {"B": 1, "H": query.shape[1], "T": query.shape[2], "d": query.shape[3]},
                "errors": errors,
                "composition_timings": timings,
                "composition_paired_speedup": speedup,
                "break_even": speedup > 1.0,
                "qk_q_left_component": q_component,
                "pv_vt_left_component": pv_component,
                "includes": ["packing", "per_head_dispatch", "transpose", "scale", "causal_mask", "softmax"],
            }
        )
    return records


def benchmark_mm_components(matrix: torch.Tensor, rhs: torch.Tensor, config: dict, *, seed: int) -> dict[str, Any]:
    measurement = config["measurement"]
    workspace = ExactEllWorkspace()
    workspace.ensure(matrix.shape[0], matrix.shape[1], rhs.shape[1], device=matrix.device, torch=torch)

    def dense() -> torch.Tensor:
        return matrix @ rhs

    def pack() -> None:
        torch.ops.sparse_ops.dense_to_ell_exact_out(
            matrix, workspace.values, workspace.columns, workspace.counts
        )

    def kernel() -> torch.Tensor:
        torch.ops.sparse_ops.ell_spmm_raw_out(
            workspace.values,
            workspace.columns,
            workspace.counts,
            rhs,
            matrix.shape[0],
            matrix.shape[1],
            rhs.shape[1],
            matrix.shape[1],
            matrix.shape[1],
            workspace.output,
        )
        return workspace.output

    def both() -> torch.Tensor:
        return exact_sparse_mm(matrix, rhs, workspace, torch=torch)

    pack()
    contract = validate_exact_ell(matrix, workspace.values, workspace.columns, workspace.counts, torch=torch)
    errors = relative_errors(both(), matrix.float() @ rhs.float(), torch=torch)
    if errors["relative_l2"] > measurement["correctness_relative_l2_tolerance"]:
        raise RuntimeError(f"Attention component failed correctness: {errors}")
    timings = benchmark_interleaved(
        {
            "dense_bf16": dense,
            "pack_only": pack,
            "kernel_only": kernel,
            "pack_plus_kernel": both,
        },
        prepare=None,
        warmups=measurement["primitive_warmups"],
        blocks=measurement["primitive_blocks"],
        target_seconds=measurement["primitive_target_seconds_per_block"],
        seed=seed,
        batch_size=1,
        sequence_length=1,
    )
    return {
        "shape": {"M": matrix.shape[0], "K": matrix.shape[1], "N": rhs.shape[1]},
        "contract": contract,
        "exact_zero_count": int((matrix == 0).sum().item()),
        "elements": matrix.numel(),
        "errors_vs_fp32": errors,
        "timings": timings,
    }


def combine_attention_opportunities(
    accumulators: dict[int, AttentionOpportunityAccumulator],
) -> dict[str, Any]:
    rows = [accumulator.finalize() for accumulator in accumulators.values()]
    keys = (
        "q_only_causal_zero_products",
        "v_only_causal_zero_products",
        "q_only_physical_full_gemm_zero_products",
        "v_only_physical_full_gemm_zero_products",
        "causal_products_per_operation",
        "physical_full_gemm_products_per_operation",
    )
    result = {key: sum(int(row[key]) for row in rows) for key in keys}
    result["q_only_causal_fraction"] = (
        result["q_only_causal_zero_products"] / result["causal_products_per_operation"]
    )
    result["v_only_causal_fraction"] = (
        result["v_only_causal_zero_products"] / result["causal_products_per_operation"]
    )
    result["layers"] = len(rows)
    result["per_layer"] = rows
    result["lower_bound_not_qk_union"] = True
    result["integer_pooling"] = True
    return result


def run_condition(
    condition: dict,
    condition_index: int,
    *,
    config: dict,
    tokens: np.memmap,
    output_dir: Path,
    progress_path: Path,
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    checkpoint = checkpoint_path(config, condition)
    logical = json.loads(logical_products_path(config, condition).read_text(encoding="utf-8"))
    update_progress(progress_path, "load_dense_model", condition_id=condition["id"])
    dense_model = configure_model(
        load_checkpoint_pythia(AutoModelForCausalLM, checkpoint, torch=torch).to(device=device, dtype=torch.float32)
    )
    topology = topology_metadata(dense_model)
    if topology["topology_id"] != condition["topology_id"]:
        raise RuntimeError(f"Loaded topology mismatch: {topology}")

    source_attention = (
        {
            layer: AttentionOpportunityAccumulator(config["model"]["architecture"]["sequence_length"])
            for layer in range(config["model"]["architecture"]["layers"])
        }
        if condition["attention_microbenchmark"]
        else None
    )
    source_validation = evaluate(
        dense_model,
        tokens,
        config=config,
        condition=condition,
        progress_path=progress_path,
        stage="source_fp16_validation",
        autocast_dtype=torch.float16,
        batch_size=config["validation"]["source_reproduction_batch_size"],
    )
    source_validation["archived_loss"] = condition["archived_validation_loss"]
    source_validation["absolute_loss_difference"] = abs(
        source_validation["loss"] - condition["archived_validation_loss"]
    )
    if source_validation["absolute_loss_difference"] > config["validation"]["source_loss_absolute_tolerance"]:
        raise RuntimeError(f"{condition['id']}: archived validation loss not reproduced: {source_validation}")
    attention_opportunity_validation = None
    if source_attention is not None:
        attention_opportunity_validation = evaluate(
            dense_model,
            tokens,
            config=config,
            condition=condition,
            progress_path=progress_path,
            stage="source_attention_opportunity",
            autocast_dtype=torch.float16,
            batch_size=config["validation"]["logical_opportunity_batch_size"],
            capture_sites=["q_post", "v"],
            attention_opportunities=source_attention,
        )
    attention_opportunity = combine_attention_opportunities(source_attention) if source_attention else None

    dense_model.to(dtype=torch.bfloat16)
    accumulators = make_accumulators(condition, config)
    samples: dict[str, torch.Tensor] = {}
    capture_sites = sorted({name.split(".")[0] for name in accumulators})
    dense_bf16_validation = evaluate(
        dense_model,
        tokens,
        config=config,
        condition=condition,
        progress_path=progress_path,
        stage="dense_bf16_validation_and_occupancy",
        autocast_dtype=None,
        batch_size=config["validation"]["runtime_equivalence_batch_size"],
        capture_sites=capture_sites,
        accumulators=accumulators,
        samples=samples,
    )
    occupancy = {name: accumulator.finalize() for name, accumulator in accumulators.items()}

    update_progress(progress_path, "load_sparse_model", condition_id=condition["id"])
    sparse_model = configure_model(
        load_checkpoint_pythia(AutoModelForCausalLM, checkpoint, torch=torch).to(device=device, dtype=torch.bfloat16)
    )
    adapters = install_sparse_linears(sparse_model, condition["linear_operations"], torch=torch)
    sparse_model.eval()
    set_adapter_mode(adapters, "sparse")
    sparse_validation = evaluate(
        sparse_model,
        tokens,
        config=config,
        condition=condition,
        progress_path=progress_path,
        stage="sparse_linear_bf16_validation",
        autocast_dtype=None,
        batch_size=config["validation"]["runtime_equivalence_batch_size"],
    )
    sparse_validation["absolute_loss_difference_from_dense_bf16"] = abs(
        sparse_validation["loss"] - dense_bf16_validation["loss"]
    )
    if (
        sparse_validation["absolute_loss_difference_from_dense_bf16"]
        > config["validation"]["sparse_vs_dense_loss_absolute_tolerance"]
    ):
        raise RuntimeError(f"{condition['id']}: sparse full-validation loss mismatch: {sparse_validation}")

    update_progress(progress_path, "full_model_timing", condition_id=condition["id"])
    full_model = benchmark_full_model(
        dense_model,
        sparse_model,
        adapters,
        tokens,
        config=config,
        condition_index=condition_index,
    )
    update_progress(progress_path, "linear_primitive_timing", condition_id=condition["id"])
    linear_primitives = benchmark_linear_primitives(
        dense_model, samples, condition, config, condition_index=condition_index
    )
    attention = None
    if condition["attention_microbenchmark"]:
        update_progress(progress_path, "attention_composition_timing", condition_id=condition["id"])
        attention = benchmark_attention(
            samples,
            config,
            condition_id=condition["id"],
            condition_index=condition_index,
        )
    full_model_coverage = covered_opportunity(
        logical,
        condition["linear_operations"],
    )
    linear_plus_attention_coverage = (
        covered_opportunity(
            logical,
            condition["linear_operations"],
            attention=attention_opportunity,
        )
        if attention_opportunity is not None
        else None
    )
    result = {
        "schema_version": 1,
        "run": config["name"],
        "derivative_label": config["upstream"]["derivative_label"],
        "condition": condition,
        "checkpoint": str(checkpoint),
        "topology": topology,
        "validation": {
            "source_fp32_parameters_fp16_autocast": source_validation,
            "native_dense_bf16": dense_bf16_validation,
            "sparse_linear_bf16": sparse_validation,
        },
        "canonical_logical_products": logical,
        "kernel_covered_opportunity": full_model_coverage,
        "kernel_covered_opportunity_with_separate_attention": linear_plus_attention_coverage,
        "attention_opportunity": attention_opportunity,
        "attention_opportunity_validation": attention_opportunity_validation,
        "runtime_occupancy": occupancy,
        "full_model": full_model,
        "linear_primitives": linear_primitives,
        "attention_compositions": attention,
        "memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "interpretation_contract": {
            "r_model_is_logical_not_speedup": True,
            "r_covered_is_kernel_specific_not_speedup": True,
            "full_model_r_covered_excludes_separately_timed_attention": True,
            "full_model_sparse_path_covers_linear_operations_only": condition["linear_operations"],
            "attention_is_separate_unfused_composition": condition["attention_microbenchmark"],
            "lm_head_remains_dense": True,
            "prefill_only": True,
            "teal_not_applied": True,
        },
    }
    write_json(output_dir / f"{condition['id']}.json", result)
    del dense_model, sparse_model, adapters, samples
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("sentinel", "remainder"), default="sentinel")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if tuple(torch.cuda.get_device_capability(0)) != tuple(config["runtime"]["compute_capability"]):
        raise RuntimeError("Run 023 requires exact Hopper compute capability 9.0.")
    if torch.__version__.split("+")[0] != config["runtime"]["pythia_torch"]:
        raise RuntimeError(f"PyTorch mismatch: {torch.__version__}")
    if transformers.__version__ != config["runtime"]["pythia_transformers"]:
        raise RuntimeError(f"Transformers mismatch: {transformers.__version__}")
    if np.__version__ != config["runtime"]["pythia_numpy"]:
        raise RuntimeError(f"NumPy mismatch: {np.__version__}")
    load_sparse_ops(args.upstream_dir)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    tokens = np.memmap(repo_path(config["validation"]["tokens"]), mode="r", dtype=np.int32)
    conditions = conditions_for_phase(config, args.phase)
    started = time.perf_counter()
    results = []
    for index, condition in enumerate(conditions):
        torch.cuda.reset_peak_memory_stats()
        result = run_condition(
            condition,
            index,
            config=config,
            tokens=tokens,
            output_dir=output_dir,
            progress_path=progress_path,
        )
        results.append(result)
        elapsed = time.perf_counter() - started
        remaining = elapsed / len(results) * (len(conditions) - len(results))
        update_progress(
            progress_path,
            "condition_complete",
            phase=args.phase,
            condition_id=condition["id"],
            completed_conditions=len(results),
            total_conditions=len(conditions),
            current_loss=result["validation"]["sparse_linear_bf16"]["loss"],
            tokens_per_second=result["full_model"]["1"]["sparse_linear"]["tokens_per_second"],
            phase_etc_seconds=remaining,
        )
    attention_records = [
        record
        for result in results
        for record in (result["attention_compositions"] or [])
    ]
    cohort = {
        "schema_version": 1,
        "run": config["name"],
        "phase": args.phase,
        "condition_ids": [condition["id"] for condition in conditions],
        "condition_files": [f"{condition['id']}.json" for condition in conditions],
        "completed_conditions": len(results),
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu": gpu_snapshot(),
            "upstream_directory": str(args.upstream_dir.resolve()),
            "upstream_commit": config["upstream"]["commit"],
            "patch_sha256": config["upstream"]["patch_sha256"],
        },
        "attention_promotion": {
            "tested": bool(attention_records),
            "all_layers_break_even": bool(attention_records) and all(record["break_even"] for record in attention_records),
            "any_layer_break_even": any(record["break_even"] for record in attention_records),
            "stop_if_not_break_even": True,
            "custom_fused_causal_kernel_authorized": False,
        },
    }
    write_json(output_dir / "cohort.json", cohort)
    update_progress(
        progress_path,
        "benchmark_complete",
        phase=args.phase,
        completed_conditions=len(results),
        total_conditions=len(conditions),
        phase_etc_seconds=0,
    )
    print(f"PASS: Run 023 {args.phase} benchmark -> {output_dir}")


if __name__ == "__main__":
    main()
