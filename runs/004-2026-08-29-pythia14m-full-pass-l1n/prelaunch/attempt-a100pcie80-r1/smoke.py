"""Non-evidence prelaunch checks for the exact Run 004 optimization boundary."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from sparsity_research.capture import ActivationCapture
from sparsity_research.pressure import parse_pressure_config
from sparsity_research.pythia import build_random_pythia, topology_metadata

from initialization import apply_pythia_14m_initialization, verify_recipe_model
from optimizer_boundary import DynamicLossScaler, build_recipe_adamw, run_recipe_boundary
from run_config import (
    RUN_DIR,
    load_config,
    mapping,
    parameter_sha256,
    require_cuda,
    resolved_condition_config,
    seed_everything,
    write_json,
)


def run_smoke(
    *,
    batch_sizes: Iterable[int] = (2, 4),
    sequence_length: int = 2048,
    boundaries: int = 2,
    accumulation_steps: int = 1,
    target_gpu: str = "NVIDIA GeForce RTX 4090",
    target_memory_bytes: int = 24 * 1024**3,
) -> dict[str, Any]:
    import torch

    config = load_config()
    sizes = tuple(int(value) for value in batch_sizes)
    if not sizes or any(value <= 0 for value in sizes):
        raise ValueError("Smoke batch sizes must be positive.")
    if sequence_length <= 1 or sequence_length > int(mapping(config, "data")["sequence_length"]):
        raise ValueError("Smoke sequence length must be in [2, 2048].")
    if boundaries <= 0:
        raise ValueError("Smoke boundary count must be positive.")
    if accumulation_steps <= 0:
        raise ValueError("Smoke accumulation steps must be positive.")
    if not target_gpu:
        raise ValueError("Target GPU name must be non-empty.")
    if target_memory_bytes <= 0:
        raise ValueError("Target GPU memory must be positive.")

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the prelaunch smoke.")
    flash_probe = getattr(torch.backends.cuda, "is_flash_attention_available", None)
    flash_available = bool(flash_probe is not None and flash_probe())
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "non_evidence_prelaunch_smoke",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "flash_attention_available": flash_available,
        },
        "target": {
            "gpu": target_gpu,
            "memory_bytes": int(target_memory_bytes),
            "micro_batch_size": int(mapping(config, "training")["micro_batch_size"]),
            "sequence_length": int(mapping(config, "data")["sequence_length"]),
            "gradient_accumulation_steps": int(mapping(config, "training")["gradient_accumulation_steps"]),
        },
        "probe": {
            "batch_sizes": list(sizes),
            "sequence_length": int(sequence_length),
            "boundaries_per_sample": int(boundaries),
            "accumulation_steps": int(accumulation_steps),
            "note": "Each sampled boundary repeats one resident input tensor; this represents the graph, component-gradient, optimizer-state, and accumulation path without consuming scientific cache blocks.",
        },
        "samples": [],
    }
    if not flash_available:
        result.update(
            status="blocked_local_environment",
            blocker="The local PyTorch CUDA build has no flash-attention kernel; math-attention memory/timing would not represent the approved recipe.",
            projection={"status": "unavailable", "reason": "flash_attention_unavailable"},
        )
        return _publish(result)
    require_cuda(torch)
    for batch_size in sizes:
        for condition_id in ("relu-control", "relu-l1n-1"):
            try:
                sample = _sample(
                    config=config,
                    condition_id=condition_id,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    boundaries=boundaries,
                    accumulation_steps=accumulation_steps,
                    torch=torch,
                    device=device,
                )
            except torch.cuda.OutOfMemoryError as error:
                sample = {
                    "condition_id": condition_id,
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "status": "cuda_out_of_memory",
                    "error": str(error),
                }
                gc.collect()
                torch.cuda.empty_cache()
            result["samples"].append(sample)
    result["projection"] = _project_target(result["samples"], result["target"])
    result["exact_target"] = _exact_target(result["samples"], result["target"], sequence_length, accumulation_steps)
    result["status"] = "completed"
    return _publish(result)


def _publish(result: dict[str, Any]) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = RUN_DIR / "prelaunch" / f"smoke-{stamp}.json"
    result["output"] = output.relative_to(RUN_DIR).as_posix()
    write_json(output, result)
    return result


def _sample(
    *,
    config: dict[str, Any],
    condition_id: str,
    batch_size: int,
    sequence_length: int,
    boundaries: int,
    accumulation_steps: int,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM

    condition = next(row for row in _conditions(config) if row["id"] == condition_id)
    resolved = resolved_condition_config(config, condition)
    seed = int(mapping(config, "seeds")["model"])
    seed_everything(torch, seed)
    model = build_random_pythia(
        dict(mapping(resolved, "model")),
        device=device,
        torch=torch,
        auto_config=AutoConfig,
        auto_model=AutoModelForCausalLM,
    )
    seed_everything(torch, seed)
    initialization = apply_pythia_14m_initialization(model, torch=torch)
    recipe = verify_recipe_model(model)
    initial_hash = parameter_sha256(model)
    optimizer, optimizer_mapping = build_recipe_adamw(
        model, dict(mapping(config, "training")), torch=torch
    )
    pressure = parse_pressure_config(dict(mapping(resolved, "activation_pressure")))
    training = mapping(config, "training")
    scaler = DynamicLossScaler(
        scale=2.0 ** int(training["initial_loss_scale_power"]),
        growth_interval=int(training["loss_scale_window"]),
        hysteresis=int(training["loss_scale_hysteresis"]),
        minimum_scale=float(training["minimum_loss_scale"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(19)
    batch = torch.randint(
        0,
        int(model.config.vocab_size),
        (batch_size, sequence_length),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    capture_context = ActivationCapture(model, ["h"], torch=torch) if pressure.enabled else nullcontext(None)
    wall_seconds = []
    results = []
    torch.cuda.reset_peak_memory_stats()
    with capture_context as capture:
        for _ in range(boundaries):
            model.train()
            torch.cuda.synchronize()
            started = perf_counter()
            results.append(
                run_recipe_boundary(
                    model=model,
                    optimizer=optimizer,
                    batches=[batch] * accumulation_steps,
                    pressure=pressure,
                    capture=capture,
                    loss_scaler=scaler,
                    gradient_clip_norm=float(training["gradient_clip_norm"]),
                    torch=torch,
                    device=device,
                )
            )
            torch.cuda.synchronize()
            wall_seconds.append(perf_counter() - started)
    sample = {
        "condition_id": condition_id,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "gradient_accumulation_steps": accumulation_steps,
        "input_tokens_per_boundary": batch_size * sequence_length * accumulation_steps,
        "status": "completed",
        "initial_parameter_sha256": initial_hash,
        "topology": topology_metadata(model),
        "recipe": recipe,
        "initialization": initialization,
        "optimizer": optimizer_mapping,
        "boundary_seconds": wall_seconds,
        "boundary_health": [
            {
                "optimizer_step_skipped": bool(row["optimizer_step_skipped"]),
                "gradient_overflow": bool(row["gradient_overflow"]),
            }
            for row in results
        ],
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "last_boundary": results[-1],
    }
    del batch, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return sample


def _conditions(config: dict[str, Any]) -> list[dict[str, Any]]:
    from run_config import condition_specs

    return condition_specs(config)


def _project_target(samples: list[dict[str, Any]], target: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "method": "two-or-more-point affine projection of peak allocated bytes by batch size, separately by condition",
        "target_headroom_fraction": 0.9,
        "conditions": {},
    }
    target_batch = int(target["micro_batch_size"])
    limit = float(target["memory_bytes"]) * float(output["target_headroom_fraction"])
    for condition_id in ("relu-control", "relu-l1n-1"):
        rows = [
            row for row in samples
            if row.get("condition_id") == condition_id and row.get("status") == "completed"
        ]
        pairs = sorted(
            (float(row["batch_size"]), float(row["peak_memory_allocated_bytes"]))
            for row in rows
        )
        if len(pairs) < 2 or len({x for x, _ in pairs}) < 2:
            output["conditions"][condition_id] = {"status": "insufficient_samples"}
            continue
        count = float(len(pairs))
        mean_x = sum(x for x, _ in pairs) / count
        mean_y = sum(y for _, y in pairs) / count
        denominator = sum((x - mean_x) ** 2 for x, _ in pairs)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in pairs) / denominator
        intercept = mean_y - slope * mean_x
        projected = max(0.0, intercept + slope * target_batch)
        output["conditions"][condition_id] = {
            "status": "projected",
            "intercept_bytes": int(intercept),
            "bytes_per_sequence": int(slope),
            "target_peak_allocated_bytes": int(projected),
            "target_peak_allocated_gib": projected / 1024**3,
            "target_90pct_limit_gib": limit / 1024**3,
            "fits_with_10pct_headroom": bool(math.isfinite(projected) and projected <= limit),
            "caveat": "Projection is not a substitute for the exact target-GPU preflight.",
        }
    return output


def _exact_target(
    samples: list[dict[str, Any]],
    target: dict[str, Any],
    sequence_length: int,
    accumulation_steps: int,
) -> dict[str, Any]:
    expected = {"relu-control", "relu-l1n-1"}
    rows = [
        row for row in samples
        if row.get("condition_id") in expected
        and int(row.get("batch_size", -1)) == int(target["micro_batch_size"])
        and int(row.get("sequence_length", -1)) == int(target["sequence_length"])
        and int(row.get("gradient_accumulation_steps", -1)) == int(target["gradient_accumulation_steps"])
    ]
    if sequence_length != int(target["sequence_length"]) or accumulation_steps != int(target["gradient_accumulation_steps"]):
        return {"status": "not_sampled"}
    by_condition = {str(row["condition_id"]): row for row in rows}
    if set(by_condition) != expected:
        return {"status": "incomplete", "sampled_conditions": sorted(by_condition)}
    memory_limit = 0.9 * float(target["memory_bytes"])
    conditions = {}
    for name, row in by_condition.items():
        completed = row.get("status") == "completed"
        healthy = completed and all(
            not bool(item.get("optimizer_step_skipped")) and not bool(item.get("gradient_overflow"))
            for item in row.get("boundary_health", [])
        ) and bool(row.get("boundary_health"))
        reserved = float(row.get("peak_memory_reserved_bytes", math.inf))
        conditions[name] = {
            "status": row.get("status"),
            "peak_memory_reserved_gib": None if not math.isfinite(reserved) else reserved / 1024**3,
            "finite_nonoverflowing_boundaries": healthy,
            "fits_with_10pct_headroom": bool(healthy and reserved <= memory_limit),
            "boundary_seconds": row.get("boundary_seconds", []),
        }
    return {
        "status": "sampled",
        "memory_limit_gib": memory_limit / 1024**3,
        "conditions": conditions,
        "all_conditions_fit": all(row["fits_with_10pct_headroom"] for row in conditions.values()),
    }
