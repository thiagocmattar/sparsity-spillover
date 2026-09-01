"""Non-evidence structural and target-memory probes for Run 017 endpoints."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import math
from time import perf_counter
from typing import Any, Iterable

from sparsity_research.capture import ActivationCapture
from sparsity_research.pressure import parse_pressure_config
from sparsity_research.pythia import topology_metadata

from initialization import apply_pythia_70m_initialization, verify_recipe_model
from model_factory import build_pinned_run017_model
from optimizer_boundary import DynamicLossScaler, build_recipe_adamw, run_recipe_boundary
from run_config import (
    EXPECTED_SENTINEL,
    RUN_DIR,
    condition_specs,
    load_config,
    mapping,
    EXPECTED_INITIAL_PARAMETER_SHA256,
    parameter_sha256,
    require_cuda,
    resolved_condition_config,
    seed_everything,
    write_json,
)


def run_smoke(
    *,
    condition_ids: Iterable[str] = EXPECTED_SENTINEL,
    batch_size: int = 1,
    sequence_length: int = 128,
    boundaries: int = 1,
    accumulation_steps: int = 1,
    target_gpu: str = "NVIDIA H200",
    target_memory_bytes: int = 141 * 1024**3,
) -> dict[str, Any]:
    import torch

    config = load_config()
    selected = tuple(str(value) for value in condition_ids)
    if not selected or not set(selected).issubset(EXPECTED_SENTINEL):
        raise ValueError("Smoke conditions must be a non-empty subset of the sentinel wave.")
    if batch_size <= 0 or sequence_length <= 1 or boundaries <= 0 or accumulation_steps <= 0:
        raise ValueError("Smoke dimensions must be positive and sequence length must exceed one.")
    if sequence_length > int(mapping(config, "data")["sequence_length"]):
        raise ValueError("Smoke sequence length exceeds the scientific sequence length.")
    if target_memory_bytes <= 0 or not target_gpu:
        raise ValueError("A target GPU and positive target memory are required.")
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
            "condition_ids": list(selected),
            "batch_size": int(batch_size),
            "sequence_length": int(sequence_length),
            "boundaries": int(boundaries),
            "accumulation_steps": int(accumulation_steps),
            "timing_scope": "resident synthetic input; optimizer graph only",
            "evidence_use": "none",
        },
        "samples": [],
    }
    if not flash_available:
        result.update(status="blocked_environment", exact_target={"status": "unavailable"})
        return _publish(result)
    require_cuda(torch)
    device = torch.device("cuda")
    for condition_id in selected:
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
            sample = {"condition_id": condition_id, "status": "cuda_out_of_memory", "error": str(error)}
            gc.collect()
            torch.cuda.empty_cache()
        result["samples"].append(sample)
    result["status"] = "completed"
    result["exact_target"] = _exact_target(result)
    return _publish(result)


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
    from transformers import AutoModelForCausalLM

    condition = next(row for row in condition_specs(config) if row["id"] == condition_id)
    resolved = resolved_condition_config(config, condition)
    seed = int(mapping(config, "seeds")["model"])
    seed_everything(torch, seed)
    model = build_pinned_run017_model(
        dict(mapping(resolved, "model")),
        device=torch.device("cpu"),
        torch=torch,
        auto_model=AutoModelForCausalLM,
    )
    model.config.pressure_sites = list(condition["pressure_sites"])
    seed_everything(torch, seed)
    initialization = apply_pythia_70m_initialization(model, torch=torch)
    recipe = verify_recipe_model(model)
    initial_hash = parameter_sha256(model)
    if initial_hash != EXPECTED_INITIAL_PARAMETER_SHA256:
        raise RuntimeError(
            "Pinned portable CPU initialization mismatch in smoke: "
            f"realized={initial_hash}, expected={EXPECTED_INITIAL_PARAMETER_SHA256}"
        )
    model.to(device=device, dtype=torch.float32)
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
    capture_context = (
        ActivationCapture(model, list(pressure.sites), torch=torch)
        if pressure.enabled
        else nullcontext(None)
    )
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


def _exact_target(result: dict[str, Any]) -> dict[str, Any]:
    probe = result["probe"]
    target = result["target"]
    exact_dimensions = (
        int(probe["batch_size"]) == int(target["micro_batch_size"])
        and int(probe["sequence_length"]) == int(target["sequence_length"])
        and int(probe["accumulation_steps"]) == int(target["gradient_accumulation_steps"])
    )
    if not exact_dimensions:
        return {
            "status": "not_sampled",
            "all_conditions_fit": False,
            "reason": "probe dimensions differ from the scientific workload",
        }
    limit = 0.9 * float(target["memory_bytes"])
    conditions = {}
    for row in result["samples"]:
        health = row.get("boundary_health", [])
        healthy = (
            row.get("status") == "completed"
            and bool(health)
            and all(not item["optimizer_step_skipped"] and not item["gradient_overflow"] for item in health)
        )
        reserved = float(row.get("peak_memory_reserved_bytes", math.inf))
        conditions[row["condition_id"]] = {
            "healthy": healthy,
            "peak_memory_reserved_gib": None if not math.isfinite(reserved) else reserved / 1024**3,
            "fits_with_10pct_headroom": bool(healthy and reserved <= limit),
        }
    return {
        "status": "sampled",
        "memory_limit_gib": limit / 1024**3,
        "conditions": conditions,
        "all_conditions_fit": bool(conditions) and all(row["fits_with_10pct_headroom"] for row in conditions.values()),
        "timing_representative": False,
        "note": "Exact resident-input dimensions test memory; cache construction/staging ETC is measured separately.",
    }


def _publish(result: dict[str, Any]) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = RUN_DIR / "prelaunch" / f"smoke-{stamp}.json"
    result["output"] = output.relative_to(RUN_DIR).as_posix()
    write_json(output, result)
    return result
