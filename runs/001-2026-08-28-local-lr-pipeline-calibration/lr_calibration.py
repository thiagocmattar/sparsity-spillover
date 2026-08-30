"""Production-shaped non-evidence timing and transparent cohort ETC arithmetic."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import math
from pathlib import Path
from statistics import median
import tempfile
from time import perf_counter
from typing import Any, Mapping

from sparsity_research.artifacts import build_transfer_inventory
from sparsity_research.metrics import weight_statistics
from sparsity_research.optimization import (
    build_adamw,
    learning_rate,
    run_optimizer_boundary,
    set_learning_rate,
    warmup_steps,
)
from sparsity_research.pressure import parse_pressure_config
from sparsity_research.pythia import build_random_pythia, load_checkpoint_pythia

from lr_run_config import (
    DEFAULT_CONFIG,
    RUN_DIR,
    build_schedule,
    cache_identity,
    inventory_content_sha256,
    load_config,
    load_verified_caches,
    mapping,
    microbatches_for_step,
    parameter_sha256,
    percentile,
    require_cuda,
    resolved_condition_config,
    seed_everything,
    write_json,
)
from lr_training import timed_diagnostic_validation, timed_validation


def calibrate(config_path: str | Path = DEFAULT_CONFIG) -> Path:
    """Measure the exact workload without creating a scientific attempt."""

    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = load_config(config_path)
    require_cuda(torch)
    train_tokens, validation_tokens, train_meta, validation_meta, cache_seconds = (
        load_verified_caches(config, np=np)
    )
    starts, schedule_hash, schedule_metadata = build_schedule(config, train_meta, np=np)
    center_lr = float(mapping(config, "conditions")["peak_learning_rates"][1])
    resolved = resolved_condition_config(config, center_lr)
    training = mapping(config, "training")
    seed_everything(torch, int(mapping(config, "seeds")["model"]))
    torch.cuda.reset_peak_memory_stats()

    setup_started = perf_counter()
    model = build_random_pythia(
        dict(mapping(config, "model")),
        device=torch.device("cuda"),
        torch=torch,
        auto_config=AutoConfig,
        auto_model=AutoModelForCausalLM,
    )
    model.config.use_cache = False
    initial_hash = parameter_sha256(model)
    optimizer = build_adamw(model, dict(mapping(resolved, "training")), torch=torch)
    model_setup_seconds = perf_counter() - setup_started
    pressure = parse_pressure_config(dict(mapping(config, "activation_pressure")))
    max_steps = int(training["max_steps"])
    warmup = warmup_steps(max_steps, float(training["warmup_fraction"]))
    step_seconds: list[float] = []
    sample_steps = min(max_steps, 10)
    if sample_steps < 5:
        raise ValueError("Calibration requires at least five provisional steps.")
    for step in range(1, sample_steps + 1):
        set_learning_rate(
            optimizer,
            learning_rate(
                step,
                peak=center_lr,
                max_steps=max_steps,
                warmup=warmup,
                minimum_ratio=float(training["minimum_learning_rate_ratio"]),
            ),
        )
        model.train()
        batches = microbatches_for_step(
            train_tokens,
            starts[step - 1],
            block_size=int(mapping(config, "data")["sequence_length"]),
            device=torch.device("cuda"),
            torch=torch,
            np=np,
        )
        torch.cuda.synchronize()
        started = perf_counter()
        run_optimizer_boundary(
            model=model,
            optimizer=optimizer,
            batches=batches,
            pressure=pressure,
            capture=None,
            torch=torch,
            device=torch.device("cuda"),
            autocast_dtype=torch.bfloat16,
        )
        torch.cuda.synchronize()
        elapsed = perf_counter() - started
        if step > 2:
            step_seconds.append(elapsed)

    validation_seconds = []
    validation_results = []
    for _ in range(2):
        result, elapsed = timed_validation(
            model=model,
            tokens=validation_tokens,
            config=config,
            torch=torch,
            np=np,
        )
        validation_results.append(result)
        validation_seconds.append(elapsed)
    diagnostic_results = []
    for _ in range(2):
        result, diagnostics, elapsed = timed_diagnostic_validation(
            model=model,
            tokens=validation_tokens,
            config=config,
            torch=torch,
            np=np,
        )
        validation_results.append(result)
        diagnostic_results.append(diagnostics)
        validation_seconds.append(elapsed)

    weight_started = perf_counter()
    weights = weight_statistics(model)
    weight_seconds = perf_counter() - weight_started
    with tempfile.TemporaryDirectory(prefix="sparsity-lr-calibration-") as temporary:
        checkpoint_dir = Path(temporary) / "final"
        checkpoint_started = perf_counter()
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
        inventory = build_transfer_inventory(checkpoint_dir)
        inventory_content_sha256(inventory)
        del optimizer
        del model
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = load_checkpoint_pythia(AutoModelForCausalLM, checkpoint_dir, torch=torch)
        reloaded.to(device=torch.device("cuda"), dtype=torch.float32)
        torch.cuda.synchronize()
        checkpoint_seconds = perf_counter() - checkpoint_started
        del reloaded
    gc.collect()
    torch.cuda.empty_cache()

    condition_count = len(mapping(config, "conditions")["peak_learning_rates"])
    calibration = {
        "schema_version": 1,
        "kind": "non_evidence_prelaunch_calibration",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
        "sample": {
            "peak_learning_rate": center_lr,
            "warmup_boundaries_excluded": 2,
            "timed_optimizer_boundaries": len(step_seconds),
            "micro_batch_size": int(training["micro_batch_size"]),
            "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
            "global_batch_size": int(training["global_batch_size"]),
            "sequence_length": int(mapping(config, "data")["sequence_length"]),
        },
        "optimizer_step_seconds": step_seconds,
        "full_validation_seconds": validation_seconds,
        "cache_verification_seconds": cache_seconds,
        "model_setup_seconds_per_condition": model_setup_seconds,
        "weight_statistics_seconds_per_condition": weight_seconds,
        "checkpoint_seconds_per_condition": checkpoint_seconds,
        "setup_seconds": cache_seconds + condition_count * model_setup_seconds,
        "diagnostics_seconds": condition_count * weight_seconds,
        "checkpoint_seconds": condition_count * checkpoint_seconds,
        "validation_passes_per_condition": 2,
        "condition_count": condition_count,
        "initial_parameter_sha256": initial_hash,
        "training_schedule_hash": schedule_hash,
        "training_schedule": schedule_metadata,
        "train_cache": cache_identity(train_meta),
        "validation_cache": cache_identity(validation_meta),
        "validation_results": validation_results,
        "diagnostic_rows_last_sample": {
            "layer_rows": len(diagnostic_results[-1]["rows"]),
            "pooled_rows": len(diagnostic_results[-1]["pooled_by_site"]),
        },
        "weight_parameter_rows": len(weights),
    }
    recommended = budget_steps_from_calibration(
        calibration,
        planning_seconds=float(training["planning_cohort_seconds"]),
    )
    calibration["recommended_common_steps"] = recommended
    calibration["recommended_etc"] = estimate_cohort(calibration, common_steps=recommended)
    output = RUN_DIR / "prelaunch" / (
        "calibration-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + ".json"
    )
    write_json(output, calibration)
    return output


def budget_steps_from_calibration(
    calibration: Mapping[str, Any], *, planning_seconds: float
) -> int:
    condition_count = int(calibration["condition_count"])
    validation_passes = condition_count * int(calibration["validation_passes_per_condition"])
    fixed = sum(
        float(calibration.get(key, 0.0))
        for key in ("setup_seconds", "diagnostics_seconds", "checkpoint_seconds")
    )
    step_upper = percentile(calibration["optimizer_step_seconds"], 0.90)
    validation_upper = percentile(calibration["full_validation_seconds"], 0.90)
    available = float(planning_seconds) - fixed - validation_passes * validation_upper
    steps = math.floor(available / (condition_count * step_upper))
    if steps <= 0:
        raise ValueError("Measured fixed work leaves no optimizer-step budget.")
    return steps


def estimate_cohort(calibration: Mapping[str, Any], *, common_steps: int) -> dict[str, float | int]:
    condition_count = int(calibration["condition_count"])
    validation_passes = condition_count * int(calibration["validation_passes_per_condition"])
    fixed = sum(
        float(calibration.get(key, 0.0))
        for key in ("setup_seconds", "diagnostics_seconds", "checkpoint_seconds")
    )
    step_median = median(float(value) for value in calibration["optimizer_step_seconds"])
    step_upper = percentile(calibration["optimizer_step_seconds"], 0.90)
    validation_median = median(
        float(value) for value in calibration["full_validation_seconds"]
    )
    validation_upper = percentile(calibration["full_validation_seconds"], 0.90)
    total_steps = condition_count * int(common_steps)
    return {
        "common_steps": int(common_steps),
        "total_optimizer_steps": total_steps,
        "validation_passes": validation_passes,
        "fixed_seconds": fixed,
        "median_seconds": fixed
        + total_steps * step_median
        + validation_passes * validation_median,
        "p90_seconds": fixed
        + total_steps * step_upper
        + validation_passes * validation_upper,
    }
