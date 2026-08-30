"""Production-shaped timing and transparent two-hour cohort budgeting."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import math
from pathlib import Path
from statistics import median
import tempfile
from time import perf_counter
from typing import Any, Mapping

from sparsity_research.artifacts import build_transfer_inventory
from sparsity_research.capture import ActivationCapture
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

from run_config import (
    DEFAULT_CONFIG,
    RUN_DIR,
    build_schedule,
    condition_specs,
    git_identity,
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
from training import timed_diagnostic_validation, timed_validation


TERMINAL_HEADROOM_SECONDS = 30.0


def calibrate(config_path: str | Path = DEFAULT_CONFIG) -> Path:
    """Measure controls and pressured paths without creating evidence attempts."""

    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = load_config(config_path)
    require_cuda(torch)
    repository_identity = git_identity()
    train_tokens, validation_tokens, train_meta, _, cache_seconds = load_verified_caches(
        config, np=np
    )
    starts, schedule_hash, schedule_metadata = build_schedule(config, train_meta, np=np)
    representatives = _representative_conditions(config)
    samples: dict[str, dict[str, Any]] = {}
    initial_hashes = set()
    global_peak_allocated = 0
    global_peak_reserved = 0

    for class_name, condition in representatives.items():
        sample = _measure_condition_class(
            config=config,
            condition=condition,
            train_tokens=train_tokens,
            validation_tokens=validation_tokens,
            starts=starts,
            torch=torch,
            np=np,
            auto_config=AutoConfig,
            auto_model=AutoModelForCausalLM,
        )
        samples[class_name] = sample
        initial_hashes.add(sample["initial_parameter_sha256"])
        global_peak_allocated = max(
            global_peak_allocated, sample["peak_gpu_memory_allocated_bytes"]
        )
        global_peak_reserved = max(
            global_peak_reserved, sample["peak_gpu_memory_reserved_bytes"]
        )

    if len(initial_hashes) != 1:
        raise RuntimeError("Calibration representatives did not share initialization.")
    calibration = {
        "schema_version": 1,
        "kind": "non_evidence_prelaunch_calibration",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repository_identity": repository_identity,
        "device": {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "peak_memory_allocated_bytes": global_peak_allocated,
            "peak_memory_reserved_bytes": global_peak_reserved,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
        "cache_verification_seconds": cache_seconds,
        "condition_counts": _condition_class_counts(config),
        "samples": samples,
        "terminal_headroom_seconds": TERMINAL_HEADROOM_SECONDS,
        "training_schedule_hash": schedule_hash,
        "training_schedule": schedule_metadata,
        "initial_parameter_sha256": next(iter(initial_hashes)),
    }
    recommended = budget_steps_from_calibration(
        calibration,
        planning_seconds=float(mapping(config, "training")["planning_cohort_seconds"]),
    )
    calibration["recommended_common_steps"] = recommended
    calibration["recommended_etc"] = estimate_cohort(
        calibration, common_steps=recommended
    )
    output = RUN_DIR / "prelaunch" / (
        "calibration-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + ".json"
    )
    write_json(output, calibration)
    return output


def _measure_condition_class(
    *,
    config: Mapping[str, Any],
    condition: Mapping[str, Any],
    train_tokens: Any,
    validation_tokens: Any,
    starts: Any,
    torch: Any,
    np: Any,
    auto_config: Any,
    auto_model: Any,
) -> dict[str, Any]:
    resolved = resolved_condition_config(config, condition)
    training = mapping(config, "training")
    seed_everything(torch, int(mapping(config, "seeds")["model"]))
    torch.cuda.reset_peak_memory_stats()
    setup_started = perf_counter()
    model = build_random_pythia(
        dict(mapping(resolved, "model")),
        device=torch.device("cuda"),
        torch=torch,
        auto_config=auto_config,
        auto_model=auto_model,
    )
    model.config.use_cache = False
    initial_hash = parameter_sha256(model)
    optimizer = build_adamw(model, dict(training), torch=torch)
    setup_seconds = perf_counter() - setup_started
    pressure = parse_pressure_config(dict(mapping(resolved, "activation_pressure")))
    max_steps = int(training["max_steps"])
    warmup = warmup_steps(max_steps, float(training["warmup_fraction"]))
    sample_steps = min(max_steps, 8)
    if sample_steps < 6:
        raise ValueError("Calibration requires at least six provisional steps.")
    step_seconds = []
    boundary_results = []
    capture_context = (
        ActivationCapture(model, ["h"], torch=torch)
        if pressure.enabled
        else nullcontext(None)
    )
    with capture_context as capture:
        for step in range(1, sample_steps + 1):
            set_learning_rate(
                optimizer,
                learning_rate(
                    step,
                    peak=float(training["peak_learning_rate"]),
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
            result = run_optimizer_boundary(
                model=model,
                optimizer=optimizer,
                batches=batches,
                pressure=pressure,
                capture=capture,
                torch=torch,
                device=torch.device("cuda"),
                autocast_dtype=torch.bfloat16,
            )
            torch.cuda.synchronize()
            elapsed = perf_counter() - started
            boundary_results.append(result)
            if step > 2:
                step_seconds.append(elapsed)

    ordinary_validation, ordinary_seconds = timed_validation(
        model=model,
        tokens=validation_tokens,
        config=config,
        torch=torch,
        np=np,
    )
    diagnostic_validation, diagnostics, diagnostic_seconds = timed_diagnostic_validation(
        model=model,
        tokens=validation_tokens,
        config=config,
        torch=torch,
        np=np,
    )
    if ordinary_validation["loss"] != diagnostic_validation["loss"]:
        raise RuntimeError("Repeated no-update validation losses differ.")
    weight_started = perf_counter()
    weights = weight_statistics(model)
    weight_seconds = perf_counter() - weight_started
    with tempfile.TemporaryDirectory(prefix="sparsity-run002-calibration-") as temporary:
        checkpoint_dir = Path(temporary) / "final"
        checkpoint_started = perf_counter()
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
        inventory = build_transfer_inventory(checkpoint_dir)
        inventory_content_sha256(inventory)
        del optimizer
        del model
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = load_checkpoint_pythia(auto_model, checkpoint_dir, torch=torch)
        reloaded.to(device=torch.device("cuda"), dtype=torch.float32)
        torch.cuda.synchronize()
        checkpoint_seconds = perf_counter() - checkpoint_started
        del reloaded
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "condition": dict(condition),
        "warmup_boundaries_excluded": 2,
        "timed_optimizer_boundaries": len(step_seconds),
        "optimizer_step_seconds": step_seconds,
        "model_setup_seconds": setup_seconds,
        "ordinary_full_validation_seconds": ordinary_seconds,
        "diagnostic_full_validation_seconds": diagnostic_seconds,
        "weight_statistics_seconds": weight_seconds,
        "checkpoint_save_hash_reload_seconds": checkpoint_seconds,
        "peak_gpu_memory_allocated_bytes": peak_allocated,
        "peak_gpu_memory_reserved_bytes": peak_reserved,
        "initial_parameter_sha256": initial_hash,
        "last_boundary": boundary_results[-1],
        "validation": diagnostic_validation,
        "diagnostic_layer_rows": len(diagnostics["rows"]),
        "diagnostic_pooled_rows": len(diagnostics["pooled_by_site"]),
        "weight_parameter_rows": len(weights),
    }


def budget_steps_from_calibration(
    calibration: Mapping[str, Any], *, planning_seconds: float
) -> int:
    fixed = _fixed_seconds(calibration, upper=True)
    coefficient = _per_common_step_seconds(calibration, upper=True)
    steps = math.floor((float(planning_seconds) - fixed) / coefficient)
    if steps <= 0:
        raise ValueError("Measured fixed work leaves no optimizer-step budget.")
    return steps


def estimate_cohort(
    calibration: Mapping[str, Any], *, common_steps: int
) -> dict[str, float | int]:
    median_fixed = _fixed_seconds(calibration, upper=False)
    upper_fixed = _fixed_seconds(calibration, upper=True)
    median_step = _per_common_step_seconds(calibration, upper=False)
    upper_step = _per_common_step_seconds(calibration, upper=True)
    return {
        "common_steps": int(common_steps),
        "total_optimizer_steps": int(common_steps)
        * sum(int(value) for value in calibration["condition_counts"].values()),
        "median_fixed_seconds": median_fixed,
        "p90_fixed_seconds": upper_fixed,
        "median_seconds": median_fixed + int(common_steps) * median_step,
        "p90_seconds": upper_fixed + int(common_steps) * upper_step,
    }


def _fixed_seconds(calibration: Mapping[str, Any], *, upper: bool) -> float:
    total = float(calibration["cache_verification_seconds"]) + float(
        calibration["terminal_headroom_seconds"]
    )
    for class_name, count in calibration["condition_counts"].items():
        sample = calibration["samples"][class_name]
        total += int(count) * sum(
            float(sample[key])
            for key in (
                "model_setup_seconds",
                "ordinary_full_validation_seconds",
                "diagnostic_full_validation_seconds",
                "weight_statistics_seconds",
                "checkpoint_save_hash_reload_seconds",
            )
        )
    return total


def _per_common_step_seconds(calibration: Mapping[str, Any], *, upper: bool) -> float:
    total = 0.0
    for class_name, count in calibration["condition_counts"].items():
        values = calibration["samples"][class_name]["optimizer_step_seconds"]
        representative = percentile(values, 0.90) if upper else median(values)
        total += int(count) * representative
    return total


def _representative_conditions(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    specs = condition_specs(config)
    selected = {}
    for activation in ("gelu", "relu"):
        selected[f"{activation}_control"] = next(
            row for row in specs if row["activation"] == activation and row["is_control"]
        )
        selected[f"{activation}_pressure"] = next(
            row
            for row in specs
            if row["activation"] == activation and row["pressure_weight"] == 5.0
        )
    return selected


def _condition_class_counts(config: Mapping[str, Any]) -> dict[str, int]:
    counts = {
        "gelu_control": 0,
        "relu_control": 0,
        "gelu_pressure": 0,
        "relu_pressure": 0,
    }
    for row in condition_specs(config):
        suffix = "control" if row["is_control"] else "pressure"
        counts[f"{row['activation']}_{suffix}"] += 1
    return counts
