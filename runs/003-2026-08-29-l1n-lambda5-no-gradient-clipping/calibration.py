"""Production-shaped smoke timing for the approved unclipped cohort."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
from pathlib import Path
from statistics import median
import tempfile
from time import perf_counter
from typing import Any, Mapping

from sparsity_research.artifacts import build_transfer_inventory, git_identity
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
from sparsity_research.pythia import (
    build_random_pythia,
    load_checkpoint_pythia,
    topology_metadata,
)

from run_config import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    RUN_DIR,
    baseline_identity,
    build_schedule,
    condition_specs,
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
SAMPLE_BOUNDARIES = 8
EXCLUDED_WARMUP_BOUNDARIES = 2


def calibrate(config_path: str | Path = DEFAULT_CONFIG) -> Path:
    """Measure both exact condition paths without creating evidence attempts."""

    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = load_config(config_path)
    require_cuda(torch)
    train_tokens, validation_tokens, train_meta, _, cache_seconds = load_verified_caches(
        config, np=np
    )
    starts, schedule_hash, schedule_metadata = build_schedule(config, train_meta, np=np)
    samples = {}
    initial_hashes = set()
    peak_allocated = peak_reserved = 0
    for condition in condition_specs(config):
        sample = _measure_condition(
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
        samples[condition["activation"]] = sample
        initial_hashes.add(sample["initial_parameter_sha256"])
        peak_allocated = max(peak_allocated, sample["peak_gpu_memory_allocated_bytes"])
        peak_reserved = max(peak_reserved, sample["peak_gpu_memory_reserved_bytes"])
    if len(initial_hashes) != 1:
        raise RuntimeError("Calibration conditions did not share initialization.")

    calibration = {
        "schema_version": 1,
        "kind": "non_evidence_prelaunch_calibration",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repository_identity": git_identity(REPO_ROOT),
        "comparison_baseline": baseline_identity(config),
        "device": {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
        "cache_verification_seconds": cache_seconds,
        "terminal_headroom_seconds": TERMINAL_HEADROOM_SECONDS,
        "condition_count": 2,
        "samples": samples,
        "training_schedule_hash": schedule_hash,
        "training_schedule": schedule_metadata,
        "initial_parameter_sha256": next(iter(initial_hashes)),
        "common_steps": int(mapping(config, "training")["max_steps"]),
    }
    calibration["etc"] = estimate_cohort(calibration)
    target = float(mapping(config, "training")["target_cohort_seconds"])
    calibration["fits_target"] = float(calibration["etc"]["p90_seconds"]) <= target
    output = RUN_DIR / "prelaunch" / (
        "calibration-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + ".json"
    )
    write_json(output, calibration)
    return output


def _measure_condition(
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
    realized_topology = topology_metadata(model)
    initial_hash = parameter_sha256(model)
    optimizer = build_adamw(model, dict(training), torch=torch)
    setup_seconds = perf_counter() - setup_started
    pressure = parse_pressure_config(dict(mapping(resolved, "activation_pressure")))
    max_steps = int(training["max_steps"])
    warmup = warmup_steps(max_steps, float(training["warmup_fraction"]))
    step_seconds = []
    boundary_results = []

    with ActivationCapture(model, ["h"], torch=torch) as capture:
        for step in range(1, SAMPLE_BOUNDARIES + 1):
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
                gradient_clip_norm=None,
            )
            torch.cuda.synchronize()
            elapsed = perf_counter() - started
            if result["adamw_gradient_clipping_enabled"] is not False:
                raise RuntimeError("Calibration unexpectedly enabled gradient clipping.")
            boundary_results.append(result)
            if step > EXCLUDED_WARMUP_BOUNDARIES:
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
    if float(ordinary_validation["loss"]) != float(diagnostic_validation["loss"]):
        raise RuntimeError("Repeated no-update validation losses differ.")
    weight_started = perf_counter()
    weights = weight_statistics(model)
    weight_seconds = perf_counter() - weight_started
    trained_hash = parameter_sha256(model)

    temporary_root = RUN_DIR / "prelaunch" / ".calibration_tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="condition-", dir=temporary_root) as temporary:
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
        reloaded.config.use_cache = False
        reloaded.to(device=torch.device("cuda"), dtype=torch.float32)
        if topology_metadata(reloaded) != realized_topology:
            raise RuntimeError("Calibration checkpoint topology changed on reload.")
        if parameter_sha256(reloaded) != trained_hash:
            raise RuntimeError("Calibration checkpoint parameters changed on reload.")
        torch.cuda.synchronize()
        checkpoint_seconds = perf_counter() - checkpoint_started
        del reloaded
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "condition": dict(condition),
        "warmup_boundaries_excluded": EXCLUDED_WARMUP_BOUNDARIES,
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
        "attention_output_boundary_equivalence": diagnostics[
            "attention_output_boundary_equivalence"
        ],
        "weight_parameter_rows": len(weights),
    }


def estimate_cohort(calibration: Mapping[str, Any]) -> dict[str, float | int]:
    steps = int(calibration["common_steps"])
    median_fixed = _fixed_seconds(calibration)
    p90_fixed = median_fixed
    median_per_step = sum(
        median(sample["optimizer_step_seconds"])
        for sample in calibration["samples"].values()
    )
    p90_per_step = sum(
        percentile(sample["optimizer_step_seconds"], 0.90)
        for sample in calibration["samples"].values()
    )
    return {
        "common_steps": steps,
        "total_optimizer_steps": steps * int(calibration["condition_count"]),
        "median_fixed_seconds": median_fixed,
        "p90_fixed_seconds": p90_fixed,
        "median_seconds": median_fixed + steps * median_per_step,
        "p90_seconds": p90_fixed + steps * p90_per_step,
    }


def _fixed_seconds(calibration: Mapping[str, Any]) -> float:
    total = float(calibration["cache_verification_seconds"]) + float(
        calibration["terminal_headroom_seconds"]
    )
    for sample in calibration["samples"].values():
        total += sum(
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
