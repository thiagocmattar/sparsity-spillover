"""Production-shaped non-evidence calibration and transparent Run 005 ETC."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import math
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Mapping

from sparsity_research.artifacts import build_transfer_inventory
from sparsity_research.capture import ActivationCapture
from sparsity_research.metrics import pool_weight_norm, weight_statistics
from sparsity_research.optimization import (
    build_adamw,
    learning_rate,
    run_optimizer_boundary,
    set_learning_rate,
    warmup_steps,
)
from sparsity_research.pressure import parse_pressure_config
from sparsity_research.pythia import build_random_pythia, load_checkpoint_pythia, topology_metadata

from diagnostics import activation_diagnostic_validation, logical_product_validation
from run_config import (
    REPO_ROOT,
    RUN_DIR,
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
from training import timed_validation


CALIBRATION_STEPS = 8
WARMUP_SAMPLES_EXCLUDED = 2
TERMINAL_HEADROOM_SECONDS = 60.0


def calibrate() -> Path:
    import numpy as np
    import torch

    config = load_config()
    require_cuda(torch)
    train_tokens, validation_tokens, train_meta, _validation_meta, cache_seconds = (
        load_verified_caches(config, np=np)
    )
    starts, schedule_hash, schedule_metadata = build_schedule(config, train_meta, np=np)
    conditions = condition_specs(config)
    selected = {
        "control": conditions[0],
        "ol1_pressure": conditions[-1],
    }
    samples = {}
    initial_hashes = set()
    for name, condition in selected.items():
        sample = calibrate_condition(
            config=config,
            condition=condition,
            train_tokens=train_tokens,
            validation_tokens=validation_tokens,
            starts=starts,
            torch=torch,
            np=np,
        )
        samples[name] = sample
        initial_hashes.add(sample["initial_parameter_sha256"])
        gc.collect()
        torch.cuda.empty_cache()
    if len(initial_hashes) != 1:
        raise RuntimeError("Calibration conditions did not share one initialization.")

    measured = {
        "schema_version": 1,
        "kind": "non_evidence_prelaunch_calibration",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cache_verification_seconds": cache_seconds,
        "terminal_headroom_seconds": TERMINAL_HEADROOM_SECONDS,
        "condition_counts": {"control": 1, "ol1_pressure": 4},
        "samples": samples,
        "initial_parameter_sha256": next(iter(initial_hashes)),
        "training_schedule_hash": schedule_hash,
        "training_schedule": schedule_metadata,
        "device": {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "peak_memory_allocated_bytes": max(
                int(sample["peak_gpu_memory_allocated_bytes"]) for sample in samples.values()
            ),
            "peak_memory_reserved_bytes": max(
                int(sample["peak_gpu_memory_reserved_bytes"]) for sample in samples.values()
            ),
        },
    }
    planning = float(mapping(config, "training")["planning_cohort_seconds"])
    steps = budget_steps_from_calibration(measured, planning_seconds=planning)
    measured["recommended_common_steps"] = steps
    measured["recommended_etc"] = estimate_cohort(measured, common_steps=steps)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = RUN_DIR / "prelaunch" / f"calibration-{stamp}.json"
    write_json(path, measured)
    return path


def calibrate_condition(
    *,
    config: Mapping[str, Any],
    condition: Mapping[str, Any],
    train_tokens: Any,
    validation_tokens: Any,
    starts: Any,
    torch: Any,
    np: Any,
) -> dict[str, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM

    resolved = resolved_condition_config(config, condition)
    seed_everything(torch, int(mapping(config, "seeds")["model"]))
    torch.cuda.reset_peak_memory_stats()
    setup_started = perf_counter()
    model = build_random_pythia(
        dict(mapping(resolved, "model")),
        device=torch.device("cuda"),
        torch=torch,
        auto_config=AutoConfig,
        auto_model=AutoModelForCausalLM,
    )
    model.config.use_cache = False
    realized_topology = topology_metadata(model)
    initial_hash = parameter_sha256(model)
    training = mapping(config, "training")
    optimizer = build_adamw(model, dict(training), torch=torch)
    setup_seconds = perf_counter() - setup_started
    pressure = parse_pressure_config(dict(mapping(resolved, "activation_pressure")))
    max_steps = int(training["max_steps"])
    warmup = warmup_steps(max_steps, float(training["warmup_fraction"]))
    step_samples = []
    final_boundary = None
    capture = ActivationCapture(model, ["h"], torch=torch) if pressure.enabled else None
    context = capture if capture is not None else _NullContext()
    with context:
        for step in range(1, CALIBRATION_STEPS + 1):
            effective_lr = learning_rate(
                step,
                peak=float(training["peak_learning_rate"]),
                max_steps=max_steps,
                warmup=warmup,
                minimum_ratio=float(training["minimum_learning_rate_ratio"]),
            )
            set_learning_rate(optimizer, effective_lr)
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
            final_boundary = run_optimizer_boundary(
                model=model,
                optimizer=optimizer,
                batches=batches,
                pressure=pressure,
                capture=capture,
                torch=torch,
                device=torch.device("cuda"),
                autocast_dtype=torch.bfloat16,
                gradient_clip_norm=float(training["gradient_clip_norm"]),
            )
            torch.cuda.synchronize()
            elapsed = perf_counter() - started
            if step > WARMUP_SAMPLES_EXCLUDED:
                step_samples.append(elapsed)
    if final_boundary is None or len(step_samples) != CALIBRATION_STEPS - WARMUP_SAMPLES_EXCLUDED:
        raise RuntimeError("Calibration did not produce the requested optimizer timing samples.")

    validation, ordinary_validation_seconds = timed_validation(
        model=model,
        tokens=validation_tokens,
        config=config,
        torch=torch,
        np=np,
    )

    temp_root = REPO_ROOT / "tmp"
    temp_root.mkdir(exist_ok=True)
    checkpoint_started = perf_counter()
    with TemporaryDirectory(prefix="run005-calibration-", dir=temp_root) as temporary:
        checkpoint = Path(temporary) / "checkpoint"
        model.save_pretrained(checkpoint, safe_serialization=True)
        inventory = build_transfer_inventory(checkpoint)
        checkpoint_hash = inventory_content_sha256(inventory)
        del optimizer
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = load_checkpoint_pythia(AutoModelForCausalLM, checkpoint, torch=torch)
        model.config.use_cache = False
        model.to(device=torch.device("cuda"), dtype=torch.float32)
        torch.cuda.synchronize()
        checkpoint_seconds = perf_counter() - checkpoint_started
        if topology_metadata(model) != realized_topology:
            raise RuntimeError("Calibration checkpoint changed the realized topology.")

        torch.cuda.synchronize()
        activation_started = perf_counter()
        activation_coverage, activation_statistics = activation_diagnostic_validation(
            model=model,
            tokens=validation_tokens,
            config=config,
            torch=torch,
            np=np,
        )
        torch.cuda.synchronize()
        activation_seconds = perf_counter() - activation_started

        torch.cuda.synchronize()
        logical_started = perf_counter()
        logical_coverage, logical_products, architecture_maximum = logical_product_validation(
            model=model,
            tokens=validation_tokens,
            config=config,
            torch=torch,
            np=np,
        )
        torch.cuda.synchronize()
        logical_seconds = perf_counter() - logical_started

        weight_started = perf_counter()
        weights = weight_statistics(model)
        pooled = pool_weight_norm(weights)
        weight_seconds = perf_counter() - weight_started

    if not math.isfinite(float(validation["loss"])):
        raise RuntimeError("Calibration validation loss is non-finite.")
    if len(activation_statistics["rows"]) != 36 or len(activation_statistics["pooled_by_site"]) != 6:
        raise RuntimeError("Calibration activation diagnostics are incomplete.")
    if architecture_maximum.get("topology_id") != "A1-H":
        raise RuntimeError("Calibration architecture ceiling uses the wrong topology.")
    if not 0.0 <= float(logical_products["R_model"]) <= 1.0:
        raise RuntimeError("Calibration logical-product summary is invalid.")
    if int(pooled["parameter_tensors"]) != len(weights):
        raise RuntimeError("Calibration weight pooling is inconsistent.")

    return {
        "condition": dict(condition),
        "model_setup_seconds": setup_seconds,
        "optimizer_step_seconds": step_samples,
        "timed_optimizer_boundaries": len(step_samples),
        "warmup_boundaries_excluded": WARMUP_SAMPLES_EXCLUDED,
        "ordinary_full_validation_seconds": ordinary_validation_seconds,
        "activation_diagnostic_seconds": activation_seconds,
        "logical_product_diagnostic_seconds": logical_seconds,
        "weight_statistics_seconds": weight_seconds,
        "checkpoint_save_hash_reload_seconds": checkpoint_seconds,
        "checkpoint_bytes": inventory["total_bytes"],
        "checkpoint_content_sha256": checkpoint_hash,
        "initial_parameter_sha256": initial_hash,
        "final_boundary": final_boundary,
        "validation": validation,
        "activation_coverage": activation_coverage,
        "logical_product_coverage": logical_coverage,
        "logical_products": {
            "R_block": logical_products["R_block"],
            "R_model": logical_products["R_model"],
            "R_model_max": architecture_maximum["R_model_max_fraction"],
        },
        "activation_layer_rows": len(activation_statistics["rows"]),
        "activation_pooled_rows": len(activation_statistics["pooled_by_site"]),
        "weight_parameter_rows": len(weights),
        "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def estimate_cohort(measured: Mapping[str, Any], *, common_steps: int) -> dict[str, Any]:
    if isinstance(common_steps, bool) or not isinstance(common_steps, int) or common_steps <= 0:
        raise ValueError("Common steps must be a positive integer.")
    median_fixed = float(measured["cache_verification_seconds"]) + float(
        measured["terminal_headroom_seconds"]
    )
    p90_fixed = median_fixed
    median_step_total = p90_step_total = 0.0
    for name, count in measured["condition_counts"].items():
        sample = measured["samples"][name]
        fixed = sum(
            float(sample[key])
            for key in (
                "model_setup_seconds",
                "ordinary_full_validation_seconds",
                "activation_diagnostic_seconds",
                "logical_product_diagnostic_seconds",
                "weight_statistics_seconds",
                "checkpoint_save_hash_reload_seconds",
            )
        )
        median_fixed += int(count) * fixed
        p90_fixed += int(count) * fixed
        timings = sample["optimizer_step_seconds"]
        median_step_total += int(count) * median(float(value) for value in timings)
        p90_step_total += int(count) * percentile(timings, 0.9)
    return {
        "common_steps": common_steps,
        "condition_count": sum(int(value) for value in measured["condition_counts"].values()),
        "total_optimizer_steps": common_steps
        * sum(int(value) for value in measured["condition_counts"].values()),
        "median_fixed_seconds": median_fixed,
        "p90_fixed_seconds": p90_fixed,
        "median_seconds": median_fixed + common_steps * median_step_total,
        "p90_seconds": p90_fixed + common_steps * p90_step_total,
    }


def budget_steps_from_calibration(
    measured: Mapping[str, Any], *, planning_seconds: float
) -> int:
    if not math.isfinite(planning_seconds) or planning_seconds <= 0.0:
        raise ValueError("Planning seconds must be positive and finite.")
    fixed = estimate_cohort(measured, common_steps=1)["p90_fixed_seconds"]
    one = estimate_cohort(measured, common_steps=1)["p90_seconds"] - fixed
    steps = math.floor((planning_seconds - fixed) / one)
    if steps <= 0:
        raise RuntimeError("The fixed diagnostic bundle does not fit the planning envelope.")
    if estimate_cohort(measured, common_steps=steps)["p90_seconds"] > planning_seconds:
        raise RuntimeError("ETC arithmetic failed to honor the planning envelope.")
    return steps


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: Any) -> None:
        return None
