"""Matched condition execution, complete validation, and artifact publication."""

from __future__ import annotations

from copy import deepcopy
import gc
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Mapping

from sparsity_research.artifacts import attempt_lifecycle, build_transfer_inventory
from sparsity_research.capture import ActivationCapture
from sparsity_research.evaluation import evaluate_complete_blocks
from sparsity_research.metrics import ActivationAccumulator, pool_weight_norm, weight_statistics
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

from lr_run_config import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    RUN_DIR,
    build_schedule,
    cache_identity,
    inventory_content_sha256,
    load_config,
    load_verified_caches,
    mapping,
    microbatches_for_step,
    parameter_sha256,
    require_cuda,
    require_validation_coverage,
    resolved_condition_config,
    run_code_identity,
    seed_everything,
    write_json,
)


def run_cohort(config_path: str | Path = DEFAULT_CONFIG) -> list[Path]:
    import numpy as np
    import torch

    config = load_config(config_path)
    require_cuda(torch)
    train_tokens, validation_tokens, train_meta, validation_meta, cache_seconds = (
        load_verified_caches(config, np=np)
    )
    starts, schedule_hash, schedule_metadata = build_schedule(config, train_meta, np=np)
    results = []
    for peak_lr in mapping(config, "conditions")["peak_learning_rates"]:
        results.append(
            run_condition(
                config=config,
                peak_lr=float(peak_lr),
                train_tokens=train_tokens,
                validation_tokens=validation_tokens,
                train_metadata=train_meta,
                validation_metadata=validation_meta,
                starts=starts,
                schedule_hash=schedule_hash,
                schedule_metadata=schedule_metadata,
                cache_verification_seconds=cache_seconds,
                torch=torch,
                np=np,
            )
        )
        cache_seconds = 0.0
        gc.collect()
        torch.cuda.empty_cache()
    return results


def run_condition(
    *,
    config: Mapping[str, Any],
    peak_lr: float,
    train_tokens: Any,
    validation_tokens: Any,
    train_metadata: Mapping[str, Any],
    validation_metadata: Mapping[str, Any],
    starts: Any,
    schedule_hash: str,
    schedule_metadata: Mapping[str, Any],
    cache_verification_seconds: float,
    torch: Any,
    np: Any,
) -> Path:
    from transformers import AutoConfig, AutoModelForCausalLM

    resolved = resolved_condition_config(config, peak_lr)
    condition_id = resolved["condition"]["id"]
    command = f"python {RUN_DIR.relative_to(REPO_ROOT).as_posix()}/02_train.py"
    extra_identity = {
        "condition": deepcopy(resolved["condition"]),
        "model": {
            **deepcopy(dict(mapping(config, "model"))),
            "loaded_checkpoint_weights": False,
        },
        "data": {
            "train": cache_identity(train_metadata),
            "validation": cache_identity(validation_metadata),
            "training_schedule_hash": schedule_hash,
            "training_schedule": dict(schedule_metadata),
        },
        "seeds": deepcopy(dict(mapping(config, "seeds"))),
        "run_code": run_code_identity(),
    }
    with attempt_lifecycle(
        RUN_DIR,
        config=resolved,
        command=command,
        mode="pretrain",
        extra_identity=extra_identity,
    ) as attempt:
        condition_started = perf_counter()
        setup_started = perf_counter()
        seed_everything(torch, int(mapping(config, "seeds")["model"]))
        torch.cuda.reset_peak_memory_stats()
        model = build_random_pythia(
            dict(mapping(config, "model")),
            device=torch.device("cuda"),
            torch=torch,
            auto_config=AutoConfig,
            auto_model=AutoModelForCausalLM,
        )
        model.config.use_cache = False
        realized_topology = topology_metadata(model)
        initial_hash = parameter_sha256(model)
        optimizer = build_adamw(model, dict(mapping(resolved, "training")), torch=torch)
        setup_seconds = perf_counter() - setup_started

        pressure = parse_pressure_config(dict(mapping(config, "activation_pressure")))
        training = mapping(config, "training")
        max_steps = int(training["max_steps"])
        warmup = warmup_steps(max_steps, float(training["warmup_fraction"]))
        step_samples: list[float] = []
        first_validation = None
        first_validation_seconds = None
        final_train = None
        tokens_per_update = int(training["global_batch_size"]) * int(
            mapping(config, "data")["sequence_length"]
        )

        for step in range(1, max_steps + 1):
            effective_lr = learning_rate(
                step,
                peak=peak_lr,
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
            step_started = perf_counter()
            result = run_optimizer_boundary(
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
            step_seconds = perf_counter() - step_started
            step_samples.append(step_seconds)
            final_train = result
            attempt.append_event(
                {
                    "event": "train",
                    "condition_id": condition_id,
                    "step": step,
                    "input_tokens_seen": step * tokens_per_update,
                    "elapsed_seconds": perf_counter() - condition_started,
                    "step_wall_seconds": step_seconds,
                    "learning_rate": effective_lr,
                    "tokens_per_second": tokens_per_update / step_seconds,
                    "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                    **result,
                }
            )
            if step == 1:
                first_validation, first_validation_seconds = timed_validation(
                    model=model,
                    tokens=validation_tokens,
                    config=config,
                    torch=torch,
                    np=np,
                )
                attempt.append_event(
                    {
                        "event": "validation",
                        "condition_id": condition_id,
                        "step": step,
                        "elapsed_seconds": perf_counter() - condition_started,
                        "wall_seconds": first_validation_seconds,
                        **first_validation,
                    }
                )

        if final_train is None or first_validation is None:
            raise RuntimeError("Training did not produce the required events.")

        checkpoint_started = perf_counter()
        checkpoint_dir = attempt.attempt_dir / "checkpoints" / "final"
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
        torch.cuda.synchronize()
        checkpoint_save_seconds = perf_counter() - checkpoint_started
        checkpoint_inventory = build_transfer_inventory(checkpoint_dir)
        checkpoint_hash = inventory_content_sha256(checkpoint_inventory)
        write_json(
            attempt.attempt_dir / "diagnostics" / "checkpoint_inventory.json",
            checkpoint_inventory,
        )

        del optimizer
        del model
        gc.collect()
        torch.cuda.empty_cache()
        reload_started = perf_counter()
        model = load_checkpoint_pythia(AutoModelForCausalLM, checkpoint_dir, torch=torch)
        model.config.use_cache = False
        model.to(device=torch.device("cuda"), dtype=torch.float32)
        torch.cuda.synchronize()
        checkpoint_reload_seconds = perf_counter() - reload_started

        final_validation, activation_diagnostics, final_validation_seconds = (
            timed_diagnostic_validation(
                model=model,
                tokens=validation_tokens,
                config=config,
                torch=torch,
                np=np,
            )
        )
        weight_started = perf_counter()
        weights = weight_statistics(model)
        pooled_weights = pool_weight_norm(weights)
        weight_seconds = perf_counter() - weight_started
        write_json(
            attempt.attempt_dir / "diagnostics" / "activation_statistics.json",
            activation_diagnostics,
        )
        write_json(
            attempt.attempt_dir / "diagnostics" / "weight_statistics.json",
            {"rows": weights, "pooled": pooled_weights},
        )
        attempt.append_event(
            {
                "event": "validation",
                "condition_id": condition_id,
                "step": max_steps,
                "source": "reloaded_final_checkpoint",
                "elapsed_seconds": perf_counter() - condition_started,
                "wall_seconds": final_validation_seconds,
                **final_validation,
            }
        )

        total_seconds = perf_counter() - condition_started
        final_lr = learning_rate(
            max_steps,
            peak=peak_lr,
            max_steps=max_steps,
            warmup=warmup,
            minimum_ratio=float(training["minimum_learning_rate_ratio"]),
        )
        metrics = {
            "condition": deepcopy(resolved["condition"]),
            "training": {
                "completed_steps": max_steps,
                "input_tokens": max_steps * tokens_per_update,
                "warmup_steps": warmup,
                "schedule_hash": schedule_hash,
                "task_loss_final": final_train["task_loss"],
                "learning_rate_final": final_lr,
                "optimizer_step_seconds": step_samples,
                "median_step_seconds": median(step_samples),
                "median_tokens_per_second": tokens_per_update / median(step_samples),
                "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
            "validation": {"step_one": first_validation, "final": final_validation},
            "diagnostics": {
                "activation_statistics_path": "diagnostics/activation_statistics.json",
                "weight_statistics_path": "diagnostics/weight_statistics.json",
                "activation_sites": list(mapping(config, "diagnostics")["activation_sites"]),
                "near_zero_thresholds": list(
                    mapping(config, "diagnostics")["near_zero_thresholds"]
                ),
            },
            "checkpoint": {
                "path": "checkpoints/final",
                "content_sha256": checkpoint_hash,
                "bytes": checkpoint_inventory["total_bytes"],
                "optimizer_saved": False,
            },
            "timing": {
                "cache_verification_seconds": cache_verification_seconds,
                "setup_seconds": setup_seconds,
                "step_one_validation_seconds": first_validation_seconds,
                "final_diagnostic_validation_seconds": final_validation_seconds,
                "weight_statistics_seconds": weight_seconds,
                "checkpoint_save_seconds": checkpoint_save_seconds,
                "checkpoint_reload_seconds": checkpoint_reload_seconds,
                "total_seconds": total_seconds,
            },
        }
        attempt.complete(
            metrics=metrics,
            predictions=[],
            manifest_updates={
                "completed_steps": max_steps,
                "input_tokens": max_steps * tokens_per_update,
                "initial_parameter_sha256": initial_hash,
                "training_schedule_hash": schedule_hash,
                "topology": realized_topology,
                "validation_coverage": final_validation,
                "checkpoint": metrics["checkpoint"],
            },
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return attempt.attempt_dir


def timed_validation(
    *, model: Any, tokens: Any, config: Mapping[str, Any], torch: Any, np: Any
) -> tuple[dict[str, Any], float]:
    torch.cuda.synchronize()
    started = perf_counter()
    result = evaluate_complete_blocks(
        model=model,
        tokens=tokens,
        block_size=int(mapping(config, "data")["sequence_length"]),
        batch_size=int(mapping(config, "validation")["batch_size"]),
        device=torch.device("cuda"),
        torch=torch,
        np=np,
        autocast_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    require_validation_coverage(result, config)
    return result, perf_counter() - started


def timed_diagnostic_validation(
    *, model: Any, tokens: Any, config: Mapping[str, Any], torch: Any, np: Any
) -> tuple[dict[str, Any], dict[str, Any], float]:
    diagnostics = mapping(config, "diagnostics")
    accumulator = ActivationAccumulator(
        tuple(float(value) for value in diagnostics["near_zero_thresholds"])
    )
    torch.cuda.synchronize()
    started = perf_counter()
    with ActivationCapture(
        model, list(diagnostics["activation_sites"]), torch=torch
    ) as capture:

        def observe(_output: Any, _batch_sequences: int) -> None:
            accumulator.update(capture.activations, torch=torch)
            capture.clear()

        result = evaluate_complete_blocks(
            model=model,
            tokens=tokens,
            block_size=int(mapping(config, "data")["sequence_length"]),
            batch_size=int(mapping(config, "validation")["batch_size"]),
            device=torch.device("cuda"),
            torch=torch,
            np=np,
            autocast_dtype=torch.bfloat16,
            after_batch=observe,
        )
    torch.cuda.synchronize()
    require_validation_coverage(result, config)
    return (
        result,
        {"rows": accumulator.rows(), "pooled_by_site": accumulator.pooled_by_site()},
        perf_counter() - started,
    )
