"""Five-worker, six-condition Run 004 training and immutable attempt publication."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import gc
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Mapping

from sparsity_research.artifacts import attempt_lifecycle, build_transfer_inventory
from sparsity_research.capture import ActivationCapture
from sparsity_research.evaluation import evaluate_complete_blocks
from sparsity_research.metrics import pool_weight_norm, weight_statistics
from sparsity_research.optimization import set_learning_rate
from sparsity_research.pressure import parse_pressure_config
from sparsity_research.pythia import build_random_pythia, load_checkpoint_pythia, topology_metadata

from diagnostics import activation_diagnostic_validation, logical_product_validation
from initialization import apply_pythia_14m_initialization, verify_recipe_model
from optimizer_boundary import (
    DynamicLossScaler,
    build_recipe_adamw,
    recipe_attention_context,
    recipe_learning_rate,
    run_recipe_boundary,
)
from run_config import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    RUN_DIR,
    build_schedule,
    cache_identity,
    git_identity,
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
    worker_conditions,
    write_json,
)


def run_worker(worker_id: str, config_path: str | Path = DEFAULT_CONFIG) -> list[Path]:
    import numpy as np
    import torch

    config = load_config(config_path)
    require_cuda(torch)
    conditions = worker_conditions(config, worker_id)
    _require_attempt_slots_available(conditions)
    train_tokens, validation_tokens, train_meta, validation_meta, cache_seconds = load_verified_caches(config, np=np)
    starts, schedule_hash, schedule_metadata = build_schedule(config, train_meta, np=np)
    code_identity = run_code_identity()
    repository_identity = git_identity()
    progress_path = RUN_DIR / "artifacts" / "workers" / worker_id / "progress.json"
    worker_started = perf_counter()
    results = []
    for index, condition in enumerate(conditions):
        write_json(
            progress_path,
            {
                "status": "running",
                "worker_id": worker_id,
                "condition_count": len(conditions),
                "completed_conditions": index,
                "current_condition": condition["id"],
                "elapsed_seconds": perf_counter() - worker_started,
            },
        )
        results.append(
            run_condition(
                config=config,
                worker_id=worker_id,
                condition=condition,
                train_tokens=train_tokens,
                validation_tokens=validation_tokens,
                train_metadata=train_meta,
                validation_metadata=validation_meta,
                starts=starts,
                schedule_hash=schedule_hash,
                schedule_metadata=schedule_metadata,
                cache_verification_seconds=cache_seconds,
                code_identity=code_identity,
                repository_identity=repository_identity,
                torch=torch,
                np=np,
            )
        )
        cache_seconds = 0.0
        gc.collect()
        torch.cuda.empty_cache()
    write_json(
        progress_path,
        {
            "status": "completed",
            "worker_id": worker_id,
            "condition_count": len(conditions),
            "completed_conditions": len(conditions),
            "current_condition": None,
            "attempts": [path.name for path in results],
            "elapsed_seconds": perf_counter() - worker_started,
        },
    )
    return results


def run_condition(
    *,
    config: Mapping[str, Any],
    worker_id: str,
    condition: Mapping[str, Any],
    train_tokens: Any,
    validation_tokens: Any,
    train_metadata: Mapping[str, Any],
    validation_metadata: Mapping[str, Any],
    starts: Any,
    schedule_hash: str,
    schedule_metadata: Mapping[str, Any],
    cache_verification_seconds: float,
    code_identity: Mapping[str, Any],
    repository_identity: Mapping[str, Any],
    torch: Any,
    np: Any,
) -> Path:
    from transformers import AutoConfig, AutoModelForCausalLM

    resolved = resolved_condition_config(config, condition)
    condition_id = str(condition["id"])
    condition_order = int(condition["order"])
    command = f"python {RUN_DIR.relative_to(REPO_ROOT).as_posix()}/02_train.py --worker {worker_id}"
    extra_identity = {
        "code": deepcopy(dict(repository_identity)),
        "worker_id": worker_id,
        "condition": deepcopy(dict(condition)),
        "model": {**deepcopy(dict(mapping(resolved, "model"))), "loaded_checkpoint_weights": False},
        "recipe": deepcopy(dict(mapping(resolved, "recipe"))),
        "data": {
            "train": cache_identity(train_metadata),
            "validation": cache_identity(validation_metadata),
            "training_schedule_hash": schedule_hash,
            "training_schedule": dict(schedule_metadata),
        },
        "seeds": deepcopy(dict(mapping(config, "seeds"))),
        "run_code": deepcopy(dict(code_identity)),
        "activation_pressure": deepcopy(dict(mapping(resolved, "activation_pressure"))),
    }
    with attempt_lifecycle(
        RUN_DIR,
        config=resolved,
        command=command,
        mode="pretrain",
        extra_identity=extra_identity,
        attempt_sequence=condition_order,
    ) as attempt:
        condition_started = perf_counter()
        seed = int(mapping(config, "seeds")["model"])
        seed_everything(torch, seed)
        torch.cuda.reset_peak_memory_stats()
        model = build_random_pythia(
            dict(mapping(resolved, "model")),
            device=torch.device("cuda"),
            torch=torch,
            auto_config=AutoConfig,
            auto_model=AutoModelForCausalLM,
        )
        seed_everything(torch, seed)
        initialization = apply_pythia_14m_initialization(model, torch=torch)
        recipe_model = verify_recipe_model(model)
        realized_topology = topology_metadata(model)
        initial_hash = parameter_sha256(model)
        optimizer, optimizer_mapping = build_recipe_adamw(
            model, dict(mapping(resolved, "training")), torch=torch
        )
        training = mapping(config, "training")
        scaler = DynamicLossScaler(
            scale=2.0 ** int(training["initial_loss_scale_power"]),
            growth_interval=int(training["loss_scale_window"]),
            hysteresis=int(training["loss_scale_hysteresis"]),
            minimum_scale=float(training["minimum_loss_scale"]),
        )
        pressure = parse_pressure_config(dict(mapping(resolved, "activation_pressure")))
        max_steps = int(training["max_steps"])
        warmup_iterations = float(training["warmup_fraction"]) * max_steps
        tokens_per_update = int(training["global_batch_size"]) * int(mapping(config, "data")["sequence_length"])
        checkpoint_steps = set(int(value) for value in mapping(config, "checkpoints")["model_steps"])
        optimizer_checkpoint_steps = set(int(value) for value in mapping(config, "checkpoints")["optimizer_steps"])
        checkpoints_root = attempt.attempt_dir / "checkpoints"
        _save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=0,
            include_optimizer=False,
            root=checkpoints_root,
            schedule_hash=schedule_hash,
            torch=torch,
        )

        first_validation = None
        first_validation_seconds = None
        final_train = None
        step_samples: list[float] = []
        overflow_steps: list[int] = []
        capture_context = ActivationCapture(model, ["h"], torch=torch) if pressure.enabled else nullcontext(None)
        with capture_context as training_capture:
            for step in range(1, max_steps + 1):
                effective_lr = recipe_learning_rate(
                    step,
                    peak=float(training["peak_learning_rate"]),
                    max_steps=max_steps,
                    warmup_fraction=float(training["warmup_fraction"]),
                    minimum=float(training["minimum_learning_rate"]),
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
                result = run_recipe_boundary(
                    model=model,
                    optimizer=optimizer,
                    batches=batches,
                    pressure=pressure,
                    capture=training_capture,
                    loss_scaler=scaler,
                    gradient_clip_norm=float(training["gradient_clip_norm"]),
                    torch=torch,
                    device=torch.device("cuda"),
                )
                torch.cuda.synchronize()
                step_seconds = perf_counter() - step_started
                del batches
                step_samples.append(step_seconds)
                final_train = result
                if result["optimizer_step_skipped"]:
                    overflow_steps.append(step)
                attempt.append_event(
                    {
                        "event": "train",
                        "worker_id": worker_id,
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
                if step in checkpoint_steps:
                    _save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=step,
                        include_optimizer=step in optimizer_checkpoint_steps,
                        root=checkpoints_root,
                        schedule_hash=schedule_hash,
                        torch=torch,
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
                    if training_capture is not None:
                        training_capture.clear()

        if final_train is None or first_validation is None:
            raise RuntimeError("Training did not produce all required events.")
        final_checkpoint_dir = checkpoints_root / f"step_{max_steps:06d}"
        checkpoint_inventory = build_transfer_inventory(checkpoints_root)
        checkpoint_inventory_hash = inventory_content_sha256(checkpoint_inventory)
        final_checkpoint_inventory = build_transfer_inventory(final_checkpoint_dir)
        final_checkpoint_hash = inventory_content_sha256(final_checkpoint_inventory)
        write_json(attempt.attempt_dir / "diagnostics" / "checkpoint_inventory.json", checkpoint_inventory)

        del optimizer
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = load_checkpoint_pythia(AutoModelForCausalLM, final_checkpoint_dir, torch=torch)
        model.to(device=torch.device("cuda"), dtype=torch.float32)
        verify_recipe_model(model)
        if topology_metadata(model) != realized_topology:
            raise RuntimeError("Reloaded final topology differs from the trained topology.")

        final_validation, final_validation_seconds = timed_validation(
            model=model,
            tokens=validation_tokens,
            config=config,
            torch=torch,
            np=np,
        )
        activation_started = perf_counter()
        activation_coverage, activation_statistics = activation_diagnostic_validation(
            model=model,
            tokens=validation_tokens,
            config=config,
            torch=torch,
            np=np,
        )
        activation_seconds = perf_counter() - activation_started
        logical_started = perf_counter()
        logical_coverage, logical_products, architecture_maximum = logical_product_validation(
            model=model,
            tokens=validation_tokens,
            config=config,
            torch=torch,
            np=np,
        )
        logical_seconds = perf_counter() - logical_started
        weights = weight_statistics(model)
        pooled_weights = pool_weight_norm(weights)
        write_json(attempt.attempt_dir / "diagnostics" / "activation_statistics.json", activation_statistics)
        write_json(
            attempt.attempt_dir / "diagnostics" / "weight_statistics.json",
            {"inclusion": "all named parameters, including bias and normalization", "rows": weights, "pooled": pooled_weights},
        )
        write_json(
            attempt.attempt_dir / "diagnostics" / "logical_products.json",
            {
                "coverage": logical_coverage,
                "measured": logical_products,
                "architecture_maximum": architecture_maximum,
                "interpretation": "logical zero-product opportunities; not removed FLOPs or measured speedup",
            },
        )
        attempt.append_event(
            {
                "event": "validation",
                "condition_id": condition_id,
                "step": max_steps,
                "source": "reloaded_final_checkpoint_recipe_sdpa",
                "elapsed_seconds": perf_counter() - condition_started,
                "wall_seconds": final_validation_seconds,
                **final_validation,
            }
        )

        metrics = {
            "condition": deepcopy(dict(condition)),
            "worker_id": worker_id,
            "recipe_mapping": {
                "model": recipe_model,
                "initialization": initialization,
                "optimizer": optimizer_mapping,
                "framework_bitwise_reproduction": False,
            },
            "training": {
                "completed_steps": max_steps,
                "input_tokens": max_steps * tokens_per_update,
                "warmup_iterations": warmup_iterations,
                "lr_schedule_semantics": "GPT-NeoX v1 scheduler value before each optimizer step",
                "schedule_hash": schedule_hash,
                "wrapped_blocks": int(schedule_metadata["wrapped_blocks"]),
                "task_loss_final": final_train["task_loss"],
                "final_boundary": deepcopy(dict(final_train)),
                "overflow_steps": overflow_steps,
                "optimizer_step_count": max_steps - len(overflow_steps),
                "learning_rate_final": recipe_learning_rate(
                    max_steps,
                    peak=float(training["peak_learning_rate"]),
                    max_steps=max_steps,
                    warmup_fraction=float(training["warmup_fraction"]),
                    minimum=float(training["minimum_learning_rate"]),
                ),
                "optimizer_step_seconds": step_samples,
                "median_step_seconds": median(step_samples),
                "median_tokens_per_second": tokens_per_update / median(step_samples),
                "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                "final_loss_scaler": scaler.state_dict(),
            },
            "validation": {
                "step_one": first_validation,
                "final": final_validation,
                "activation_diagnostic": activation_coverage,
                "logical_product_diagnostic_eager": logical_coverage,
            },
            "diagnostics": {
                "activation_statistics_path": "diagnostics/activation_statistics.json",
                "weight_statistics_path": "diagnostics/weight_statistics.json",
                "logical_products_path": "diagnostics/logical_products.json",
                "activation_sites": list(mapping(config, "diagnostics")["activation_sites"]),
                "near_zero_thresholds": list(mapping(config, "diagnostics")["near_zero_thresholds"]),
                "activation_seconds": activation_seconds,
                "logical_product_seconds": logical_seconds,
            },
            "checkpoints": {
                "root": "checkpoints",
                "inventory_content_sha256": checkpoint_inventory_hash,
                "total_bytes": checkpoint_inventory["total_bytes"],
                "model_steps": sorted(checkpoint_steps),
                "optimizer_steps": sorted(optimizer_checkpoint_steps),
                "final": {
                    "path": f"checkpoints/step_{max_steps:06d}",
                    "content_sha256": final_checkpoint_hash,
                    "bytes": final_checkpoint_inventory["total_bytes"],
                    "optimizer_saved": True,
                },
            },
            "timing": {
                "cache_verification_seconds": cache_verification_seconds,
                "step_one_validation_seconds": first_validation_seconds,
                "final_validation_seconds": final_validation_seconds,
                "total_seconds": perf_counter() - condition_started,
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
                "checkpoints": metrics["checkpoints"],
                "gradient_overflow_steps": overflow_steps,
            },
        )
        transfer_inventory = build_transfer_inventory(attempt.attempt_dir)
        write_json(attempt.attempt_dir / "transfer_inventory.json", transfer_inventory)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return attempt.attempt_dir


def timed_validation(
    *, model: Any, tokens: Any, config: Mapping[str, Any], torch: Any, np: Any
) -> tuple[dict[str, Any], float]:
    torch.cuda.synchronize()
    started = perf_counter()
    device = torch.device("cuda")
    with recipe_attention_context(torch, device):
        result = evaluate_complete_blocks(
            model=model,
            tokens=tokens,
            block_size=int(mapping(config, "data")["sequence_length"]),
            batch_size=int(mapping(config, "validation")["batch_size"]),
            device=device,
            torch=torch,
            np=np,
            autocast_dtype=torch.float16,
        )
    torch.cuda.synchronize()
    require_validation_coverage(result, config)
    return result, perf_counter() - started


def _save_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    scaler: DynamicLossScaler,
    step: int,
    include_optimizer: bool,
    root: Path,
    schedule_hash: str,
    torch: Any,
) -> Path:
    target = root / f"step_{int(step):06d}"
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    target.mkdir(parents=True)
    model.save_pretrained(target, safe_serialization=True)
    state = {
        "step": int(step),
        "schedule_hash": schedule_hash,
        "loss_scaler": scaler.state_dict(),
        "optimizer_saved": bool(include_optimizer),
    }
    if include_optimizer:
        torch.save(
            {
                **state,
                "optimizer": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all(),
            },
            target / "training_state.pt",
        )
    write_json(target / "checkpoint_metadata.json", state)
    return target


def _require_attempt_slots_available(conditions: list[Mapping[str, Any]]) -> None:
    root = RUN_DIR / "artifacts" / "attempts"
    if not root.exists():
        return
    existing = [path.name for path in root.iterdir() if path.is_dir()]
    for condition in conditions:
        prefix = f"{int(condition['order']):03d}-"
        if any(name.startswith(prefix) for name in existing):
            raise RuntimeError(
                f"Attempt slot {prefix[:-1]} already exists; infrastructure retry requires a new attempt plan."
            )
