"""Matched five-condition Run 010 training and immutable attempt publication."""

from __future__ import annotations

from copy import deepcopy
import gc
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Mapping

from sparsity_research.artifacts import attempt_lifecycle, build_transfer_inventory, config_sha256
from sparsity_research.capture import ActivationCapture
from sparsity_research.data import file_sha256
from sparsity_research.evaluation import evaluate_complete_blocks
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
    DEFAULT_CONFIG,
    REPO_ROOT,
    RUN_DIR,
    build_schedule,
    cache_identity,
    condition_specs,
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
    launch_plan_path = RUN_DIR / "prelaunch" / "launch-plan.json"
    if not launch_plan_path.exists():
        raise RuntimeError("Run 010 has no launch plan; launch approval is required.")
    launch_plan = json.loads(launch_plan_path.read_text(encoding="utf-8"))
    if launch_plan.get("launch_approved") is not True:
        raise RuntimeError("Run 010 launch plan is not explicitly approved.")
    provenance_path = RUN_DIR / "artifacts" / "launch-provenance.json"
    if not provenance_path.exists():
        raise RuntimeError("Run 010 launch provenance sidecar is missing.")
    launch_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    attempts_root = RUN_DIR / "artifacts" / "attempts"
    if attempts_root.exists() and any(path.is_dir() for path in attempts_root.iterdir()):
        raise RuntimeError("Run 010 already has attempts; do not rewrite its scientific cohort.")
    train_tokens, validation_tokens, train_meta, validation_meta, cache_seconds = (
        load_verified_caches(config, np=np)
    )
    starts, schedule_hash, schedule_metadata = build_schedule(config, train_meta, np=np)
    conditions = condition_specs(config)
    code_identity = run_code_identity()
    if launch_provenance.get("config_sha256") != config_sha256(config):
        raise RuntimeError("Base config changed after launch provenance was recorded.")
    if launch_provenance.get("run_code_content_sha256") != code_identity["content_sha256"]:
        raise RuntimeError("Run code changed after launch provenance was recorded.")
    if launch_provenance.get("training_schedule_sha256") != schedule_hash:
        raise RuntimeError("Training schedule changed after launch provenance was recorded.")
    calibration = launch_provenance.get("calibration")
    if not isinstance(calibration, Mapping):
        raise RuntimeError("Launch provenance calibration identity is missing.")
    calibration_path = RUN_DIR / str(calibration.get("path"))
    if file_sha256(calibration_path) != calibration.get("sha256"):
        raise RuntimeError("Calibration artifact changed after launch approval.")
    progress_path = RUN_DIR / "artifacts" / "progress.json"
    cohort_started = perf_counter()
    results = []
    write_json(
        progress_path,
        {
            "status": "running",
            "condition_count": len(conditions),
            "completed_conditions": 0,
            "completed_optimizer_steps": 0,
            "total_optimizer_steps": len(conditions) * int(mapping(config, "training")["max_steps"]),
            "current_condition": conditions[0]["id"],
        },
    )
    for index, condition in enumerate(conditions):
        write_json(
            progress_path,
            {
                "status": "running",
                "condition_count": len(conditions),
                "completed_conditions": index,
                "completed_optimizer_steps": index * int(mapping(config, "training")["max_steps"]),
                "total_optimizer_steps": len(conditions) * int(mapping(config, "training")["max_steps"]),
                "current_condition": condition["id"],
                "elapsed_seconds": perf_counter() - cohort_started,
            },
        )
        results.append(
            run_condition(
                config=config,
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
                launch_provenance=launch_provenance,
                torch=torch,
                np=np,
            )
        )
        cache_seconds = 0.0
        gc.collect()
        torch.cuda.empty_cache()
        complete = index + 1
        write_json(
            progress_path,
            {
                "status": "running" if complete < len(conditions) else "training_completed",
                "condition_count": len(conditions),
                "completed_conditions": complete,
                "completed_optimizer_steps": complete * int(mapping(config, "training")["max_steps"]),
                "total_optimizer_steps": len(conditions) * int(mapping(config, "training")["max_steps"]),
                "current_condition": conditions[complete]["id"] if complete < len(conditions) else None,
                "elapsed_seconds": perf_counter() - cohort_started,
            },
        )
    return results


def run_condition(
    *,
    config: Mapping[str, Any],
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
    launch_provenance: Mapping[str, Any],
    torch: Any,
    np: Any,
) -> Path:
    from transformers import AutoConfig, AutoModelForCausalLM

    resolved = resolved_condition_config(config, condition)
    command = f"python {RUN_DIR.relative_to(REPO_ROOT).as_posix()}/02_train.py"
    extra_identity = {
        "code": {
            "git_commit": launch_provenance["git_commit"],
            "git_dirty": bool(launch_provenance["git_dirty"]),
        },
        "launch_provenance": deepcopy(dict(launch_provenance)),
        "condition": deepcopy(dict(condition)),
        "model": {**deepcopy(dict(mapping(resolved, "model"))), "loaded_checkpoint_weights": False},
        "data": {
            "train": cache_identity(train_metadata),
            "validation": cache_identity(validation_metadata),
            "training_schedule_hash": schedule_hash,
            "training_schedule": dict(schedule_metadata),
        },
        "seeds": deepcopy(dict(mapping(config, "seeds"))),
        "run_code": deepcopy(dict(code_identity)),
        "activation_pressure": deepcopy(dict(mapping(resolved, "activation_pressure"))),
        "gradient_clipping": {"enabled": True, "max_norm": 1.0},
    }
    with attempt_lifecycle(
        RUN_DIR,
        config=resolved,
        command=command,
        mode="pretrain",
        extra_identity=extra_identity,
        attempt_sequence=int(condition["order"]),
    ) as attempt:
        condition_started = perf_counter()
        setup_started = perf_counter()
        seed_everything(torch, int(mapping(config, "seeds")["model"]))
        torch.cuda.reset_peak_memory_stats()
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
        if not pressure.orthogonal or pressure.sites != (
            "a", "m", "h", "q_post", "k_post", "v", "z"
        ):
            raise RuntimeError("Run 010 requires post-threshold OL1 at all seven A7-Z-POST sites.")
        max_steps = int(training["max_steps"])
        warmup = warmup_steps(max_steps, float(training["warmup_fraction"]))
        step_samples: list[float] = []
        first_validation = None
        first_validation_seconds = None
        final_train = None
        tokens_per_update = int(training["global_batch_size"]) * int(
            mapping(config, "data")["sequence_length"]
        )

        capture_context = ActivationCapture(model, pressure.sites, torch=torch)
        with capture_context as training_capture:
            for step in range(1, max_steps + 1):
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
                step_started = perf_counter()
                result = run_optimizer_boundary(
                    model=model,
                    optimizer=optimizer,
                    batches=batches,
                    pressure=pressure,
                    capture=training_capture,
                    torch=torch,
                    device=torch.device("cuda"),
                    autocast_dtype=torch.bfloat16,
                    gradient_clip_norm=float(training["gradient_clip_norm"]),
                )
                torch.cuda.synchronize()
                step_seconds = perf_counter() - step_started
                _require_boundary_contract(result)
                step_samples.append(step_seconds)
                final_train = result
                attempt.append_event(
                    {
                        "event": "train",
                        "condition_id": condition["id"],
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
                            "condition_id": condition["id"],
                            "step": step,
                            "source": "step_one",
                            "elapsed_seconds": perf_counter() - condition_started,
                            "wall_seconds": first_validation_seconds,
                            **first_validation,
                        }
                    )
                    training_capture.clear()
        if final_train is None or first_validation is None or first_validation_seconds is None:
            raise RuntimeError("Training did not produce the required step and validation events.")

        checkpoint_started = perf_counter()
        checkpoint_dir = attempt.attempt_dir / "checkpoints" / "final"
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
        torch.cuda.synchronize()
        checkpoint_save_seconds = perf_counter() - checkpoint_started
        checkpoint_inventory = build_transfer_inventory(checkpoint_dir)
        checkpoint_hash = inventory_content_sha256(checkpoint_inventory)
        write_json(attempt.attempt_dir / "diagnostics" / "checkpoint_inventory.json", checkpoint_inventory)

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
        if topology_metadata(model) != realized_topology:
            raise RuntimeError("Reloaded topology differs from the trained topology.")

        torch.cuda.synchronize()
        activation_started = perf_counter()
        final_validation, activation_statistics = activation_diagnostic_validation(
            model=model, tokens=validation_tokens, config=config, torch=torch, np=np
        )
        torch.cuda.synchronize()
        activation_seconds = perf_counter() - activation_started
        attempt.append_event(
            {
                "event": "validation",
                "condition_id": condition["id"],
                "step": max_steps,
                "source": "reloaded_final_activation_diagnostic",
                "elapsed_seconds": perf_counter() - condition_started,
                "wall_seconds": activation_seconds,
                **final_validation,
            }
        )

        torch.cuda.synchronize()
        logical_started = perf_counter()
        logical_coverage, logical_products, architecture_maximum = logical_product_validation(
            model=model, tokens=validation_tokens, config=config, torch=torch, np=np
        )
        torch.cuda.synchronize()
        logical_seconds = perf_counter() - logical_started
        attempt.append_event(
            {
                "event": "validation",
                "condition_id": condition["id"],
                "step": max_steps,
                "source": "reloaded_final_logical_product_diagnostic_eager",
                "elapsed_seconds": perf_counter() - condition_started,
                "wall_seconds": logical_seconds,
                **logical_coverage,
            }
        )

        weight_started = perf_counter()
        weights = weight_statistics(model)
        pooled_weights = pool_weight_norm(weights)
        weight_seconds = perf_counter() - weight_started
        write_json(attempt.attempt_dir / "diagnostics" / "activation_statistics.json", activation_statistics)
        write_json(
            attempt.attempt_dir / "diagnostics" / "weight_statistics.json",
            {
                "inclusion": "all named parameters, including bias and normalization",
                "rows": weights,
                "pooled": pooled_weights,
            },
        )
        write_json(
            attempt.attempt_dir / "diagnostics" / "logical_products.json",
            {
                "coverage": logical_coverage,
                "measured": logical_products,
                "architecture_maximum": architecture_maximum,
                "interpretation": "logical-product opportunities, not removed FLOPs or speedup",
            },
        )

        metrics = {
            "condition": deepcopy(dict(condition)),
            "training": {
                "completed_steps": max_steps,
                "input_tokens": max_steps * tokens_per_update,
                "warmup_steps": warmup,
                "schedule_hash": schedule_hash,
                "task_loss_final": final_train["task_loss"],
                "final_boundary": deepcopy(dict(final_train)),
                "learning_rate_final": learning_rate(
                    max_steps,
                    peak=float(training["peak_learning_rate"]),
                    max_steps=max_steps,
                    warmup=warmup,
                    minimum_ratio=float(training["minimum_learning_rate_ratio"]),
                ),
                "optimizer_step_seconds": step_samples,
                "median_step_seconds": median(step_samples),
                "median_tokens_per_second": tokens_per_update / median(step_samples),
                "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
            "validation": {
                "step_one": first_validation,
                "final": final_validation,
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
                "final_activation_diagnostic_seconds": activation_seconds,
                "logical_product_seconds": logical_seconds,
                "weight_statistics_seconds": weight_seconds,
                "checkpoint_save_seconds": checkpoint_save_seconds,
                "checkpoint_reload_seconds": checkpoint_reload_seconds,
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
                "logical_product_coverage": logical_coverage,
                "checkpoint": metrics["checkpoint"],
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


def _require_boundary_contract(result: Mapping[str, Any]) -> None:
    if result.get("adamw_gradient_clipping_enabled") is not True:
        raise RuntimeError("The task-gradient clipping contract was not honored.")
    if float(result.get("adamw_gradient_clip_norm", -1.0)) != 1.0:
        raise RuntimeError("Unexpected task-gradient clipping norm.")
    required = {
        "pressure_loss",
        "pressure_gradient_norm",
        "task_pressure_gradient_dot",
        "task_direction_norm",
        "pressure_direction_norm_raw",
        "task_pressure_dot_before",
        "task_pressure_dot_after",
        "projection_applied",
        "pressure_to_task_ratio_raw",
        "trust_scale",
        "pressure_to_task_ratio_final",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise RuntimeError("OL1 boundary metrics missing: " + ", ".join(missing))
    if float(result["pressure_to_task_ratio_final"]) > 1.0 + 1.0e-6:
        raise RuntimeError("OL1 exceeded the approved trust budget.")

