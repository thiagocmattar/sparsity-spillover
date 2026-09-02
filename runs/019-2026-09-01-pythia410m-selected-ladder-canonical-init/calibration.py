"""Exact non-evidence GPU calibration and 12-condition ETC projection for Run 019."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import platform
from statistics import median
import sys
import tempfile
from time import perf_counter
from typing import Any, Mapping

from sparsity_research.artifacts import build_transfer_inventory
from sparsity_research.metrics import pool_weight_norm, weight_statistics
from sparsity_research.optimization import set_learning_rate
from sparsity_research.pressure import parse_pressure_config
from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata

from diagnostics import activation_diagnostic_validation, logical_product_validation
from initialization import verify_recipe_model
from initialization_artifact import load_pinned_initialization
from model_factory import build_pinned_run019_model
from optimizer_boundary import (
    DynamicLossScaler,
    build_recipe_adamw,
    recipe_learning_rate,
    run_recipe_boundary,
)
from run019_capture import ConditionPressureCapture
from run_config import (
    EXPECTED_CALIBRATION_CONDITIONS,
    EXPECTED_INITIAL_PARAMETER_SHA256,
    build_schedule,
    cache_identity,
    condition_specs,
    load_config,
    load_verified_caches,
    mapping,
    require_cuda,
    resolved_condition_config,
    run_code_identity,
    seed_everything,
    write_json,
)
from teal_posthoc import _calibrate as calibrate_teal_thresholds
from teal_posthoc import _evaluate_point as evaluate_teal_point
from teal_posthoc import _thresholds_for_target
from training import (
    _microbatches_for_step,
    _save_checkpoint,
    _verified_initial_parameter_sha256,
    timed_validation,
)


DEVICE_TOKENS = {
    "NVIDIA RTX A6000": ("RTX", "A6000"),
    "NVIDIA A40": ("A40",),
    "NVIDIA L40S": ("L40S",),
    "NVIDIA A100 80GB PCIe": ("A100", "80GB"),
    "NVIDIA A100-SXM4-80GB": ("A100", "SXM4", "80GB"),
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": ("RTX", "PRO", "6000"),
    "NVIDIA H100 80GB HBM3": ("H100",),
    "NVIDIA H200": ("H200",),
}


def run_calibration(
    *,
    gpu_type_id: str,
    cloud_type: str,
    hourly_price_usd: float,
    output: Path,
) -> dict[str, Any]:
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    config = load_config()
    calibration = mapping(config, "calibration")
    if gpu_type_id not in tuple(calibration["candidate_gpu_type_ids"]):
        raise ValueError(f"GPU type is outside the approved calibration candidates: {gpu_type_id}")
    if cloud_type not in {"COMMUNITY", "SECURE"}:
        raise ValueError("cloud_type must be COMMUNITY or SECURE.")
    if not math.isfinite(hourly_price_usd) or hourly_price_usd <= 0.0:
        raise ValueError("A positive live hourly price is required.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite calibration output: {output}")

    realized_runtime = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": torch.__version__.split("+", 1)[0],
        "transformers": transformers.__version__,
        "cuda_runtime": str(torch.version.cuda),
    }
    if realized_runtime != dict(mapping(config, "runtime")):
        raise RuntimeError(
            f"Pinned runtime mismatch: realized={realized_runtime}, "
            f"expected={dict(mapping(config, 'runtime'))}"
        )
    require_cuda(torch)
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    required_tokens = DEVICE_TOKENS[gpu_type_id]
    if any(token.casefold() not in device_name.casefold() for token in required_tokens):
        raise RuntimeError(
            f"Visible device {device_name!r} does not match catalog type {gpu_type_id!r}."
        )
    flash_probe = getattr(torch.backends.cuda, "is_flash_attention_available", None)
    if flash_probe is None or not bool(flash_probe()):
        raise RuntimeError("The pinned training mapping requires CUDA flash SDPA support.")
    properties = torch.cuda.get_device_properties(0)

    cache_started = perf_counter()
    train, validation, train_metadata, validation_metadata, verification_seconds = (
        load_verified_caches(config, np=np)
    )
    starts, schedule_hash, schedule = build_schedule(config, train_metadata, np=np)
    cache_result = {
        "train": cache_identity(train_metadata),
        "validation": cache_identity(validation_metadata),
        "verification_seconds": verification_seconds,
        "wall_seconds": perf_counter() - cache_started,
        "schedule": schedule,
        "schedule_sha256": schedule_hash,
    }
    _event("cache_verified", schedule_sha256=schedule_hash, wall_seconds=cache_result["wall_seconds"])

    samples = []
    initial_hashes = set()
    probe_ids = tuple(str(value) for value in calibration["condition_ids"])
    boundaries = int(calibration["boundaries_per_condition"])
    warmup = int(calibration["warmup_boundaries_excluded_from_timing"])
    for condition_id in probe_ids:
        condition = next(row for row in condition_specs(config) if row["id"] == condition_id)
        resolved = resolved_condition_config(config, condition)
        seed = int(mapping(config, "seeds")["model"])
        seed_everything(torch, seed)
        setup_started = perf_counter()
        model = build_pinned_run019_model(
            dict(mapping(resolved, "model")),
            device=torch.device("cpu"),
            torch=torch,
            auto_model=AutoModelForCausalLM,
        )
        model.config.pressure_sites = list(condition["pressure_sites"])
        initialization = load_pinned_initialization(model, torch=torch)
        recipe = verify_recipe_model(model)
        initial_hash = _verified_initial_parameter_sha256(model)
        initial_hashes.add(initial_hash)
        optimizer, optimizer_mapping = build_recipe_adamw(
            model, dict(mapping(config, "training")), torch=torch
        )
        setup_seconds = perf_counter() - setup_started
        realized_topology = topology_metadata(model)
        _event(
            "condition_ready",
            condition_id=condition_id,
            setup_seconds=setup_seconds,
            initial_parameter_sha256=initial_hash,
        )

        pressure = parse_pressure_config(dict(mapping(resolved, "activation_pressure")))
        training = mapping(config, "training")
        scaler = DynamicLossScaler(
            scale=2.0 ** int(training["initial_loss_scale_power"]),
            growth_interval=int(training["loss_scale_window"]),
            hysteresis=int(training["loss_scale_hysteresis"]),
            minimum_scale=float(training["minimum_loss_scale"]),
        )
        capture_context = (
            ConditionPressureCapture(model, ["h"], torch=torch)
            if pressure.enabled
            else nullcontext(None)
        )
        boundary_rows = []
        torch.cuda.reset_peak_memory_stats()
        with capture_context as capture:
            for index in range(boundaries):
                set_learning_rate(
                    optimizer,
                    recipe_learning_rate(
                        index + 1,
                        peak=float(training["peak_learning_rate"]),
                        max_steps=int(training["max_steps"]),
                        warmup_fraction=float(training["warmup_fraction"]),
                        minimum=float(training["minimum_learning_rate"]),
                    ),
                )
                model.train()
                batches_for_boundary = _microbatches_for_step(
                    train,
                    starts[index],
                    block_size=int(mapping(config, "data")["sequence_length"]),
                    device=device,
                    torch=torch,
                    np=np,
                )
                torch.cuda.synchronize()
                started = perf_counter()
                boundary = run_recipe_boundary(
                    model=model,
                    optimizer=optimizer,
                    batches=batches_for_boundary,
                    pressure=pressure,
                    capture=capture,
                    loss_scaler=scaler,
                    gradient_clip_norm=float(training["gradient_clip_norm"]),
                    torch=torch,
                    device=device,
                )
                torch.cuda.synchronize()
                seconds = perf_counter() - started
                row = {
                    "step": index + 1,
                    "warmup_excluded": index < warmup,
                    "wall_seconds": seconds,
                    "tokens_per_second": 2_097_152 / seconds,
                    **boundary,
                }
                boundary_rows.append(row)
                _event(
                    "boundary_complete",
                    condition_id=condition_id,
                    step=index + 1,
                    task_loss=boundary["task_loss"],
                    wall_seconds=seconds,
                    tokens_per_second=row["tokens_per_second"],
                    peak_memory_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                    optimizer_step_skipped=boundary["optimizer_step_skipped"],
                )

        timed_rows = boundary_rows[warmup:]
        sample: dict[str, Any] = {
            "condition_id": condition_id,
            "setup_seconds": setup_seconds,
            "initial_parameter_sha256": initial_hash,
            "topology": realized_topology,
            "initialization": initialization,
            "recipe": recipe,
            "optimizer": optimizer_mapping,
            "boundaries": boundary_rows,
            "timed_boundary_count": len(timed_rows),
            "median_end_to_end_step_seconds": median(row["wall_seconds"] for row in timed_rows),
            "min_end_to_end_step_seconds": min(row["wall_seconds"] for row in timed_rows),
            "max_end_to_end_step_seconds": max(row["wall_seconds"] for row in timed_rows),
        }

        if condition_id != "a7-ol1-kappa-0p5":
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()

        if condition_id == "a0-gelu":
            teal = mapping(config, "teal_posthoc")
            teal_calibration_started = perf_counter()
            teal_thresholds = calibrate_teal_thresholds(
                model,
                train,
                targets=(0.5,),
                blocks=int(teal["calibration_blocks"]),
                block_size=int(mapping(config, "data")["sequence_length"]),
                device=device,
                torch=torch,
                np=np,
            )
            teal_calibration_seconds = perf_counter() - teal_calibration_started
            teal_point = evaluate_teal_point(
                model,
                validation,
                _thresholds_for_target(teal_thresholds, 0.5),
                block_size=int(mapping(config, "data")["sequence_length"]),
                batch_size=int(teal["evaluation_batch_size"]),
                device=device,
                torch=torch,
                np=np,
            )
            sample["teal_probe"] = {
                "target_sparsity": 0.5,
                "calibration_seconds": teal_calibration_seconds,
                "point": teal_point,
            }
            _event(
                "teal_probe_complete",
                condition_id=condition_id,
                calibration_seconds=teal_calibration_seconds,
                point_seconds=teal_point["evaluation_seconds"],
            )

        if condition_id == "a7-ol1-kappa-0p5":
            with tempfile.TemporaryDirectory(prefix="run019-calibration-checkpoint-") as temporary:
                checkpoint_started = perf_counter()
                checkpoint = _save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=boundaries,
                    include_optimizer=True,
                    root=Path(temporary),
                    schedule_hash=schedule_hash,
                    torch=torch,
                )
                checkpoint_write_seconds = perf_counter() - checkpoint_started
                inventory_started = perf_counter()
                checkpoint_inventory = build_transfer_inventory(checkpoint)
                inventory_seconds = perf_counter() - inventory_started
                del optimizer, model
                gc.collect()
                torch.cuda.empty_cache()
                reload_started = perf_counter()
                model = load_checkpoint_pythia(AutoModelForCausalLM, checkpoint, torch=torch)
                model.to(device=device, dtype=torch.float32)
                verify_recipe_model(model)
                if topology_metadata(model) != realized_topology:
                    raise RuntimeError("Reloaded calibration topology differs from its trained topology.")
                reload_seconds = perf_counter() - reload_started
                validation_result, validation_seconds = timed_validation(
                    model=model, tokens=validation, config=config, torch=torch, np=np
                )
                activation_started = perf_counter()
                activation_coverage, activation_statistics = activation_diagnostic_validation(
                    model=model, tokens=validation, config=config, torch=torch, np=np
                )
                activation_seconds = perf_counter() - activation_started
                logical_started = perf_counter()
                logical_coverage, logical, ceiling = logical_product_validation(
                    model=model, tokens=validation, config=config, torch=torch, np=np
                )
                logical_seconds = perf_counter() - logical_started
                weight_started = perf_counter()
                weights = weight_statistics(model)
                pooled_weights = pool_weight_norm(weights)
                weight_seconds = perf_counter() - weight_started
                sample.update(
                    full_validation={"wall_seconds": validation_seconds, "coverage": validation_result},
                    activation_diagnostic={
                        "wall_seconds": activation_seconds,
                        "coverage": activation_coverage,
                        "statistics_row_count": len(activation_statistics["rows"]),
                        "attention_implementation": activation_statistics["attention_implementation"],
                    },
                    logical_diagnostic={
                        "wall_seconds": logical_seconds,
                        "coverage": logical_coverage,
                        "R_model": logical["R_model"],
                        "R_model_max": ceiling["R_model_max_fraction"],
                    },
                    weight_diagnostic={
                        "wall_seconds": weight_seconds,
                        "parameter_tensor_count": len(weights),
                        "pooled": pooled_weights,
                    },
                    checkpoint={
                        "write_seconds": checkpoint_write_seconds,
                        "inventory_seconds": inventory_seconds,
                        "reload_seconds": reload_seconds,
                        "bytes": checkpoint_inventory["total_bytes"],
                    },
                )
                _event(
                    "diagnostics_complete",
                    condition_id=condition_id,
                    validation_seconds=validation_seconds,
                    activation_seconds=activation_seconds,
                    logical_seconds=logical_seconds,
                    weight_seconds=weight_seconds,
                    checkpoint_seconds=checkpoint_write_seconds + inventory_seconds,
                    reload_seconds=reload_seconds,
                    R_model=logical["R_model"],
                )

        sample["peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        sample["peak_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        samples.append(sample)
        if "optimizer" in locals():
            del optimizer
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if initial_hashes != {EXPECTED_INITIAL_PARAMETER_SHA256}:
        raise RuntimeError("Calibration conditions did not share the pinned initialization hash.")
    memory_limit = (1.0 - float(calibration["minimum_vram_headroom_fraction"])) * int(
        properties.total_memory
    )
    boundary_health = all(
        len(sample["boundaries"]) == boundaries
        and all(
            not row["optimizer_step_skipped"]
            and not row["gradient_overflow"]
            and math.isfinite(float(row["task_loss"]))
            for row in sample["boundaries"]
        )
        for sample in samples
    )
    memory_health = all(
        int(sample["peak_memory_reserved_bytes"]) <= memory_limit for sample in samples
    )
    diagnostics_health = _diagnostics_are_complete(samples, config)
    passed = boundary_health and memory_health and diagnostics_health
    projection = _project_science(
        config,
        samples,
        cache_verification_seconds=float(cache_result["wall_seconds"]),
        hourly_price_usd=hourly_price_usd,
    )
    result = {
        "schema_version": 1,
        "kind": "non_evidence_exact_run019_gpu_calibration",
        "status": "passed" if passed else "failed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "runtime": realized_runtime,
        "price_snapshot": {
            "gpu_type_id": gpu_type_id,
            "cloud_type": cloud_type,
            "hourly_price_usd": hourly_price_usd,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "source": "RunPod live catalog supplied at launch",
        },
        "device": {
            "name": device_name,
            "total_memory_bytes": int(properties.total_memory),
            "headroom_limit_bytes": int(memory_limit),
        },
        "cache": cache_result,
        "run_code": run_code_identity(),
        "probe": {
            "condition_ids": list(probe_ids),
            "boundaries_per_condition": boundaries,
            "warmup_boundaries_excluded_from_timing": warmup,
            "timing_scope": (
                "cache slicing, host-to-device staging, forward/backward, gradient processing, "
                "optimizer update, complete validation, eager and weight diagnostics, checkpoint "
                "write/hash/reload, and one calibrated TEAL point"
            ),
            "scientific_evidence": False,
        },
        "samples": samples,
        "checks": {
            "same_pinned_initial_parameter_hash": True,
            "finite_nonoverflowing_boundaries": boundary_health,
            "fits_with_required_vram_headroom": memory_health,
            "required_diagnostics_complete": diagnostics_health,
        },
        "projection": projection,
    }
    write_json(output, result)
    return result


def _diagnostics_are_complete(samples: list[Mapping[str, Any]], config: Mapping[str, Any]) -> bool:
    by_id = {str(sample["condition_id"]): sample for sample in samples}
    a0 = by_id.get("a0-gelu", {})
    a7 = by_id.get("a7-ol1-kappa-0p5", {})
    teal = a0.get("teal_probe", {})
    point = teal.get("point", {}) if isinstance(teal, Mapping) else {}
    pooled = point.get("activations_by_site", []) if isinstance(point, Mapping) else []
    return bool(
        a7.get("full_validation")
        and a7.get("activation_diagnostic", {}).get("attention_implementation") == "eager"
        and a7.get("activation_diagnostic", {}).get("statistics_row_count")
        == 24 * len(mapping(config, "diagnostics")["activation_sites"])
        and a7.get("logical_diagnostic")
        and a7.get("weight_diagnostic", {}).get("parameter_tensor_count", 0) > 0
        and a7.get("checkpoint", {}).get("bytes", 0) > 0
        and teal.get("calibration_seconds", 0) > 0
        and point.get("evaluation_seconds", 0) > 0
        and {row.get("name") for row in pooled}
        == set(mapping(config, "teal_posthoc")["diagnostic_sites"])
    )


def _project_science(
    config: Mapping[str, Any],
    samples: list[Mapping[str, Any]],
    *,
    cache_verification_seconds: float,
    hourly_price_usd: float,
) -> dict[str, Any]:
    by_id = {str(sample["condition_id"]): sample for sample in samples}
    diagnostic = by_id["a7-ol1-kappa-0p5"]
    checkpoint = diagnostic["checkpoint"]
    common = (
        2.0 * float(diagnostic["full_validation"]["wall_seconds"])
        + float(diagnostic["activation_diagnostic"]["wall_seconds"])
        + float(diagnostic["logical_diagnostic"]["wall_seconds"])
        + float(diagnostic["weight_diagnostic"]["wall_seconds"])
        + float(checkpoint["write_seconds"])
        + 3.0 * float(checkpoint["inventory_seconds"])
        + float(checkpoint["reload_seconds"])
    )
    teal = by_id["a0-gelu"]["teal_probe"]
    teal_seconds = (
        float(teal["calibration_seconds"])
        + 10.0 * float(teal["point"]["evaluation_seconds"])
        + 4.0 * float(checkpoint["inventory_seconds"])
    )
    per_condition = {}
    for condition in condition_specs(config):
        if condition["is_control"]:
            reference = by_id["a0-gelu"]
        elif condition["topology_id"] == "A4-Z":
            reference = by_id["a4-ol1-kappa-0p5"]
        else:
            reference = by_id["a7-ol1-kappa-0p5"]
        training_seconds = 712.0 * float(reference["median_end_to_end_step_seconds"])
        posthoc = teal_seconds if condition["id"] in ("a0-gelu", "a1h-relu") else 0.0
        total = (
            cache_verification_seconds
            + float(reference["setup_seconds"])
            + training_seconds
            + common
            + posthoc
        )
        per_condition[condition["id"]] = {
            "reference_condition_id": reference["condition_id"],
            "cache_verification_seconds": cache_verification_seconds,
            "model_optimizer_setup_seconds": float(reference["setup_seconds"]),
            "training_seconds": training_seconds,
            "training_seconds_min_step_projection": 712.0
            * float(reference["min_end_to_end_step_seconds"]),
            "training_seconds_max_step_projection": 712.0
            * float(reference["max_end_to_end_step_seconds"]),
            "validation_diagnostic_checkpoint_seconds": common,
            "teal_seconds_if_control": posthoc,
            "projected_total_seconds": total,
            "projected_gpu_cost_usd": total / 3600.0 * hourly_price_usd,
        }
    total_seconds = sum(row["projected_total_seconds"] for row in per_condition.values())
    return {
        "method": (
            "median of four exact end-to-end boundaries after one warm-up for each of A0, "
            "A4 kappa=0.5, and A7 kappa=0.5; measured full passes, weight statistics, "
            "and artifact operations with inventory-pass multiplicities matched to execution"
        ),
        "conditions": per_condition,
        "projected_gpu_seconds": total_seconds,
        "projected_gpu_hours": total_seconds / 3600.0,
        "projected_total_gpu_cost_usd": total_seconds / 3600.0 * hourly_price_usd,
        "projected_twelve_way_makespan_seconds": max(
            row["projected_total_seconds"] for row in per_condition.values()
        ),
        "twelve_gpu_hourly_burn_usd": 12.0 * hourly_price_usd,
        "provision_package_install_input_upload_output_download_not_included": True,
        "a1_reference_note": "A1-H uses measured A0 control boundaries.",
        "threshold_reference_note": (
            "Each A4/A7 threshold uses its exact topology's measured kappa=0.5 path; "
            "this is conservative if lower thresholds are faster."
        ),
    }


def _event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)
