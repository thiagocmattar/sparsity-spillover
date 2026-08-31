#!/usr/bin/env python
"""Exact A40 non-evidence calibration; creates no scientific attempt."""

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

from sparsity_research.artifacts import build_transfer_inventory
from sparsity_research.capture import ActivationCapture
from sparsity_research.pressure import parse_pressure_config
from sparsity_research.pythia import topology_metadata

from diagnostics import activation_diagnostic_validation, logical_product_validation
from initialization import apply_pythia_70m_initialization, verify_recipe_model
from model_factory import build_pinned_run016_model
from optimizer_boundary import DynamicLossScaler, build_recipe_adamw, run_recipe_boundary
from run_config import (
    EXPECTED_INITIAL_PARAMETER_SHA256,
    RUN_DIR,
    build_schedule,
    cache_identity,
    condition_specs,
    load_config,
    load_verified_caches,
    mapping,
    parameter_sha256,
    require_cuda,
    resolved_condition_config,
    run_code_identity,
    seed_everything,
    write_json,
)
from training import _microbatches_for_step, _save_checkpoint, timed_validation


PROBE_CONDITIONS = ("a0-gelu", "a7-ol1-kappa-0")
BOUNDARIES = 5


def main() -> None:
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    config = load_config()
    runtime = mapping(config, "runtime")
    realized = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": torch.__version__.split("+", 1)[0],
        "transformers": transformers.__version__,
        "cuda_runtime": str(torch.version.cuda),
    }
    if realized != dict(runtime):
        raise RuntimeError(f"Pinned runtime mismatch: realized={realized}, expected={dict(runtime)}")
    require_cuda(torch)
    device = torch.device("cuda")
    device_properties = torch.cuda.get_device_properties(0)
    target = mapping(config, "runpod")
    if "A40" not in torch.cuda.get_device_name(0) or int(device_properties.total_memory) < 47 * 1024**3:
        raise RuntimeError("The approved exact preflight must run on an NVIDIA A40 with nominal 48 GB VRAM.")

    cache_started = perf_counter()
    train, validation, train_metadata, validation_metadata, verification_seconds = load_verified_caches(
        config, np=np
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
    print(
        json.dumps(
            {
                "event": "cache_verified",
                "schedule_sha256": schedule_hash,
                "wall_seconds": cache_result["wall_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    samples = []
    initial_hashes = set()
    for condition_id in PROBE_CONDITIONS:
        condition = next(row for row in condition_specs(config) if row["id"] == condition_id)
        resolved = resolved_condition_config(config, condition)
        seed = int(mapping(config, "seeds")["model"])
        seed_everything(torch, seed)
        setup_started = perf_counter()
        model = build_pinned_run016_model(
            dict(mapping(resolved, "model")),
            device=device,
            torch=torch,
            auto_model=AutoModelForCausalLM,
        )
        model.config.pressure_sites = list(condition["pressure_sites"])
        seed_everything(torch, seed)
        initialization = apply_pythia_70m_initialization(model, torch=torch)
        recipe = verify_recipe_model(model)
        initial_hash = parameter_sha256(model)
        initial_hashes.add(initial_hash)
        optimizer, optimizer_mapping = build_recipe_adamw(
            model, dict(mapping(config, "training")), torch=torch
        )
        setup_seconds = perf_counter() - setup_started
        print(
            json.dumps(
                {
                    "event": "condition_ready",
                    "condition_id": condition_id,
                    "setup_seconds": setup_seconds,
                    "initial_parameter_sha256": initial_hash,
                },
                sort_keys=True,
            ),
            flush=True,
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
            ActivationCapture(model, list(pressure.sites), torch=torch)
            if pressure.enabled
            else nullcontext(None)
        )
        boundary_rows = []
        torch.cuda.reset_peak_memory_stats()
        with capture_context as capture:
            for index in range(BOUNDARIES):
                model.train()
                batches = _microbatches_for_step(
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
                    batches=batches,
                    pressure=pressure,
                    capture=capture,
                    loss_scaler=scaler,
                    gradient_clip_norm=float(training["gradient_clip_norm"]),
                    torch=torch,
                    device=device,
                )
                torch.cuda.synchronize()
                seconds = perf_counter() - started
                boundary_rows.append(
                    {
                        "step": index + 1,
                        "wall_seconds": seconds,
                        "tokens_per_second": 2_097_152 / seconds,
                        **boundary,
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "boundary_complete",
                            "condition_id": condition_id,
                            "step": index + 1,
                            "task_loss": boundary["task_loss"],
                            "wall_seconds": seconds,
                            "tokens_per_second": 2_097_152 / seconds,
                            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                            "optimizer_step_skipped": boundary["optimizer_step_skipped"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        sample = {
            "condition_id": condition_id,
            "setup_seconds": setup_seconds,
            "initial_parameter_sha256": initial_hash,
            "topology": topology_metadata(model),
            "initialization": initialization,
            "recipe": recipe,
            "optimizer": optimizer_mapping,
            "boundaries": boundary_rows,
            "median_end_to_end_step_seconds": median(row["wall_seconds"] for row in boundary_rows),
            "min_end_to_end_step_seconds": min(row["wall_seconds"] for row in boundary_rows),
            "max_end_to_end_step_seconds": max(row["wall_seconds"] for row in boundary_rows),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        if condition_id == "a7-ol1-kappa-0":
            validation_result, validation_seconds = timed_validation(
                model=model, tokens=validation, config=config, torch=torch, np=np
            )
            activation_started = perf_counter()
            activation_coverage, _ = activation_diagnostic_validation(
                model=model, tokens=validation, config=config, torch=torch, np=np
            )
            activation_seconds = perf_counter() - activation_started
            logical_started = perf_counter()
            logical_coverage, logical, ceiling = logical_product_validation(
                model=model, tokens=validation, config=config, torch=torch, np=np
            )
            logical_seconds = perf_counter() - logical_started
            with tempfile.TemporaryDirectory(prefix="run016-checkpoint-") as temporary:
                checkpoint_started = perf_counter()
                checkpoint = _save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=712,
                    include_optimizer=True,
                    root=Path(temporary),
                    schedule_hash=schedule_hash,
                    torch=torch,
                )
                checkpoint_seconds = perf_counter() - checkpoint_started
                checkpoint_inventory = build_transfer_inventory(checkpoint)
            sample["full_validation"] = {
                "wall_seconds": validation_seconds,
                "coverage": validation_result,
            }
            sample["activation_diagnostic"] = {
                "wall_seconds": activation_seconds,
                "coverage": activation_coverage,
            }
            sample["logical_diagnostic"] = {
                "wall_seconds": logical_seconds,
                "coverage": logical_coverage,
                "R_model": logical["R_model"],
                "R_model_max": ceiling["R_model_max_fraction"],
            }
            sample["checkpoint"] = {
                "wall_seconds": checkpoint_seconds,
                "bytes": checkpoint_inventory["total_bytes"],
            }
            print(
                json.dumps(
                    {
                        "event": "diagnostics_complete",
                        "condition_id": condition_id,
                        "validation_seconds": validation_seconds,
                        "activation_seconds": activation_seconds,
                        "logical_seconds": logical_seconds,
                        "checkpoint_seconds": checkpoint_seconds,
                        "R_model": logical["R_model"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        samples.append(sample)
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()

    if len(initial_hashes) != 1:
        raise RuntimeError("Preflight endpoint conditions did not share one initialization hash.")
    if initial_hashes != {EXPECTED_INITIAL_PARAMETER_SHA256}:
        raise RuntimeError("Preflight initialization hash differs from the pinned CUDA identity.")
    memory_limit = 0.9 * int(device_properties.total_memory)
    healthy = all(
        len(sample["boundaries"]) == BOUNDARIES
        and all(
            not row["optimizer_step_skipped"]
            and not row["gradient_overflow"]
            and math.isfinite(float(row["task_loss"]))
            for row in sample["boundaries"]
        )
        and int(sample["peak_memory_reserved_bytes"]) <= memory_limit
        for sample in samples
    )
    a7 = next(row for row in samples if row["condition_id"] == "a7-ol1-kappa-0")
    projected = _project_science(config, samples, a7)
    result = {
        "schema_version": 1,
        "kind": "non_evidence_exact_a40_preflight",
        "status": "passed" if healthy else "failed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "runtime": realized,
        "device": {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": int(device_properties.total_memory),
            "headroom_limit_bytes": int(memory_limit),
        },
        "cache": cache_result,
        "run_code": run_code_identity(),
        "probe": {
            "condition_ids": list(PROBE_CONDITIONS),
            "boundaries_per_condition": BOUNDARIES,
            "timing_scope": "cache slicing, host-to-device staging, forward/backward, gradient processing, and optimizer update",
            "scientific_evidence": False,
        },
        "samples": samples,
        "checks": {
            "same_pinned_initial_parameter_hash": True,
            "finite_nonoverflowing_boundaries": healthy,
            "fits_with_10pct_vram_headroom": healthy,
        },
        "projection": projected,
    }
    output = RUN_DIR / "prelaunch" / "remote-preflight.json"
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(2)


def _project_science(config, samples, a7):
    per_condition = {}
    common = (
        2 * float(a7["full_validation"]["wall_seconds"])
        + float(a7["activation_diagnostic"]["wall_seconds"])
        + float(a7["logical_diagnostic"]["wall_seconds"])
        + float(a7["checkpoint"]["wall_seconds"])
    )
    teal_point_seconds = float(a7["logical_diagnostic"]["wall_seconds"])
    by_id = {sample["condition_id"]: sample for sample in samples}
    for condition in condition_specs(config):
        reference = by_id["a0-gelu"] if condition["is_control"] else by_id["a7-ol1-kappa-0"]
        training_seconds = 712 * float(reference["median_end_to_end_step_seconds"])
        low_training = 712 * float(reference["min_end_to_end_step_seconds"])
        high_training = 712 * float(reference["max_end_to_end_step_seconds"])
        posthoc = (
            10 * teal_point_seconds
            if condition["id"] in mapping(config, "teal_posthoc")["condition_ids"]
            else 0.0
        )
        per_condition[condition["id"]] = {
            "reference_condition_id": reference["condition_id"],
            "training_seconds": training_seconds,
            "training_seconds_min_step_projection": low_training,
            "training_seconds_max_step_projection": high_training,
            "common_validation_diagnostic_checkpoint_seconds": common,
            "teal_seconds_if_control": posthoc,
            "projected_total_seconds": float(reference["setup_seconds"]) + training_seconds + common + posthoc,
        }
    waves = mapping(mapping(config, "runpod"), "launch_waves")
    return {
        "method": "median of five exact end-to-end updates; full validation/diagnostics/checkpoint measured on A7",
        "conditions": per_condition,
        "projected_gpu_seconds": sum(row["projected_total_seconds"] for row in per_condition.values()),
        "projected_parallel_wave_seconds": {
            name: max(per_condition[condition_id]["projected_total_seconds"] for condition_id in ids)
            for name, ids in waves.items()
        },
        "teal_caveat": "TEAL uses the full logical-diagnostic duration as a conservative per-point proxy until one real point completes.",
        "a4_caveat": "A4 uses the measured A7 OL1 step as a conservative reference.",
        "provision_setup_transfer_not_included": True,
    }


if __name__ == "__main__":
    main()
