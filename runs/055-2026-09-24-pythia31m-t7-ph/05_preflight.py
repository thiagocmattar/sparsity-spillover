"""Bounded non-evidence calibration, including all recurring optimizer work."""
import argparse
from contextlib import nullcontext
import gc
import json
import platform
import statistics
import sys
from time import perf_counter
import traceback
from run_config import (RUN_DIR, load_config, load_verified_caches, build_schedule,
    condition_specs, resolved_condition_config, seed_everything, parameter_sha256,
    microbatches_for_step, run_code_identity, write_json)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True)
    p.add_argument("--attempt", required=True)
    p.add_argument("--boundaries", type=int, default=6)
    p.add_argument("--diagnostics", action="store_true")
    args = p.parse_args()
    if not args.attempt.replace("-", "").isalnum() or args.boundaries < 2:
        raise ValueError("Alphanumeric attempt and at least two boundaries required")
    out = RUN_DIR/"prelaunch"/("calibration-"+args.attempt)
    out.mkdir(exist_ok=False)
    result = dict(kind="non_evidence_real_data_calibration", status="running", arguments=vars(args))
    write_json(out/"result.json", result)
    started = perf_counter()
    try:
        import numpy as np
        import torch
        import transformers
        from sparsity_research.capture import ActivationCapture
        from sparsity_research.pressure import parse_pressure_config
        from sparsity_research.optimization import set_learning_rate
        from model_factory import build_pinned_run055_model
        from initialization_artifact import load_pinned_initialization
        from optimizer_boundary import build_recipe_adamw, DynamicLossScaler, run_recipe_boundary, recipe_learning_rate
        from training import timed_validation, _save_checkpoint
        from diagnostics import activation_diagnostic_validation, logical_product_validation
        c = load_config()
        t = c["training"]
        actual_runtime = dict(python=f"{sys.version_info.major}.{sys.version_info.minor}",
            torch=torch.__version__.split("+")[0], transformers=transformers.__version__, cuda_runtime=torch.version.cuda)
        if actual_runtime != c["runtime"]:
            raise RuntimeError(f"Pinned runtime mismatch: {actual_runtime}")
        result.update(config=c, run_code=run_code_identity(), runtime=dict(python=platform.python_version(),
            torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda),
            gpu=torch.cuda.get_device_name())
        train, val, tm, vm, cache_seconds = load_verified_caches(c, np=np)
        starts, schedule_hash, schedule = build_schedule(c, tm, np=np)
        result.update(schedule_hash=schedule_hash, schedule=schedule, cache_verification_seconds=cache_seconds)
        row = next(r for r in condition_specs(c) if r["id"] == args.condition)
        resolved = resolved_condition_config(c, row)
        seed_everything(torch, c["seeds"]["model"])
        setup = perf_counter()
        model = build_pinned_run055_model(resolved["model"], device=torch.device("cpu"),
                                         torch=torch, auto_model=transformers.AutoModelForCausalLM)
        result["initialization"] = load_pinned_initialization(model, torch=torch)
        result["initial_parameter_sha256"] = parameter_sha256(model)
        model.to("cuda", dtype=torch.float32)
        result["model_setup_seconds"] = perf_counter()-setup
        optimizer, _ = build_recipe_adamw(model, t, torch=torch)
        scaler = DynamicLossScaler(scale=2.**t["initial_loss_scale_power"],
            growth_interval=t["loss_scale_window"], hysteresis=t["loss_scale_hysteresis"],
            minimum_scale=t["minimum_loss_scale"])
        pressure = parse_pressure_config(resolved["activation_pressure"])
        torch.cuda.reset_peak_memory_stats()
        times, health = [], []
        context = ActivationCapture(model, ["h"], torch=torch) if pressure.enabled else nullcontext(None)
        with context as capture:
            for step in range(1, args.boundaries+1):
                model.train()
                torch.cuda.synchronize()
                tick = perf_counter()
                batches = microbatches_for_step(train, starts[step-1], block_size=2048,
                    device=torch.device("cuda"), torch=torch, np=np)
                set_learning_rate(optimizer, recipe_learning_rate(step, peak=t["peak_learning_rate"],
                    max_steps=t["max_steps"], warmup_fraction=t["warmup_fraction"], minimum=t["minimum_learning_rate"]))
                metrics = run_recipe_boundary(model=model, optimizer=optimizer, batches=batches,
                    pressure=pressure, capture=capture, loss_scaler=scaler,
                    gradient_clip_norm=t["gradient_clip_norm"], torch=torch, device=torch.device("cuda"))
                torch.cuda.synchronize()
                elapsed = perf_counter()-tick
                if metrics["optimizer_step_skipped"]:
                    raise RuntimeError("Calibration skipped an optimizer update")
                if pressure.enabled and metrics["pressure_capture_tensor_count"] != 6:
                    raise RuntimeError("Missing h pressure tensors")
                if step > 1:
                    times.append(elapsed)
                health.append(metrics)
                result.update(boundary_health=health, timed_boundary_seconds=times)
                write_json(out/"result.json", result)
                print(json.dumps(dict(phase="boundary", condition=args.condition, step=step,
                    loss=metrics["task_loss"], seconds=elapsed, tokens_per_second=2097152/elapsed)), flush=True)
        tick = perf_counter()
        _save_checkpoint(model=model, optimizer=optimizer, scaler=scaler, step=args.boundaries,
            include_optimizer=True, root=out/"checkpoint", schedule_hash=schedule_hash, torch=torch)
        result["checkpoint_seconds"] = perf_counter()-tick
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
        result["validation"], result["validation_seconds"] = timed_validation(
            model=model, tokens=val, config=c, torch=torch, np=np)
        if args.diagnostics:
            tick = perf_counter()
            coverage, stats = activation_diagnostic_validation(model=model, tokens=val, config=c, torch=torch, np=np)
            result["activation_seconds"] = perf_counter()-tick
            write_json(out/"activation.json", dict(coverage=coverage, statistics=stats))
            tick = perf_counter()
            coverage, measured, ceiling = logical_product_validation(model=model, tokens=val, config=c, torch=torch, np=np)
            result["logical_seconds"] = perf_counter()-tick
            write_json(out/"logical.json", dict(coverage=coverage, measured=measured, architecture_maximum=ceiling))
        result.update(status="passed", median_boundary_seconds=statistics.median(times),
            min_boundary_seconds=min(times), max_boundary_seconds=max(times),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            free_memory_bytes=torch.cuda.mem_get_info()[0])
        if result["peak_reserved_bytes"] > .9*result["total_memory_bytes"]:
            raise RuntimeError("Less than ten percent VRAM headroom")
    except BaseException as error:
        result.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result["elapsed_seconds"] = perf_counter()-started
        write_json(out/"result.json", result)


if __name__ == "__main__":
    main()
