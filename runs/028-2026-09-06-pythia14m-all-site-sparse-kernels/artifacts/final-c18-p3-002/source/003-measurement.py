"""Sealable numerical gates and rotating-input paired timers (no tuning logic)."""
from __future__ import annotations

import math
import random
import time

import torch


def numerical_gate(reference, actual, *, relative_l2, atol, rtol):
    reference, actual = reference.float(), actual.float()
    if reference.shape != actual.shape:
        raise ValueError("Numerical comparison shape mismatch")
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
    if not finite:
        return {"pass": False, "finite": False, "relative_l2": None, "max_abs": None}
    diff = actual - reference
    rel = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(reference).clamp_min(1e-12))
    max_abs = float(diff.abs().max())
    allclose = bool(torch.all(diff.abs() <= atol + rtol * reference.abs()))
    return {"pass": rel <= relative_l2 and allclose, "finite": True,
            "relative_l2": rel, "max_abs": max_abs, "elementwise_gate": allclose}


def paired_schedule(count, passes, modes, seed):
    if count <= 0 or passes <= 0 or len(set(modes)) != len(modes) or not modes:
        raise ValueError("Invalid paired schedule")
    rng = random.Random(seed)
    result = []
    for repeat in range(passes):
        blocks = list(range(count))
        rng.shuffle(blocks)
        for index in blocks:
            order = list(modes)
            rng.shuffle(order)
            result.append((repeat, index, order))
    return result


def paired_timing(forward, set_mode, inputs, *, modes, passes, warmups, seed,
                  check_deadline=lambda: None, sample_sink=lambda row: None, cuda=True):
    """Each invocation gets its declared input. Packing + full logits are inside.

    Input staging, model loading, static transforms and mode selection are outside
    steady-state timing. Both synchronized host latency and CUDA-event latency
    are retained; primary full-model latency is synchronized host milliseconds.
    """
    samples = []
    with torch.inference_mode():
        for mode in modes:
            set_mode(mode)
            for i in range(warmups):
                check_deadline()
                forward(inputs[i % len(inputs)])
            if cuda:
                torch.cuda.synchronize()
        for repeat, index, order in paired_schedule(len(inputs), passes, modes, seed):
            for mode in order:
                check_deadline()
                set_mode(mode)
                if cuda:
                    torch.cuda.synchronize()
                    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                before = time.perf_counter()
                if cuda:
                    start.record()
                output = forward(inputs[index])
                if cuda:
                    stop.record()
                    stop.synchronize()
                milliseconds = (time.perf_counter() - before) * 1000
                samples.append({"repeat": repeat, "input_index": index, "mode": mode,
                                "host_ms": milliseconds,
                                "cuda_ms": start.elapsed_time(stop) if cuda else None})
                sample_sink(samples[-1])
                del output
    return samples


def timing_summary(samples, reference="native"):
    by_mode = {}
    for row in samples:
        by_mode.setdefault(row["mode"], {})[(row["repeat"], row["input_index"])] = row["host_ms"]
    result = {}
    for mode, paired in by_mode.items():
        if paired.keys() != by_mode[reference].keys() or any(x <= 0 or not math.isfinite(x) for x in paired.values()):
            raise ValueError("Unmatched/invalid timing samples")
        values = sorted(paired.values())
        median = (values[(len(values) - 1) // 2] + values[len(values) // 2]) / 2
        ratios = [by_mode[reference][key] / value for key, value in paired.items()]
        result[mode] = {"samples": len(values), "median_host_ms": median,
                        "paired_geomean_speedup": math.exp(sum(map(math.log, ratios)) / len(ratios)),
                        "input_tokens_per_second": None}
    return result
