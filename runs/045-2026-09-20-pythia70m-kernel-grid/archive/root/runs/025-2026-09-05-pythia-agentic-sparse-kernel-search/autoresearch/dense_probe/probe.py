"""Development-only matched dense baseline probe; no infrastructure operations.

Inputs are copied into the same persistent buffer before EVERY mode's timer.
The complete forward, including gates, SDPA and dense LM head, stays inside.
This diagnostic does not change Run 025's sealed evaluator or claim a winner.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1]
ROOT = RUN.parents[1]
sys.path[:0] = [str(RUN), str(ROOT / "src")]
from measurement import numerical_gate, paired_schedule, timing_summary
from run025_common import config, inside, read_json, record, sha256, verify_record, write_json
from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata


class DenseRunner:
    """A callable over one persistent buffer, with optional native graph capture."""

    def __init__(self, forward, static_input, mode, *, cuda=True):
        self.forward, self.static_input, self.mode = forward, static_input, mode
        self.cuda, self.graph, self.output = cuda, None, None
        self.setup_seconds = 0.0

    def prepare(self):
        started = time.perf_counter()
        if self.mode == "graph":
            if not self.cuda:
                raise ValueError("CUDA graph capture requires CUDA")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.inference_mode():
                for _ in range(3):
                    self.forward(self.static_input)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph), torch.inference_mode():
                self.output = self.forward(self.static_input)
        elif self.mode in ("compile_reduce_overhead", "compile_max_autotune"):
            mode = {"compile_reduce_overhead": "reduce-overhead",
                    "compile_max_autotune": "max-autotune"}[self.mode]
            # A graph break is a visible unsupported outcome, never a silent
            # eager fallback described as a fully compiled graph.
            self.forward = torch.compile(self.forward, mode=mode, fullgraph=True, dynamic=False)
            with torch.inference_mode():
                for _ in range(3):
                    self()
        elif self.mode != "native":
            raise ValueError(f"Unknown dense mode: {self.mode}")
        if self.cuda:
            torch.cuda.synchronize()
        self.setup_seconds = time.perf_counter() - started

    def stage(self, ids):
        self.static_input.copy_(ids)

    def __call__(self):
        if self.graph is not None:
            self.graph.replay()
            return self.output
        if self.mode.startswith("compile_"):
            torch.compiler.cudagraph_mark_step_begin()
        return self.forward(self.static_input)


def paired_probe(runners, inputs, *, passes, seed, sink=lambda row: None, cuda=True):
    """Separate equal staging from resident-input full-logit timing."""
    samples = []
    with torch.inference_mode():
        for repeat, index, order in paired_schedule(len(inputs), passes, list(runners), seed):
            for mode in order:
                runner = runners[mode]
                before_staging = time.perf_counter()
                runner.stage(inputs[index])
                if cuda:
                    torch.cuda.synchronize()
                    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                staging_ms = (time.perf_counter() - before_staging) * 1000
                before = time.perf_counter()
                if cuda:
                    start.record()
                output = runner()
                if cuda:
                    stop.record()
                    stop.synchronize()
                elapsed = (time.perf_counter() - before) * 1000
                row = {"repeat": repeat, "input_index": index, "mode": mode,
                       "host_ms": elapsed, "cuda_ms": start.elapsed_time(stop) if cuda else None,
                       "staging_ms_excluded": staging_ms, "output_shape": list(output.shape)}
                samples.append(row)
                sink(row)
                del output
    return samples


def compare_inputs(runners, inputs, calibration, *, check=lambda: None, progress=lambda **fields: None):
    """Same sealed numerical bounds as P0; pooled shifted-label task loss."""
    totals = {mode: 0. for mode in runners}
    gates = {mode: [] for mode in runners if mode != "native"}
    prediction_tokens = 0
    with torch.inference_mode():
        for index, ids in enumerate(inputs):
            check()
            runners["native"].stage(ids)
            reference = runners["native"]()
            for mode, runner in runners.items():
                if mode == "native":
                    actual = reference
                else:
                    runner.stage(ids)
                    actual = runner()
                    gate = numerical_gate(reference, actual,
                        relative_l2=calibration["logit_relative_l2"],
                        atol=calibration["logit_atol"], rtol=calibration["logit_rtol"])
                    gates[mode].append({"input_index": index, **gate})
                loss = F.cross_entropy(actual[:, :-1].float().reshape(-1, actual.shape[-1]),
                                       ids[:, 1:].reshape(-1), reduction="sum")
                totals[mode] += float(loss)
                del actual, loss
            prediction_tokens += ids.shape[0] * (ids.shape[1] - 1)
            del reference
            if (index + 1) % 16 == 0:
                progress(blocks=index + 1, prediction_tokens=prediction_tokens,
                         loss={mode: value / prediction_tokens for mode, value in totals.items()})
    losses = {mode: value / prediction_tokens for mode, value in totals.items()}
    valid = {mode: all(row["pass"] for row in rows)
                    and abs(losses[mode] - losses["native"]) <= calibration["validation_loss_atol"]
             for mode, rows in gates.items()}
    valid["native"] = bool(np.isfinite(losses["native"]))
    return {"prediction_tokens": prediction_tokens, "loss": losses, "gates": gates,
            "pass": valid, "loss_delta": {mode: value - losses["native"] for mode, value in losses.items()}}


def attention_profile(forward, ids, destination):
    """Read-only operator evidence; no activation hooks or topology rewrites."""
    with torch.inference_mode(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        forward(ids)
        torch.cuda.synchronize()
    rows = [{"key": event.key, "count": event.count,
             "self_cpu_us": event.self_cpu_time_total,
             "self_device_us": event.self_device_time_total}
            for event in profiler.key_averages()]
    write_json(destination, {"all_operators": rows,
        "attention_operators": [row for row in rows if any(
            word in row["key"].lower() for word in ("attention", "flash", "softmax", "bmm"))]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", default="14m/a1h")
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--modes", nargs="+", default=["native", "graph", "compile_reduce_overhead"])
    parser.add_argument("--inputs", type=int, default=16)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=1200)
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--sdpa-backend", choices=["auto", "flash"], default="auto")
    args = parser.parse_args()
    if not 1 <= args.inputs <= 64 or args.passes < 1 or args.seconds < 1:
        raise ValueError("Invalid counts or duration")
    if len(set(args.modes)) != len(args.modes) or "native" not in args.modes:
        raise ValueError("Modes must be unique and include the native reference")
    destination = inside(HERE / "artifacts", args.attempt)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def emit(stage, **fields):
        row = {"stage": stage, "elapsed_seconds": time.monotonic() - started, **fields}
        write_json(destination / "status.json", row)
        with (destination / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)

    def check():
        if time.monotonic() - started > args.seconds:
            raise TimeoutError("Probe time budget exceeded (external process timeout must bound compilation)")

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; CPU runs are not performance evidence")
        cfg = config()
        for name, actual in (("torch", torch.__version__.split("+")[0]),
                             ("transformers", transformers.__version__), ("numpy", np.__version__)):
            if cfg["runtime"][name] != actual:
                raise RuntimeError(f"Pinned {name} mismatch")
        manifest = read_json(RUN / "prelaunch/input_manifest.json")
        if manifest["config_sha256"] != sha256(RUN / "config.json"):
            raise ValueError("Manifest config identity mismatch")
        row = next(row for row in manifest["checkpoints"] if row["id"] == args.condition)
        if row["partition"] != "development":
            raise ValueError("Probe must not query an untuned interior-kappa checkpoint")
        for item in row["files"] + row["provenance"]:
            verify_record(item)
        development = np.memmap(verify_record(manifest["development"]), dtype=np.int32, mode="r")
        development = development.reshape(-1, cfg["inputs"]["sequence_length"])
        torch.manual_seed(2503)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        before_load = time.monotonic()
        model = load_checkpoint_pythia(transformers.AutoModelForCausalLM, inside(ROOT, row["checkpoint"]), torch=torch)
        model = model.to(device="cuda", dtype=torch.bfloat16).eval()
        model.set_attn_implementation("sdpa")
        model.config.use_cache = False
        inputs = [torch.tensor(array.copy(), device="cuda", dtype=torch.long).unsqueeze(0)
                  for array in development[:args.inputs]]
        static_input = inputs[0].clone()
        forward = lambda ids: model(input_ids=ids, use_cache=False, output_attentions=False).logits
        write_json(destination / "manifest.json", {"condition": row["id"], "arguments": vars(args),
            "precision": "BF16 parameters and full logits; TF32 and reduced BF16 reduction disabled",
            "topology": topology_metadata(model), "checkpoint_files": row["files"],
            "input_cache": manifest["development"], "config_sha256": sha256(RUN / "config.json"),
            "sources": [record(Path(__file__))], "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_properties(0).name,
            "capability": list(torch.cuda.get_device_capability()),
            "load_seconds": time.monotonic() - before_load,
            "input_sha256": [hashlib.sha256(array.tobytes()).hexdigest() for array in development[:args.inputs]],
            "gate_constants": cfg["calibration"], "coverage": "Development training inputs, not final evaluation"})
        runners, setup = {}, {}
        backend = contextlib.nullcontext()
        if args.sdpa_backend == "flash":
            from torch.nn.attention import SDPBackend, sdpa_kernel
            backend = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        with backend, torch.inference_mode():
            for mode in args.modes:
                check()
                before_setup = time.monotonic()
                emit("preparing", mode=mode)
                try:
                    runner = DenseRunner(forward, static_input, mode)
                    runner.prepare()
                    # Warm every actual rotating input, not just the first shape.
                    for ids in inputs:
                        runner.stage(ids)
                        output = runner()
                        del output
                    torch.cuda.synchronize()
                    runners[mode] = runner
                    setup[mode] = {"supported": True, "setup_seconds": runner.setup_seconds,
                                   "setup_and_all_input_warmup_seconds": time.monotonic() - before_setup}
                except Exception as error:
                    setup[mode] = {"supported": False, "elapsed_seconds": time.monotonic() - before_setup,
                                   "error": str(error), "traceback": traceback.format_exc()}
                    if mode == "native":
                        raise
                write_json(destination / "setup.json", setup)
            check()
            quality = compare_inputs(runners, inputs, cfg["calibration"], check=check)
            write_json(destination / "development-quality.json", quality)
            eligible = {mode: runner for mode, runner in runners.items() if quality["pass"][mode]}
            if "native" not in eligible:
                raise RuntimeError("Native reference has nonfinite development loss")
            def sink(sample):
                check()
                with (destination / "timing-samples.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(sample) + "\n")
            samples = paired_probe(eligible, inputs, passes=args.passes,
                                   seed=cfg["inputs"]["timing_seed"], sink=sink)
            summary = timing_summary(samples)
            winner = min(summary, key=lambda mode: summary[mode]["median_host_ms"])
            write_json(destination / "timing.json", {"summary": summary, "development_fastest": winner,
                "eligibility": "Development logit and loss gates only; final full validation still required",
                "timer": "resident input; equal staging excluded; full logits and dynamic gates included"})
            emit("timed", summary=summary, provisional_fastest=winner)
            attention_profile(forward, inputs[0], destination / "native-attention-profile.json")
            if args.full_validation:
                tokens = np.memmap(verify_record(manifest["validation"]), dtype=np.int32, mode="r")
                length = cfg["inputs"]["sequence_length"]
                blocks, tail = divmod(len(tokens), length)
                if blocks != 338 or tail != 1444:
                    raise ValueError("Complete validation coverage mismatch")
                val_inputs = (torch.tensor(np.array(tokens[i * length:(i + 1) * length]),
                             device="cuda", dtype=torch.long).unsqueeze(0) for i in range(blocks))
                validation = compare_inputs(eligible, val_inputs, cfg["calibration"], check=check,
                    progress=lambda **fields: emit("full_validation", **fields))
                validation.update({"blocks": blocks, "documents": 500, "excluded_tail": tail,
                                   "input_tokens": blocks * length, "validation_cache": manifest["validation"]})
                write_json(destination / "full-validation.json", validation)
            emit("complete", valid_development_modes=list(eligible))
    except BaseException as error:
        emit("failed_or_incomplete", error=str(error))
        (destination / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        gc.collect()


if __name__ == "__main__":
    main()
