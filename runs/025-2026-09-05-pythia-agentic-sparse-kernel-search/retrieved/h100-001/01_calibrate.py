"""GPU-only P0 calibration. Does not launch infrastructure or select a winner."""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.nn import functional as F

from run025_common import ROOT, RUN, config, inside, read_json, record, select_conditions, sha256, verify_record, write_json

sys.path.insert(0, str(ROOT / "src"))
from sparsity_research.capture import ActivationCapture
from sparsity_research.metrics import ActivationAccumulator, weight_statistics
from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata
from measurement import numerical_gate, paired_timing, timing_summary
from p0 import Adapter, ExactTwELLLinear, extension, pack_reference


class Progress:
    def __init__(self, directory, seconds):
        self.directory, self.started = directory, time.monotonic()
        self.seconds = seconds

    def check(self):
        if time.monotonic() - self.started >= self.seconds:
            raise TimeoutError("Calibration worker time limit; remaining work is incomplete")

    def emit(self, stage, **fields):
        row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage,
               "elapsed_seconds": time.monotonic() - self.started, **fields}
        write_json(self.directory / "status.json", row)
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(row), flush=True)


def primitive_gate(directory, cfg, progress):
    generator = torch.Generator(device="cpu").manual_seed(2502)
    results = []
    shapes = sorted({(3, k, n) for d in (128, 512, 1024) for k, n in
                     ((d, 3 * d), (d, 4 * d), (4 * d, d), (d, d))} | {(3, 33, 129), (3, 32, 2048), (3, 64, 2048), (256, 512, 128)})
    for m, k, n in shapes:
        for pattern in ("zero", "signed_sparse", "full"):
            progress.check()
            x = (torch.randn(m, k, generator=generator) * .1).bfloat16()
            if pattern == "zero":
                x.zero_()
            elif pattern == "signed_sparse":
                x[torch.rand(x.shape, generator=generator) < .8] = 0
            layer = torch.nn.Linear(k, n, bias=True).eval()
            with torch.no_grad():
                layer.weight.copy_(torch.randn(n, k, generator=generator) * .1)
                layer.bias.copy_(torch.randn(n, generator=generator) * .1)
            layer = layer.to(device="cuda", dtype=torch.bfloat16)
            wrapped = ExactTwELLLinear(layer).eval()
            wrapped.mode = "p0"
            device_x = x.cuda()
            # Non-default-stream execution catches the upstream default-stream bug.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.inference_mode():
                actual = wrapped(device_x).clone()
                reference = (device_x.float() @ layer.weight.float().T + layer.bias.float()).bfloat16()
            stream.synchronize()
            packed_equal = torch.equal(wrapped._packed.cpu(), pack_reference(x))
            gate = numerical_gate(reference, actual, relative_l2=cfg["primitive_relative_l2"],
                                  atol=cfg["primitive_atol"], rtol=cfg["primitive_rtol"])
            gate["packed_exact"] = packed_equal
            gate["pass"] = gate["pass"] and packed_equal
            results.append({"m": m, "k": k, "n": n, "pattern": pattern, **gate})
            write_json(directory / "primitive-gates.json", results)
            if not gate["pass"]:
                raise RuntimeError(f"Primitive gate failed: K={k}, N={n}, {pattern}")
    progress.emit("primitive_gates_passed", cases=len(results))


def full_validation(model, adapter, tokens, cfg, progress, destination):
    """Matched full-logit comparisons; no capture hooks on the timed model here."""
    length = cfg["inputs"]["sequence_length"]
    cal = cfg["calibration"]
    blocks = len(tokens) // length
    if blocks != cfg["inputs"]["validation_blocks"] or len(tokens) % length != cfg["inputs"]["validation_tail"]:
        raise ValueError("Validation coverage differs from pinned complete-block contract")
    sums = {"native": 0., "p0": 0.}
    gates = []
    started = time.monotonic()
    with torch.inference_mode():
        for i in range(blocks):
            progress.check()
            ids = torch.tensor(np.array(tokens[i * length:(i + 1) * length]), device=next(model.parameters()).device, dtype=torch.long).unsqueeze(0)
            adapter.set_mode("native")
            reference = model(input_ids=ids, use_cache=False).logits
            adapter.set_mode("p0")
            actual = model(input_ids=ids, use_cache=False).logits
            gate = numerical_gate(reference, actual, relative_l2=cal["logit_relative_l2"], atol=cal["logit_atol"], rtol=cal["logit_rtol"])
            if not gate["finite"]:
                write_json(destination, {"complete": False, "failed_block": i, "gate": gate})
                raise RuntimeError(f"Nonfinite full-model logits on block {i}")
            for mode, logits in (("native", reference), ("p0", actual)):
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum")
                sums[mode] += float(loss)
            gates.append({"block": i, **gate})
            del actual, reference, logits, loss
            state = {"complete": i + 1 == blocks, "blocks": i + 1, "required_blocks": blocks,
                     "prediction_tokens": (i + 1) * (length - 1), "input_tokens": (i + 1) * length,
                     "validation_documents": cfg["inputs"]["validation_documents"],
                     "excluded_tail_tokens": len(tokens) % length,
                     "loss": {key: val / ((i + 1) * (length - 1)) for key, val in sums.items()},
                     "logit_gates": gates}
            write_json(destination, state)
            if not gate["pass"]:
                raise RuntimeError(f"Full-model logit gate failed on validation block {i}")
            if (i + 1) % 16 == 0 or i + 1 == blocks:
                seconds_per_block = (time.monotonic() - started) / (i + 1)
                progress.emit("validation", condition=destination.parent.name, blocks=i + 1,
                              total_blocks=blocks, loss=state["loss"], tokens_per_second=length / seconds_per_block if seconds_per_block > 0 else None,
                              remaining_seconds=(blocks - i - 1) * seconds_per_block)
    state["loss_delta"] = state["loss"]["p0"] - state["loss"]["native"]
    state["pass"] = abs(state["loss_delta"]) <= cal["validation_loss_atol"]
    write_json(destination, state)
    if not state["pass"]:
        raise RuntimeError("Complete-validation loss delta failed fixed tolerance")
    return state


def diagnostics(model, adapter, development, cal, destination, progress):
    """Calibration diagnostics use development only, after timing and quality gates.

    Canonical full-validation logical counts remain separate source artifacts.
    This capture changes A0 attention ports, so never time this model afterward.
    """
    result = {"coverage": "64 train-split development blocks; not canonical validation R_model",
              "weights": weight_statistics(model), "modes": {}}
    for mode in ("native", "p0"):
        adapter.set_mode(mode)
        accumulator = ActivationAccumulator(tuple(cal["activation_thresholds"]))
        with ActivationCapture(model, ["a", "m", "h", "z", "q_post", "k_post", "v"], torch=torch) as capture, torch.inference_mode():
            for array in development:
                progress.check()
                model(input_ids=torch.tensor(array.copy(), device="cuda", dtype=torch.long).unsqueeze(0), use_cache=False)
                accumulator.update(capture.activations, torch=torch)
                capture.clear()
        result["modes"][mode] = {"per_site_layer": accumulator.rows(), "pooled_by_site": accumulator.pooled_by_site()}
        write_json(destination, result)


def dense_profile(model, adapter, ids, destination):
    adapter.set_mode("native")
    with torch.inference_mode(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True, profile_memory=True) as profile:
        model(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
    profile.export_chrome_trace(str(destination / "dense-profile-trace.json"))
    write_json(destination / "dense-profile-operators.json", [
        {"name": event.key, "input_shapes": event.input_shapes, "calls": event.count,
         "self_cpu_us": event.self_cpu_time_total, "self_device_us": event.self_device_time_total}
        for event in profile.key_averages(group_by_input_shape=True)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True, help="New run-local artifacts directory name")
    parser.add_argument("--seconds", type=int, default=None, help="May shorten, never extend the configured pilot worker limit")
    parser.add_argument("--primitive-only", action="store_true", help="Compatibility / compute-sanitizer gate; no checkpoint timing or validation")
    args = parser.parse_args()
    cfg = config()
    seconds = cfg["calibration"]["max_worker_seconds"] if args.seconds is None else args.seconds
    if not 0 < seconds <= cfg["calibration"]["max_worker_seconds"]:
        raise ValueError("Invalid worker duration")
    directory = inside(RUN / "artifacts", args.attempt)
    directory.mkdir(parents=True, exist_ok=False)
    progress = Progress(directory, seconds)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU substitute for calibration evidence")
        expected = cfg["runtime"]
        if torch.__version__.split("+")[0] != expected["torch"] or transformers.__version__ != expected["transformers"] or np.__version__ != expected["numpy"]:
            raise RuntimeError("Runtime versions differ from pinned configuration")
        manifest = read_json(RUN / "prelaunch/input_manifest.json")
        if manifest["config_sha256"] != sha256(RUN / "config.json"):
            raise ValueError("Config changed after input inventory")
        conditions = [] if args.primitive_only else select_conditions(manifest, "calibration")
        for row in conditions:
            for item in row["files"] + row["provenance"]:
                verify_record(item)
        validation = np.memmap(verify_record(manifest["validation"]), dtype=np.int32, mode="r")
        verify_record(manifest["validation_metadata"])
        development = np.memmap(verify_record(manifest["development"]), dtype=np.int32, mode="r").reshape(-1, cfg["inputs"]["sequence_length"])
        torch.manual_seed(2503)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        properties = torch.cuda.get_device_properties(0)
        environment = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                       "scope": "primitive_only" if args.primitive_only else "eight_checkpoint_calibration",
                       "numpy": np.__version__, "cuda": torch.version.cuda, "gpu": properties.name,
                       "memory_bytes": properties.total_memory, "capability": list(torch.cuda.get_device_capability()),
                       "config_sha256": sha256(RUN / "config.json"), "timing_reference": "native SDPA, not yet final strongest-dense selection",
                       "sources": [record(path) for path in sorted(RUN.glob("*.py"))] + [record(RUN / "kernels/twell_pythia.cu")]}
        write_json(directory / "environment.json", environment)
        progress.emit("compiling", conditions=len(conditions))
        started = time.monotonic()
        extension()
        progress.emit("compiled", compile_seconds=time.monotonic() - started)
        primitive_gate(directory, cfg["calibration"], progress)
        for number, row in enumerate(conditions):
            progress.check()
            out = directory / row["id"].replace("/", "-")
            out.mkdir()
            progress.emit("loading", condition=row["id"], completed_conditions=number)
            started = time.monotonic()
            model = load_checkpoint_pythia(transformers.AutoModelForCausalLM, inside(ROOT, row["checkpoint"]), torch=torch)
            model = model.to(device="cuda", dtype=torch.bfloat16).eval()
            model.set_attn_implementation("sdpa")
            model.config.use_cache = False
            topology = topology_metadata(model)
            adapter = Adapter(model)
            inputs = [torch.tensor(array.copy(), device="cuda", dtype=torch.long).unsqueeze(0)
                      for array in development[:cfg["calibration"]["timing_inputs"]]]
            torch.cuda.reset_peak_memory_stats()
            load_seconds = time.monotonic() - started
            def save_sample(sample):
                with (out / "timing-samples.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(sample) + "\n")
            samples = paired_timing(lambda ids: model(input_ids=ids, use_cache=False).logits,
                adapter.set_mode, inputs, modes=["native", "adapter_dense", "p0"],
                passes=cfg["calibration"]["timing_passes"], warmups=cfg["calibration"]["warmups"],
                seed=cfg["inputs"]["timing_seed"], check_deadline=progress.check, sample_sink=save_sample)
            summary = timing_summary(samples)
            for item in summary.values():
                item["input_tokens_per_second"] = cfg["inputs"]["sequence_length"] * 1000 / item["median_host_ms"]
            write_json(out / "timing.json", {"samples": samples, "summary": summary, "topology": topology,
                "coverage": adapter.coverage(), "load_seconds": load_seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            progress.emit("timed", condition=row["id"], summary=summary)
            full_validation(model, adapter, validation, cfg, progress, out / "validation.json")
            dense_profile(model, adapter, inputs[0], out)
            diagnostics(model, adapter, development, cfg["calibration"], out / "diagnostics.json", progress)
            adapter.set_mode("native")
            del inputs, adapter, model
            gc.collect()
            torch.cuda.empty_cache()
            progress.emit("condition_complete", condition=row["id"], completed_conditions=number + 1, total_conditions=len(conditions))
        progress.emit("complete", scope=environment["scope"], completed_conditions=len(conditions), pass_all_gates=True)
    except BaseException as error:
        progress.emit("failed_or_incomplete", error_type=type(error).__name__, error=str(error))
        (directory / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
