"""Bounded K001 primitive qualification and optional synthetic linear timings.

No infrastructure API, model checkpoint, data cache, or evaluator mutation.
Default: 48 original cases plus 32 signed activation-amplitude stress cases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[2]
sys.path.insert(0, str(RUN))
sys.path.insert(0, str(HERE))
from measurement import numerical_gate, paired_timing, timing_summary
from p0 import ExactTwELLLinear
from candidate import CANDIDATE_ID, FusedExactLinear, extension


def primitive_shapes():
    return sorted({(3, k, n) for d in (128, 512, 1024) for k, n in
                   ((d, 3 * d), (d, 4 * d), (4 * d, d), (d, d))}
                  | {(3, 33, 129), (3, 32, 2048), (3, 64, 2048), (256, 512, 128)})


def primitive_cases():
    # The first 48 cases preserve original order and seed-driven tensors.
    baseline = [{"m": m, "k": k, "n": n, "pattern": pattern,
                 "activation_scale": .1, "phase": "original_48"}
                for m, k, n in primitive_shapes()
                for pattern in ("zero", "signed_sparse", "full")]
    stress = [{"m": m, "k": k, "n": n, "pattern": "signed_sparse",
               "activation_scale": scale, "phase": "amplitude_stress"}
              for m, k, n in primitive_shapes() for scale in (1., 10.)]
    return baseline + stress


def timing_cases():
    return [{"model_size": size, "site": site, "m": 2048, "k": k, "n": n,
             "requested_zero_fraction": zero_fraction}
            for size, d in (("14m", 128), ("70m", 512), ("410m", 1024))
            for site, k, n in (("a", d, 3 * d), ("m", d, 4 * d),
                               ("h", 4 * d, d), ("z", d, d))
            for zero_fraction in (0., .5, .9, .99)]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Recorder:
    def __init__(self, directory, seconds):
        self.directory, self.seconds, self.started = directory, seconds, time.monotonic()
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "outputs").mkdir()

    def write(self, name, value):
        destination = self.directory / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(destination)

    def emit(self, stage, **fields):
        row = {"candidate_id": CANDIDATE_ID, "stage": stage,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "elapsed_seconds": time.monotonic() - self.started, **fields}
        self.write("status.json", row)
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(row, allow_nan=False), flush=True)

    def check(self):
        if time.monotonic() - self.started >= self.seconds:
            raise TimeoutError("K001 probe time limit; remaining cases incomplete")


def make_layer(k, n, generator):
    layer = torch.nn.Linear(k, n, bias=True).eval()
    with torch.no_grad():
        layer.weight.copy_(torch.randn(n, k, generator=generator) * .1)
        layer.bias.copy_(torch.randn(n, generator=generator) * .1)
    return layer.to(device="cuda", dtype=torch.bfloat16)


def check_output(reference, actual, cfg):
    return numerical_gate(reference, actual,
        relative_l2=cfg["primitive_relative_l2"],
        atol=cfg["primitive_atol"], rtol=cfg["primitive_rtol"])


def qualify(recorder, cfg):
    generator = torch.Generator(device="cpu").manual_seed(2502)
    results = []
    for case_index, case in enumerate(primitive_cases()):
        recorder.check()
        m, k, n = (case[key] for key in ("m", "k", "n"))
        x = (torch.randn(m, k, generator=generator) * case["activation_scale"]).bfloat16()
        if case["pattern"] == "zero":
            x.zero_()
        elif case["pattern"] == "signed_sparse":
            x[torch.rand(x.shape, generator=generator) < .8] = 0
        layer = make_layer(k, n, generator)
        wrapped = FusedExactLinear(layer).eval()
        wrapped.mode = CANDIDATE_ID
        device_x = x.cuda()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            actual = wrapped(device_x).clone()
            reference = (device_x.float() @ layer.weight.float().T + layer.bias.float()).bfloat16()
        stream.synchronize()
        gate = check_output(reference, actual, cfg)
        output_name = f"outputs/case-{case_index:03d}.pt"
        payload = {"actual": actual.cpu(), "reference": reference.cpu()}
        if not gate["pass"]:
            # Full failing operands retained; passing inputs reproduce from seed/order.
            payload.update({"input": x, "weight": layer.weight.detach().cpu(),
                            "bias": layer.bias.detach().cpu()})
        torch.save(payload, recorder.directory / output_name)
        row = {"case_index": case_index, **case, "weight_scale": .1,
               "bias_scale": .1, "nondefault_stream": True,
               "zero_count": int((x == 0).sum()), "element_count": x.numel(),
               "output_file": output_name, "output_sha256": sha256(recorder.directory / output_name),
               "representation_gate": "no materialized packing; separate exact compaction CPU tests",
               **gate}
        results.append(row)
        recorder.write("primitive-gates.json", results)
        recorder.emit("primitive_case", completed=len(results), total=len(primitive_cases()),
                      case_index=case_index, phase=case["phase"], passed=gate["pass"],
                      failures=sum(not result["pass"] for result in results))
    return results


def benchmark(recorder, cfg, *, passes, input_count, warmups):
    generator = torch.Generator(device="cpu").manual_seed(2510)
    results = []
    for case_index, case in enumerate(timing_cases()):
        recorder.check()
        m, k, n = (case[key] for key in ("m", "k", "n"))
        layer = make_layer(k, n, generator)
        wrappers = {"p0": ExactTwELLLinear(layer).eval(),
                    CANDIDATE_ID: FusedExactLinear(layer).eval()}
        wrappers["p0"].mode = "p0"
        wrappers[CANDIDATE_ID].mode = CANDIDATE_ID
        inputs, counts = [], []
        for _ in range(input_count):
            value = (torch.randn(m, k, generator=generator) * .1).bfloat16()
            value[torch.rand(value.shape, generator=generator) < case["requested_zero_fraction"]] = 0
            counts.append({"zero_count": int((value == 0).sum()), "element_count": value.numel()})
            inputs.append(value.cuda())
        gates = []
        with torch.inference_mode():
            for input_index, value in enumerate(inputs):
                recorder.check()
                reference = (value.float() @ layer.weight.float().T + layer.bias.float()).bfloat16()
                for mode, wrapped in wrappers.items():
                    actual = wrapped(value).clone()
                    torch.cuda.synchronize()
                    gate = check_output(reference, actual, cfg)
                    gates.append({"input_index": input_index, "mode": mode, **gate})
                    if not gate["pass"]:
                        torch.save({"actual": actual.cpu(), "reference": reference.cpu(),
                                    "input": value.cpu(), "weight": layer.weight.detach().cpu(),
                                    "bias": layer.bias.detach().cpu()}, recorder.directory / "outputs" /
                                   f"timing-{case_index:03d}-{input_index}-{mode}-failed.pt")
        row = {"case_index": case_index, **case, "actual_counts": counts,
               "gates": gates, "qualified": all(gate["pass"] for gate in gates)}
        if row["qualified"]:
            selected = ["native"]

            def set_mode(mode):
                selected[0] = mode

            def forward(value):
                return F.linear(value, layer.weight, layer.bias) if selected[0] == "native" else wrappers[selected[0]](value)

            def sample_sink(sample):
                with (recorder.directory / "timing-samples.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"case_index": case_index, **sample}, allow_nan=False) + "\n")

            samples = paired_timing(forward, set_mode, inputs, modes=["native", "p0", CANDIDATE_ID],
                passes=passes, warmups=warmups, seed=2504 + case_index,
                check_deadline=recorder.check, sample_sink=sample_sink)
            row["summary"] = timing_summary(samples)
            row["sampling"] = {"passes": passes, "inputs": input_count, "warmups": warmups,
                               "primary": "synchronized host milliseconds; sparse dispatch included"}
        else:
            row["summary"] = None
            row["timing_skipped_reason"] = "unchanged primitive numerical gate failed"
        results.append(row)
        recorder.write("timing-results.json", results)
        recorder.emit("timing_case", completed=len(results), total=len(timing_cases()),
                      case_index=case_index, model_size=case["model_size"], site=case["site"],
                      zero_fraction=case["requested_zero_fraction"], qualified=row["qualified"],
                      summary=row["summary"])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New immutable-attempt output directory")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--timing-passes", type=int, default=3)
    parser.add_argument("--timing-inputs", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()
    if min(args.seconds, args.timing_passes, args.timing_inputs, args.warmups) <= 0:
        parser.error("Durations and sampling counts must be positive")
    recorder = Recorder(args.output.resolve(), args.seconds)
    summary = {"candidate_id": CANDIDATE_ID, "complete": False,
               "requested_primitive_cases": len(primitive_cases()),
               "requested_timing_cases": len(timing_cases()) if args.timing else 0}
    try:
        cfg = json.loads((RUN / "config.json").read_text())["calibration"]
        sources = [HERE / "kernel.cu", HERE / "candidate.py", Path(__file__),
                   RUN / "measurement.py", RUN / "config.json", RUN / "p0.py", RUN / "kernels/twell_pythia.cu"]
        recorder.write("source-hashes.json", {str(path.relative_to(RUN)): sha256(path) for path in sources})
        recorder.write("fixed-gates.json", {key: cfg[key] for key in
            ("primitive_relative_l2", "primitive_atol", "primitive_rtol")})
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required; CPU/static tests do not qualify this probe")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        recorder.write("environment.json", {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "allow_tf32": False, "allow_bf16_reduced_precision_reduction": False,
            "timing_enabled": args.timing, "max_seconds": args.seconds})
        recorder.emit("compiling")
        before = time.monotonic()
        extension()
        recorder.emit("compiled", compile_seconds=time.monotonic() - before)
        primitive = qualify(recorder, cfg)
        summary["primitive_cases"] = len(primitive)
        summary["primitive_failures"] = sum(not row["pass"] for row in primitive)
        if summary["primitive_failures"]:
            raise RuntimeError("Primitive numerical qualification failed; timing intentionally skipped")
        if args.timing:
            timing = benchmark(recorder, cfg, passes=args.timing_passes,
                               input_count=args.timing_inputs, warmups=args.warmups)
            summary["timing_cases"] = len(timing)
            summary["timing_numerical_failures"] = sum(not row["qualified"] for row in timing)
        summary["complete"] = True
        summary["pass"] = not summary.get("timing_numerical_failures", 0)
        recorder.write("summary.json", summary)
        recorder.emit("complete", **summary)
        return 0 if summary["pass"] else 1
    except Exception as error:
        summary.update({"pass": False, "error_type": type(error).__name__, "error": str(error)})
        recorder.write("summary.json", summary)
        (recorder.directory / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        recorder.emit("failed", **summary)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
