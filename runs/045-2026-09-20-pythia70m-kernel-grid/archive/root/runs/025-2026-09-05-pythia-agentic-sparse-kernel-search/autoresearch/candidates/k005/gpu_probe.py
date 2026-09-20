"""Bounded K005 correctness and direct K001/K005 primitive timing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[2]
sys.path[:0] = [str(RUN), str(HERE)]
from candidate import CANDIDATE_ID, WidthSpecializedLinear, extension
from measurement import numerical_gate, paired_timing, timing_summary


def load_k001():
    path = HERE.parent / "k001/candidate.py"
    spec = importlib.util.spec_from_file_location("run025_k005_reference_k001", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


K001 = load_k001()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def primitive_shapes():
    return sorted(
        {
            (3, k, n)
            for d in (128, 512, 1024)
            for k, n in ((d, 3 * d), (d, 4 * d), (4 * d, d), (d, d))
        }
        | {(3, 33, 129), (17, 512, 128), (32, 128, 512)}
    )


def primitive_cases():
    baseline = [
        {
            "m": m,
            "k": k,
            "n": n,
            "pattern": pattern,
            "activation_scale": 0.1,
            "phase": "baseline",
        }
        for m, k, n in primitive_shapes()
        for pattern in ("zero", "signed_sparse", "full")
    ]
    stress = [
        {
            "m": m,
            "k": k,
            "n": n,
            "pattern": "signed_sparse",
            "activation_scale": scale,
            "phase": "amplitude_stress",
        }
        for m, k, n in primitive_shapes()
        for scale in (1.0, 10.0)
    ]
    return baseline + stress


def timing_cases():
    return [
        {
            "model_size": size,
            "site": site,
            "m": 2048,
            "k": k,
            "n": n,
            "requested_zero_fraction": fraction,
        }
        for size, d in (("14m", 128), ("70m", 512), ("410m", 1024))
        for site, k, n in (
            ("a", d, 3 * d),
            ("m", d, 4 * d),
            ("h", 4 * d, d),
            ("z", d, d),
        )
        for fraction in (0.0, 0.8, 0.95, 0.99)
    ]


def make_layer(k, n, generator):
    layer = torch.nn.Linear(k, n, bias=True).eval()
    with torch.no_grad():
        layer.weight.copy_(torch.randn(n, k, generator=generator) * 0.1)
        layer.bias.copy_(torch.randn(n, generator=generator) * 0.1)
    return layer.cuda().bfloat16()


def check_output(reference, actual, cfg):
    return numerical_gate(
        reference,
        actual,
        relative_l2=cfg["primitive_relative_l2"],
        atol=cfg["primitive_atol"],
        rtol=cfg["primitive_rtol"],
    )


class Recorder:
    def __init__(self, directory, seconds):
        self.directory = directory
        self.seconds = seconds
        self.started = time.monotonic()
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "failures").mkdir()

    def check(self):
        if time.monotonic() - self.started >= self.seconds:
            raise TimeoutError("K005 probe deadline")

    def write(self, name, value):
        path = self.directory / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary.replace(path)

    def emit(self, stage, **fields):
        row = {
            "candidate_id": CANDIDATE_ID,
            "stage": stage,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": time.monotonic() - self.started,
            **fields,
        }
        self.write("status.json", row)
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(row, allow_nan=False), flush=True)


def qualify(recorder, cfg):
    generator = torch.Generator().manual_seed(2530)
    results = []
    for index, case in enumerate(primitive_cases()):
        recorder.check()
        host = (
            torch.randn(case["m"], case["k"], generator=generator)
            * case["activation_scale"]
        ).bfloat16()
        if case["pattern"] == "zero":
            host.zero_()
        elif case["pattern"] == "signed_sparse":
            host[torch.rand(host.shape, generator=generator) < 0.8] = 0
        layer = make_layer(case["k"], case["n"], generator)
        wrapped = WidthSpecializedLinear(layer).eval()
        wrapped.mode = CANDIDATE_ID
        value = host.cuda()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            actual = wrapped(value).clone()
            reference = F.linear(
                value.float(), layer.weight.float(), layer.bias.float()
            ).bfloat16()
        stream.synchronize()
        gate = check_output(reference, actual, cfg)
        if not gate["pass"]:
            torch.save(
                {
                    "case": case,
                    "input": host,
                    "weight": layer.weight.detach().cpu(),
                    "bias": layer.bias.detach().cpu(),
                    "actual": actual.cpu(),
                    "reference": reference.cpu(),
                },
                recorder.directory / "failures" / f"case-{index:03d}.pt",
            )
        results.append({"case_index": index, **case, **gate})
        recorder.write("primitive-gates.json", results)
        recorder.emit(
            "primitive_case",
            completed=len(results),
            total=len(primitive_cases()),
            passed=gate["pass"],
            failures=sum(not row["pass"] for row in results),
        )
    return results


def benchmark(recorder, cfg, passes, input_count, warmups):
    generator = torch.Generator().manual_seed(2531)
    results = []
    for index, case in enumerate(timing_cases()):
        recorder.check()
        layer = make_layer(case["k"], case["n"], generator)
        wrappers = {
            "k001": K001.FusedExactLinear(layer).eval(),
            "k005": WidthSpecializedLinear(layer).eval(),
        }
        wrappers["k001"].mode = "k001"
        wrappers["k005"].mode = "k005"
        inputs = []
        actual_fractions = []
        for _ in range(input_count):
            host = (torch.randn(case["m"], case["k"], generator=generator) * 0.1).bfloat16()
            host[
                torch.rand(host.shape, generator=generator)
                < case["requested_zero_fraction"]
            ] = 0
            actual_fractions.append(float((host == 0).float().mean()))
            inputs.append(host.cuda())
        gates = []
        with torch.inference_mode():
            for input_index, value in enumerate(inputs):
                reference = F.linear(
                    value.float(), layer.weight.float(), layer.bias.float()
                ).bfloat16()
                for mode, wrapped in wrappers.items():
                    actual = wrapped(value).clone()
                    torch.cuda.synchronize()
                    gates.append(
                        {
                            "input_index": input_index,
                            "mode": mode,
                            **check_output(reference, actual, cfg),
                        }
                    )
        row = {
            "case_index": index,
            **case,
            "actual_zero_fraction": actual_fractions,
            "gates": gates,
            "qualified": all(item["pass"] for item in gates),
        }
        if row["qualified"]:
            selected = ["native"]

            def set_mode(mode):
                selected[0] = mode

            def forward(value):
                if selected[0] == "native":
                    return F.linear(value, layer.weight, layer.bias)
                return wrappers[selected[0]](value)

            def sink(sample):
                with (recorder.directory / "timing-samples.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps({"case_index": index, **sample}) + "\n")

            samples = paired_timing(
                forward,
                set_mode,
                inputs,
                modes=["native", "k001", "k005"],
                passes=passes,
                warmups=warmups,
                seed=2532 + index,
                check_deadline=recorder.check,
                sample_sink=sink,
            )
            row["summary"] = timing_summary(samples)
        else:
            row["summary"] = None
        results.append(row)
        recorder.write("timing-results.json", results)
        recorder.emit(
            "timing_case",
            completed=len(results),
            total=len(timing_cases()),
            model_size=case["model_size"],
            site=case["site"],
            zero_fraction=case["requested_zero_fraction"],
            summary=row["summary"],
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--timing-passes", type=int, default=3)
    parser.add_argument("--timing-inputs", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()
    if min(args.seconds, args.timing_passes, args.timing_inputs, args.warmups) <= 0:
        parser.error("Positive bounds required")
    recorder = Recorder(args.output.resolve(), args.seconds)
    summary = {
        "candidate_id": CANDIDATE_ID,
        "complete": False,
        "requested_primitive_cases": len(primitive_cases()),
        "requested_timing_cases": len(timing_cases()) if args.timing else 0,
    }
    try:
        cfg = json.loads((RUN / "config.json").read_text())["calibration"]
        sources = [
            HERE / "kernel.cu",
            HERE / "candidate.py",
            Path(__file__),
            HERE.parent / "k001/kernel.cu",
            HERE.parent / "k001/candidate.py",
            RUN / "measurement.py",
            RUN / "config.json",
        ]
        recorder.write(
            "source-hashes.json",
            {str(path.relative_to(RUN)): sha256(path) for path in sources},
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; CPU tests do not qualify K005")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        recorder.write(
            "environment.json",
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()),
            },
        )
        recorder.emit("compiling")
        before = time.monotonic()
        extension()
        recorder.emit("compiled", compile_seconds=time.monotonic() - before)
        primitive = qualify(recorder, cfg)
        summary["primitive_cases"] = len(primitive)
        summary["primitive_failures"] = sum(not row["pass"] for row in primitive)
        if summary["primitive_failures"]:
            raise RuntimeError("K005 primitive numerical gate failed")
        if args.timing:
            timing = benchmark(
                recorder,
                cfg,
                args.timing_passes,
                args.timing_inputs,
                args.warmups,
            )
            summary["timing_cases"] = len(timing)
            summary["timing_numerical_failures"] = sum(
                not row["qualified"] for row in timing
            )
        summary["complete"] = True
        summary["pass"] = not summary.get("timing_numerical_failures", 0)
        recorder.write("summary.json", summary)
        recorder.emit("complete", **summary)
        return 0 if summary["pass"] else 1
    except BaseException as error:
        summary.update(
            {"pass": False, "error_type": type(error).__name__, "error": str(error)}
        )
        recorder.write("summary.json", summary)
        (recorder.directory / "traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        recorder.emit("failed", **summary)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
