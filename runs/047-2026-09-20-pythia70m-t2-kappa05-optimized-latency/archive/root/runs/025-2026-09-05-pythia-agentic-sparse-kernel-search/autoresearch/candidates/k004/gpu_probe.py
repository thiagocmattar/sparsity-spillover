"""Bounded K004 primitive qualification and Pythia-14M shape timing."""

from __future__ import annotations

import argparse
import hashlib
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
from candidate import CANDIDATE_ID, TileSkipLinear, tile_activity_reference
from measurement import numerical_gate, paired_timing, timing_summary


SHAPES_14M = {
    "a": (128, 384),
    "m": (128, 512),
    "h": (512, 128),
    "z": (128, 128),
}
STRATEGIES = ("inline", "flagged")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_layer(k, n, generator):
    layer = torch.nn.Linear(k, n, bias=True).eval()
    with torch.no_grad():
        layer.weight.copy_(torch.randn(n, k, generator=generator) * 0.1)
        layer.bias.copy_(torch.randn(n, generator=generator) * 0.1)
    return layer.cuda().bfloat16()


def make_input(m, k, pattern, generator, *, scale=0.1, tile_zero_fraction=0.0):
    value = (torch.randn(m, k, generator=generator) * scale).bfloat16()
    if pattern == "zero":
        value.zero_()
    elif pattern == "scalar_sparse":
        value[torch.rand(value.shape, generator=generator) < 0.8] = 0
    elif pattern == "tile_sparse":
        if m % 16 or k % 32:
            raise ValueError("Tile-sparse generator requires exact 16x32 coverage")
        mask = torch.rand(m // 16, k // 32, generator=generator) < tile_zero_fraction
        view = value.reshape(m // 16, 16, k // 32, 32)
        view[mask[:, None, :, None].expand_as(view)] = 0
    elif pattern != "dense":
        raise ValueError(pattern)
    return value


def gate(reference, actual, cfg):
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
            raise TimeoutError("K004 primitive probe deadline")

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


def primitive_cases():
    shapes = [
        (m, site, k, n)
        for m in (17, 32)
        for site, (k, n) in SHAPES_14M.items()
    ]
    shapes += [(3, "z", 33, 129), (17, "h", 2048, 512)]
    return [
        {
            "m": m,
            "site": site,
            "k": k,
            "n": n,
            "pattern": pattern,
            "tile_zero_fraction": 0.8 if pattern == "tile_sparse" else 0.0,
            "scale": scale,
            "strategy": strategy,
        }
        for m, site, k, n in shapes
        for pattern, scale in (
            ("zero", 0.1),
            ("dense", 0.1),
            ("scalar_sparse", 0.1),
            ("tile_sparse", 0.1),
            ("scalar_sparse", 10.0),
        )
        if pattern != "tile_sparse" or (m % 16 == 0 and k % 32 == 0)
        for strategy in STRATEGIES
    ]


def qualify(recorder, cfg):
    generator = torch.Generator().manual_seed(2520)
    rows = []
    for index, case in enumerate(primitive_cases()):
        recorder.check()
        layer = make_layer(case["k"], case["n"], generator)
        wrapped = TileSkipLinear(
            layer, site=case["site"], strategy=case["strategy"]
        ).eval()
        wrapped.mode = CANDIDATE_ID
        host = make_input(
            case["m"],
            case["k"],
            case["pattern"],
            generator,
            scale=case["scale"],
            tile_zero_fraction=case["tile_zero_fraction"],
        )
        device = host.cuda()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            actual = wrapped(device).clone()
            reference = F.linear(
                device.float(), layer.weight.float(), layer.bias.float()
            ).bfloat16()
        stream.synchronize()
        result = gate(reference, actual, cfg)
        if not result["pass"]:
            failure = recorder.directory / "failures" / f"case-{index:03d}.pt"
            torch.save(
                {
                    "input": host,
                    "weight": layer.weight.detach().cpu(),
                    "bias": layer.bias.detach().cpu(),
                    "actual": actual.cpu(),
                    "reference": reference.cpu(),
                    "case": case,
                },
                failure,
            )
        row = {
            "case_index": index,
            **case,
            "zero_count": int((host == 0).sum()),
            "elements": host.numel(),
            "active_tile_count": int(tile_activity_reference(host).sum()),
            "tile_count": tile_activity_reference(host).numel(),
            "nondefault_stream": True,
            **result,
        }
        rows.append(row)
        recorder.write("primitive-gates.json", rows)
        recorder.emit(
            "primitive_case",
            completed=len(rows),
            total=len(primitive_cases()),
            passed=result["pass"],
            failures=sum(not item["pass"] for item in rows),
        )
    return rows


def timing_cases():
    return [
        {"site": site, "m": 2048, "k": k, "n": n, "tile_zero_fraction": fraction}
        for site, (k, n) in SHAPES_14M.items()
        for fraction in (0.0, 0.5, 0.8, 0.95)
    ]


def benchmark(recorder, cfg, passes, input_count, warmups):
    generator = torch.Generator().manual_seed(2521)
    results = []
    for index, case in enumerate(timing_cases()):
        recorder.check()
        layer = make_layer(case["k"], case["n"], generator)
        wrappers = {
            strategy: TileSkipLinear(
                layer, site=case["site"], strategy=strategy
            ).eval()
            for strategy in STRATEGIES
        }
        for wrapped in wrappers.values():
            wrapped.mode = CANDIDATE_ID
        inputs = [
            make_input(
                case["m"],
                case["k"],
                "tile_sparse" if case["tile_zero_fraction"] else "dense",
                generator,
                tile_zero_fraction=case["tile_zero_fraction"],
            ).cuda()
            for _ in range(input_count)
        ]
        gates = []
        with torch.inference_mode():
            for input_index, value in enumerate(inputs):
                reference = F.linear(
                    value.float(), layer.weight.float(), layer.bias.float()
                ).bfloat16()
                for strategy, wrapped in wrappers.items():
                    actual = wrapped(value).clone()
                    torch.cuda.synchronize()
                    gates.append(
                        {
                            "input_index": input_index,
                            "mode": strategy,
                            **gate(reference, actual, cfg),
                        }
                    )
        qualified = all(item["pass"] for item in gates)
        row = {
            "case_index": index,
            **case,
            "actual_tile_zero_fraction": [
                1 - float(tile_activity_reference(value.cpu()).float().mean())
                for value in inputs
            ],
            "gates": gates,
            "qualified": qualified,
        }
        if qualified:
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
                modes=["native", *STRATEGIES],
                passes=passes,
                warmups=warmups,
                seed=2522 + index,
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
            site=case["site"],
            tile_zero_fraction=case["tile_zero_fraction"],
            qualified=qualified,
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
            HERE / "candidate.py",
            HERE / "kernels.py",
            Path(__file__),
            RUN / "measurement.py",
            RUN / "config.json",
        ]
        recorder.write(
            "source-hashes.json",
            {str(path.relative_to(RUN)): sha256(path) for path in sources},
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; CPU tests do not qualify K004")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        import triton

        recorder.write(
            "environment.json",
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "triton": triton.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()),
                "allow_tf32": False,
                "allow_bf16_reduced_precision_reduction": False,
            },
        )
        primitive = qualify(recorder, cfg)
        summary["primitive_cases"] = len(primitive)
        summary["primitive_failures"] = sum(not row["pass"] for row in primitive)
        if summary["primitive_failures"]:
            raise RuntimeError("K004 primitive numerical gate failed")
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
