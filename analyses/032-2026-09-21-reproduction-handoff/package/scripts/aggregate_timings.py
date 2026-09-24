"""Pool three qualified fresh processes without mixing checkpoint identities."""

import argparse
import json
import math
from pathlib import Path


def aggregate(folders):
    results = [json.loads((p / "result.json").read_text()) for p in folders]
    if len(results) != 3 or {r["replicate"] for r in results} != {1, 2, 3}:
        raise ValueError("Exactly replicates 1,2,3 required")
    for key in (
        "weight_sha256",
        "backend",
        "control",
        "operation_mode",
        "hz_mode",
        "gpu",
        "runtime",
    ):
        if any(r.get(key) != results[0].get(key) for r in results):
            raise ValueError(f"Mixed {key}")
    if not all(r.get("qualified") and r["status"] == "qualified" for r in results):
        raise ValueError("A process did not qualify")
    all_samples = []
    process_means = []
    for p in folders:
        rows = [json.loads(x) for x in (p / "timing.jsonl").read_text().splitlines()]
        means = {}
        for mode in ("native", "candidate"):
            selected = [r for r in rows if r["mode"] == mode]
            if len(selected) != 448 or {
                (r["repeat"], r["input_index"]) for r in selected
            } != {(repeat, i) for repeat in range(7) for i in range(64)}:
                raise ValueError("Incomplete/duplicate timing samples")
            times = [r["host_ms"] for r in selected]
            if not all(math.isfinite(x) and x > 0 for x in times):
                raise ValueError("Invalid host timing")
            means[mode] = math.exp(math.fsum(map(math.log, times)) / 448)
        process_means.append(means)
        all_samples += rows
    pooled = {
        mode: math.exp(
            math.fsum(math.log(r["host_ms"]) for r in all_samples if r["mode"] == mode)
            / 1344
        )
        for mode in ("native", "candidate")
    }
    return dict(
        qualified_processes=3,
        samples_per_implementation=1344,
        latency_ms=pooled,
        process_ranges_ms={
            mode: [
                min(r[mode] for r in process_means),
                max(r[mode] for r in process_means),
            ]
            for mode in pooled
        },
        same_checkpoint_speedup=pooled["native"] / pooled["candidate"],
        source_processes=[str(p) for p in folders],
        note="Caller must ensure one hardware session. Cross-process ranges are not confidence intervals.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("processes", type=Path, nargs=3)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = aggregate(a.processes)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
