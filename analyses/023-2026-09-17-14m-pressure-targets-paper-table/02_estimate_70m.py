"""Historical H200 timing evidence and provisional four-condition h-only ETC."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / "runs/018-2026-09-01-pythia70m-selected-ladder-canonical-init"
PREFLIGHT = RUN / "launch-control/timing-preflight-cnekba7p07zplh/attempt-002-passed/remote-preflight.json"


def main():
    sources = {}

    def read(path):
        sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text(encoding="utf-8"))

    verified = read(RUN / "artifacts/verification.json")
    assert verified["status"] == "verified"
    history = []
    for row in verified["conditions"]:
        condition = row["condition"]
        if condition["pressure_method"] != "orthogonal_l1":
            continue
        attempt = RUN / "artifacts/attempts" / row["attempt_id"]
        metrics, manifest = read(attempt / "metrics.json"), read(attempt / "manifest.json")
        assert manifest["environment"]["devices"] == ["NVIDIA H200"]
        assert manifest["completed_steps"] == 712 and manifest["input_tokens"] == 1493172224
        wall = (datetime.fromisoformat(manifest["finished_at"]) - datetime.fromisoformat(manifest["started_at"])).total_seconds()
        total = metrics["timing"]["total_seconds"]
        assert abs(wall - total) < 2
        history.append({
            "condition_id": condition["id"], "kappa": condition["gate_threshold"],
            "family": condition["step"], "pressure_sites": condition["pressure_sites"],
            "attempt_id": row["attempt_id"], "worker_total_seconds": total,
            "manifest_wall_seconds": wall,
            "peak_reserved_GiB": row["peak_gpu_memory_reserved_bytes"] / 2**30,
            "retained_bytes": row["transfer_bytes"],
        })
    assert len(history) == 10
    preflight = read(PREFLIGHT)
    assert preflight["status"] == "passed"
    sample = next(r for r in preflight["samples"] if r["condition_id"] == "a7-ol1-kappa-0")
    assert len(sample["boundaries"]) == 5
    forecast = []
    for family, bounds in [("A4-OL1(h)", [90, 105]), ("A7-OL1(h)", [95, 115])]:
        for kappa in [0.05, 0.5]:
            forecast.append({"family": family, "kappa": kappa, "worker_minutes": bounds})
    overhead = [20, 45]
    elapsed = {
        "four_H200_parallel_minutes": [max(r["worker_minutes"][i] for r in forecast) + overhead[i] for i in (0, 1)],
        "two_H200_two_waves_minutes": [sum(forecast[j]["worker_minutes"][i] for j in (0, 2)) + overhead[i] for i in (0, 1)],
        "one_H200_serial_minutes": [sum(r["worker_minutes"][i] for r in forecast) + overhead[i] for i in (0, 1)],
    }
    result = {
        "status": "historical_estimate_not_new_calibration_or_launch", "date": "2026-09-17",
        "sources_sha256": sources, "hardware": "one NVIDIA H200 per worker; independent conditions, no DDP",
        "protocol_assumption": "Match Run 018 canonical seed-1234 initialization, realized data order, FP16, microbatch 4 / accumulation 256, 712 updates, 1,493,172,224 tokens, AdamW recipe, complete validation/diagnostics; change pressure sites to h only at lambda=1,b=1.",
        "retention_assumption": "Allow Run 032-style 12 model checkpoints and optimizer/RNG at 256,512,712; Run 018 timing evidence retained final-only.",
        "estimated_checkpoint_bytes_per_condition": 281706496 * (12 + 2 * 3),
        "preflight": {key: sample[key] for key in ["condition_id", "median_end_to_end_step_seconds", "min_end_to_end_step_seconds", "max_end_to_end_step_seconds"]},
        "historical_all_site_conditions": history,
        "new_h_only_conditions_estimated": forecast,
        "setup_preflight_transfer_verification_allowance_minutes": overhead,
        "elapsed_estimates": elapsed,
        "uncertainty": "Judgment ranges anchored to all-site H200 observations, not statistical intervals. No h-only speedup assumed. Three historical workers took 150-159 minutes; contention or slow transfer can exceed the normal estimate. Availability and prices are not audited for this timing-only request. A100/local h-only timing is unmeasured.",
    }
    (HERE / "70m-etc.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    lines = ["# Provisional 70M h-only OL1 ETC", "", result["protocol_assumption"], "",
             "Timing basis: one H200 per independent condition. These are estimates, not a new calibration or a launch definition.", "",
             "| Condition | kappa | Worker time, including evaluation/diagnostics/checkpointing |",
             "|---|---:|---:|"]
    for row in forecast:
        lo, hi = row["worker_minutes"]
        lines.append(f"| {row['family']} | {row['kappa']:g} | {lo}–{hi} min |")
    lines.extend(["", "Add 20–45 minutes once to the elapsed schedule for provision, setup, short calibration, transfer and local verification, assuming available capacity and functioning transfers.", "",
                  "| Schedule | Total elapsed estimate |", "|---|---:|"])
    labels = ["Four H200s, four conditions in parallel", "Two H200s, two waves", "One H200, four conditions sequentially"]
    for label, bounds in zip(labels, elapsed.values()):
        lines.append(f"| {label} | {bounds[0]}–{bounds[1]} min ({bounds[0]/60:.2f}–{bounds[1]/60:.2f} h) |")
    lines.extend(["", "## Evidence and uncertainty", "",
                  f"The exact Run 018 A7 preflight measured five complete boundaries: median {sample['median_end_to_end_step_seconds']:.3f} s, range {sample['min_end_to_end_step_seconds']:.3f}–{sample['max_end_to_end_step_seconds']:.3f} s. Completed all-site runs provide the stronger total-worker-time evidence below.", "",
                  "| Historical all-site condition | Worker min | Peak reserved GiB |", "|---|---:|---:|"])
    for row in history:
        lines.append(f"| {row['condition_id']} | {row['worker_total_seconds']/60:.2f} | {row['peak_reserved_GiB']:.3f} |")
    lines.extend(["", "Typical all-site H200 workers took 88.67–98.85 minutes. Three slower workers took 150.51–159.24 minutes; the contemporaneous Run 018 recovery notes identify regional contention. Threshold alone is not a runtime predictor for these dense kernels.", "",
                  "The h-only estimate conservatively reuses the all-site timings without assuming a speed gain, and adds room for a larger retained checkpoint inventory. The exact h-only path still needs a short calibration before a launch ETC can be claimed.", "",
                  "Checkpoint storage is approximately 5.07 GB per condition (about 20.3 GB total) for the Run 032-style inventory, plus small diagnostics/logs. Run 018 retained only the final recovery checkpoint (about 0.85 GB per condition). Input cache and initialization are about 6.25 GB per worker. The 20–45 minute transfer allowance is provisional, not a measured guarantee for a new host/network.", "",
                  "Observed all-site memory was 9.99–10.17 GiB. This does not establish the h-only workload's memory headroom or runtime on the local 12 GB laptop. No valid matched 70M A100 training timing was found: the historical A100 attempt stopped at initialization. H200 ETC must not be relabeled as A100 ETC.", "",
                  "No prices or capacity are quoted, and no new run or cloud resource is created. Reproduce with `python analyses/023-2026-09-17-14m-pressure-targets-paper-table/02_estimate_70m.py`; exact inputs/hashes are in `70m-etc.json`.", ""])
    (HERE / "70M-ETC.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps(elapsed))


if __name__ == "__main__":
    main()
