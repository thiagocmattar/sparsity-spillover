"""Reproduce planning arithmetic from retained timings and a frozen live quote.

No experiment code, cloud mutation, or scientifically new measurement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN410 = ROOT / "runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init"


def main():
    sources = {}

    def read(path):
        sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text(encoding="utf-8"))

    prior70 = read(HERE / "70m-etc.json")
    quote = read(HERE / "70m-410m-price-snapshot.json")
    verified = read(RUN410 / "artifacts/verification.json")
    assert verified["status"] == "verified"
    history410 = []
    for row in verified["conditions"]:
        condition = row["condition"]
        if condition["pressure_method"] != "orthogonal_l1":
            continue
        attempt = RUN410 / "artifacts/attempts" / row["attempt_id"]
        manifest, metrics = read(attempt / "manifest.json"), read(attempt / "metrics.json")
        assert manifest["completed_steps"] == 712
        assert manifest["input_tokens"] == 1493172224
        history410.append({"condition_id": condition["id"],
                           "gpu": manifest["environment"]["devices"],
                           "worker_hours": metrics["timing"]["total_seconds"] / 3600,
                           "retained_bytes": row["transfer_bytes"]})
    assert len(history410) == 10
    calibration = read(RUN410 / "prelaunch/calibration/calibration-h200-secure.json")
    assert calibration["status"] == "passed"
    h200410 = {family: calibration["projection"]["conditions"][f"{family}-ol1-kappa-0p5"][
        "projected_total_seconds"] / 3600 for family in ["a4", "a7"]}

    # Judgment ranges around the all-site observations/calibration, NOT CIs.
    # h-only is uncalibrated: do not assume its smaller pressure scope is faster.
    assumptions = {
        "70M H200": {"gpu_id": "NVIDIA H200", "worker_hours": {
            "A4": [90 / 60, 105 / 60], "A7": [95 / 60, 115 / 60]},
            "overhead_hours_by_kappa_count": {2: [20 / 60, 45 / 60], 5: [0.5, 1.5]},
            "running_disk_GB": 80, "checkpoint_bytes_per_condition": 281706496 * 18},
        "410M A100": {"gpu_id": "NVIDIA A100-SXM4-80GB", "worker_hours": {
            "A4": [17.5, 19], "A7": [18.5, 21]},
            "overhead_hours_by_kappa_count": {2: [1, 2], 5: [1.5, 3]},
            "running_disk_GB": 120, "checkpoint_bytes_per_condition": 1621336064 * 18},
        "410M H200": {"gpu_id": "NVIDIA H200", "worker_hours": {
            "A4": [11, 12], "A7": [12, 13.5]},
            "overhead_hours_by_kappa_count": {2: [1, 2], 5: [1.5, 3]},
            "running_disk_GB": 120, "checkpoint_bytes_per_condition": 1621336064 * 18},
    }
    grids = {"two_kappas": [0.05, 0.5], "all_five_kappas": [0, 0.01, 0.05, 0.1, 0.5]}
    forecasts = {}
    for grid_name, kappas in grids.items():
        forecasts[grid_name] = {}
        for label, a in assumptions.items():
            n = 2 * len(kappas)
            overhead = a["overhead_hours_by_kappa_count"][len(kappas)]
            elapsed = [max(b[i] for b in a["worker_hours"].values()) + overhead[i] for i in [0, 1]]
            gpu_hours = [n * h for h in elapsed]
            disk_cost = [h * a["running_disk_GB"] * 0.10 / 720 for h in gpu_hours]
            price = next(r for r in quote["gpus"] if r["gpuId"] == a["gpu_id"])
            costs = {tier: [gpu_hours[i] * price[f"{tier}PricePerHr"] + disk_cost[i] for i in [0, 1]]
                     for tier in ["community", "secure"]}
            forecasts[grid_name][label] = {"conditions": n, "parallel_gpus": n,
                "elapsed_hours": elapsed, "conservative_billed_gpu_hours": gpu_hours,
                "running_disk_cost_usd": disk_cost, "total_cost_usd_by_tier": costs,
                "checkpoint_GB": n * a["checkpoint_bytes_per_condition"] / 1e9}
        for option, gpu410 in [("H200_70M_A100_410M", "410M A100"), ("H200_both_scales", "410M H200")]:
            rows = [forecasts[grid_name][label] for label in ["70M H200", gpu410]]
            forecasts[grid_name][option] = {
                "conditions": sum(r["conditions"] for r in rows),
                "parallel_gpus": sum(r["parallel_gpus"] for r in rows),
                "elapsed_hours_both_scales_concurrent": [max(r["elapsed_hours"][i] for r in rows) for i in [0, 1]],
                "total_cost_usd_by_tier": {tier: [sum(r["total_cost_usd_by_tier"][tier][i] for r in rows) for i in [0, 1]] for tier in ["community", "secure"]},
                "checkpoint_GB": sum(r["checkpoint_GB"] for r in rows)}
    result = {"status": "estimate_only_not_design_or_launch_authorization",
        "date": "2026-09-17", "sources_sha256": sources,
        "grids": grids, "assumptions": assumptions, "forecast": forecasts,
        "historical_70m_all_site": prior70["historical_all_site_conditions"],
        "historical_410m_all_site": history410,
        "h200_410m_calibrated_worker_hours": h200410,
        "cost_model": "Conservatively hold all Pods within each scale until its slowest worker plus overhead completes. Release each scale separately. Disk includes container plus volume allocation. A 30-day month is used for disk proration. Retrieval verified before termination.",
        "protocol": "Match each scale's canonical Run018/019 seed1234 initialization, schedule, FP16 AdamW, MB4 x accumulation256, 712 updates and 1493172224 input tokens. A4 gates a,m,h,z; A7 adds q_post,k_post,v. OL1 only h, lambda=1,b=1. Full 500-document/338-block validation, excluded tail1444, previous diagnostics. No old conditions rerun.",
        "retention": "Planning assumption: 12 FP32 model snapshots and three optimizer/RNG recovery states (18 model-size equivalents), matching the recent 14M retention schedule. This is larger than Run018/019 final-only retention, not an approved new design.",
        "transfer": "Overhead is a planning allowance, not measured on new hosts. Assumes one pre-staged hash-verified cache distributed between cloud workers and at least 50 MB/s aggregate download throughput. Upload, download and verification overlap where possible. No repeated slow laptop upload per Pod. Slow links can materially exceed these estimates.",
        "exclusions": "Tax/currency conversion, pre-existing network volume, third-party staging fees, extra seeds, retraining, kernel benchmarks and prolonged outages. No h-only speed advantage assumed. Hardware-specific h-only calibration remains necessary for launch ETC.",
        "capacity": "Catalog stock Low, no count guarantee. Account spendLimit read-only query returned USD80/hour on 2026-09-17. Twenty Secure H200s alone cost USD91.80/hour and exceed that limit; phase the 70M and 410M groups or verify a permitted tier mix/limit change before any launch.",
    }
    (HERE / "70m-410m-etc-cost.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(forecasts, indent=2))


if __name__ == "__main__":
    main()
