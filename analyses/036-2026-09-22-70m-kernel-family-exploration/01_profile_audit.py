"""Reduce retained Run049 profiles; no inference, tuning, or GPU execution."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / "runs/049-2026-09-22-pythia70m-short-row-limits"
SHAPES = {
    ((1536,), (2048, 512), (512, 1536)): "qkv",
    ((2048,), (2048, 512), (512, 2048)): "up",
    ((512,), (2048, 2048), (2048, 512)): "h_down",
    ((512,), (2048, 512), (512, 512)): "z_out",
}


def identity(path):
    return {"path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def audit(cid):
    paths = sorted((RUN / "artifacts/attempts").glob(
        f"final-{cid}-r1-*/profile-native_hz-eager.json"))
    assert len(paths) == 1, paths
    eager_path = paths[0]
    graph_path = eager_path.with_name("profile-native_hz-graph.json")
    eager = json.loads(eager_path.read_text())["traceEvents"]
    graph = json.loads(graph_path.read_text())["traceEvents"]
    cpu = {e["args"]["External id"]: e for e in eager
           if e.get("cat") == "cpu_op"}
    ek = sorted((e for e in eager if e.get("cat") == "kernel"),
                key=lambda e: e["ts"])
    gk = sorted((e for e in graph if e.get("cat") == "kernel"),
                key=lambda e: e["ts"])
    labels_by_name = defaultdict(list)
    for e in ek:
        parent = cpu.get(e["args"].get("External id"), {})
        if parent.get("name") == "aten::addmm":
            dims = tuple(tuple(s) for s in parent["args"]["Input Dims"][:3])
            labels_by_name[e["name"]].append(SHAPES[dims])
    # Graph traces have extra gate-copy launches. Never zip the entire traces.
    # Match only the GEMM subsequence, asserting names, counts and h/z order.
    graph_gemms = [e for e in gk if e["name"] in labels_by_name]
    eager_gemms = [e for e in ek if e["name"] in labels_by_name]
    assert [e["name"] for e in graph_gemms] == [e["name"] for e in eager_gemms]
    used = Counter()
    durations = defaultdict(list)
    ordered_labels = []
    for e in graph_gemms:
        name = e["name"]
        label = labels_by_name[name][used[name]]
        used[name] += 1
        durations[label].append(e["dur"])
        ordered_labels.append(label)
    assert ordered_labels == ["qkv", "up", "h_down", "z_out"] * 24
    assert all(len(v) == 24 for v in durations.values())
    for e in gk:
        name = e["name"]
        if "compare_scalar_kernel" in name:
            durations["gate_compare"].append(e["dur"])
        elif "masked_fill_kernel" in name:
            durations["gate_fill"].append(e["dur"])
        elif name == "memcpy128":
            durations["graph_copy_gate_associated"].append(e["dur"])
        elif "CUDAFunctor_add" in name:
            durations["branch_residual_add"].append(e["dur"])
    expected_gate_calls = 0 if cid == "c00" else 48
    for label in ("gate_compare", "gate_fill", "graph_copy_gate_associated"):
        assert len(durations[label]) == expected_gate_calls
    assert len(durations["branch_residual_add"]) == 48
    reduced = {k: {"calls": len(v), "mean_sum_us_per_forward": sum(v) / 4}
               for k, v in sorted(durations.items())}
    hz_us = sum(reduced[k]["mean_sum_us_per_forward"] for k in ("h_down", "z_out"))
    gate_us = sum(reduced[k]["mean_sum_us_per_forward"] for k in
                  ("gate_compare", "gate_fill", "graph_copy_gate_associated"))
    return {"condition": cid, "sources": [identity(eager_path), identity(graph_path)],
            "profiled_inputs": 4, "layers": 6, "components": reduced,
            "h_z_gemm_us": hz_us, "gate_and_associated_copy_us": gate_us,
            "all_kernel_duration_sum_us_per_forward": sum(e["dur"] for e in gk) / 4,
            "whole_kernel_sequence_equal": [e["name"] for e in ek] == [e["name"] for e in gk]}


def main():
    summary_path = RUN / "results/summary.json"
    conditions = {c["id"]: c for c in json.loads(summary_path.read_text())["conditions"]}
    base = conditions["c00"]["modes"]["native_hz"]["latency_ms"]
    records = []
    for cid in ("c00", "c24", "c25"):
        record = audit(cid)
        own = conditions[cid]["modes"]["native_hz"]["latency_ms"]
        record.update({"unprofiled_latency_ms": own,
                       "saving_to_match_efficient_base_us": (own - base) * 1000,
                       "ten_percent_latency_target_ms": own * .9})
        records.append(record)
    out = {"method": "Shape-correlated eager GEMM labels mapped to the identical graph GEMM subsequence; trace durations in microseconds, divided by four forwards.",
           "limitations": "Four instrumented inputs, one process per checkpoint. Copy attribution is inferred from functional masked_fill and its absence on Base. Duration sums are not counterfactual full-model savings or unprofiled timings.",
           "summary_source": identity(summary_path), "conditions": records}
    destination = HERE / "data/profile-audit.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for r in records:
        print(r["condition"], "h/z us", round(r["h_z_gemm_us"], 3),
              "gates/copy us", round(r["gate_and_associated_copy_us"], 3),
              "gap to Base us", round(r["saving_to_match_efficient_base_us"], 3))


if __name__ == "__main__":
    main()
