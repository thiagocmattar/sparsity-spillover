"""Recompute the current paper's cross-scale coordinates and execution controls."""

import csv
import gzip
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / "results" / (name + ".json")).read_text())


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def geomean(values):
    assert values and all(math.isfinite(x) and x > 0 for x in values)
    return math.exp(math.fsum(map(math.log, values)) / len(values))


def verify_scale():
    endpoints = {r["condition"]: r for r in read("endpoints")["trained_points"]}
    executions = read("scale-frontier")["executions"]
    assert len(executions) == 36 and len({r["checkpoint_key"] for r in executions}) == 33
    for r in executions:
        e = endpoints[r["condition"]]
        assert r["checkpoint_key"] == e["checkpoint_key"]
        close(r["loss"], e["loss"])
        if r["backend"] == "kernel":
            close(r["latency_ms"], e["latency_ms"])
    for size in ("14M", "31M", "70M"):
        base = [r for r in executions if r["model"] == size and r["scope"] == "0"]
        assert {r["backend"] for r in base} == {"kernel", "PyTorch"} and len(base) == 2
        assert base[0]["loss"] == base[1]["loss"] and base[0]["checkpoint_key"] == base[1]["checkpoint_key"]
        for scope in ("hz", "7"):
            group = [r for r in executions if r["model"] == size and r["scope"] == scope]
            assert sorted(r["kappa"] for r in group) == [0, .01, .05, .1, .5]
        for kappa in (0, .01, .05, .1, .5):
            a, b = [next(r for r in executions if r["model"] == size and r["scope"] == s and r["kappa"] == kappa)
                    for s in ("hz", "7")]
            assert a["loss"] < b["loss"]

    processes = json.loads(gzip.decompress((ROOT / "results/31m-timing-processes.json.gz").read_bytes()))
    assert len(processes) == 33
    by_condition = {}
    for p in processes:
        e = endpoints[p["condition"]]
        assert e["checkpoint_key"] == p["checkpoint_key"]
        q = p["quality"]
        assert (q["blocks"], q["documents"], q["excluded_tail_tokens"]) == (338, 500, 1444)
        assert all(q["pass"].values())
        for mode, gates in q["gates"].items():
            assert {r["input_index"] for r in gates} == set(range(338)) and len(gates) == 338
            assert all(r["pass"] and r["finite"] and r["elementwise_gate"] and r["relative_l2"] <= .02 for r in gates)
            assert abs(q["loss_delta"][mode]) <= .001
        means = {}
        for mode in ("kernel_graph", "native_graph"):
            samples = [r for r in p["timing"]["samples"] if r["mode"] == mode]
            assert len(samples) == 448
            assert {(r["repeat"], r["input_index"]) for r in samples} == {(j, i) for j in range(7) for i in range(64)}
            assert all(r["output_shape"] == [1, 2048, 50304] for r in samples)
            means[mode] = geomean([r["host_ms"] for r in samples])
            close(means[mode], p["timing"]["summary"][mode]["geomean_host_ms"])
        by_condition.setdefault(p["condition"], []).append((p["replicate"], means))
    assert len(by_condition) == 11
    for name, values in by_condition.items():
        assert sorted(r[0] for r in values) == [1, 2, 3]
        for mode in ("kernel_graph", "native_graph"):
            close(geomean([r[1][mode] for r in values]), endpoints[name]["implementation_latency_ms"][mode])
    for r in read("31m-diagnostics"):
        e = endpoints[r["condition"]]
        close(r["validation"]["final"]["loss"], e["loss"])
        assert r["logical_products"]["measured"]["block_zero_product_count"] == e["zero_product_count"]
        assert r["logical_products"]["measured"]["model_product_count"] == e["model_product_count"]

    evidence = read("scale-evidence")
    native_base = evidence["base_native_latency_ms"]
    for row in evidence["rows_31m"]:
        name = "31M-T0-P0" if row["recipe"] == "Base" else f"31M-T{'2' if row['recipe'].startswith('T2') else '7'}-Ph-{row['kappa']:g}"
        e = endpoints[name]
        close(row["loss"], e["loss"])
        close(row["latency_ms"], native_base if row["recipe"] == "Base" else e["latency_ms"])
        close(row["delta_latency_ms"], row["latency_ms"] - native_base)
        close(row["delta_loss"], e["loss"] - endpoints["31M-T0-P0"]["loss"])

    hz = read("14m-t2-execution-controls")
    lat = {k: v["host_geomean_ms"] for k, v in hz["latency"].items()}
    joint = 100 * (1 - lat["A_graph"] / lat["D_graph"])
    close(joint, hz["effects"]["joint_hz"]["reduction_percent"])
    assert round(joint, 1) == 12.9
    share = 100 * (lat["D_graph"] - lat["A_graph"]) / (lat["native_graph"] - lat["A_graph"])
    close(share, hz["engineering_comparison"]["hz_share_of_same_checkpoint_native_gap_percent"])
    assert round(share, 1) == 57.9
    controls = {r["implementation"]: r["candidate_ms"] for r in read("70m-controls")["rows"] if r["condition"] == "70M-T7-Ph-0.5"}
    skip_reduction = 100 * (1 - controls["specialized-70m"] / controls["hz-skips-off"])
    assert round(skip_reduction, 1) == 58.3
    current = read("14m-paper-figure")["points"]
    assert len(current) == 40 and len({r["checkpoint_key"] for r in current}) == 40
    for r in current:
        for key in ("loss", "latency_ms", "zero_product_count", "model_product_count"):
            assert r[key] == endpoints[r["condition"]][key]
    base = read("base-training")
    assert {r["scale"] for r in base["runs"]} == {"14M", "31M", "70M", "410M"}
    assert all(len(r["rows"]) == 712 for r in base["runs"])
    output = ROOT / "reproduced"
    output.mkdir(exist_ok=True)
    with (output / "scale-frontier.csv").open("w", newline="") as f:
        fields = ["condition", "model", "scope", "pressure", "kappa", "backend", "loss", "latency_ms", "timing_session", "checkpoint_key"]
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(executions)
    with (output / "31m-results.csv").open("w", newline="") as f:
        fields = list(evidence["rows_31m"][0])
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(evidence["rows_31m"])
    summary = dict(executions=36, checkpoints=33, matched_kappa_comparisons=15,
                   qualified_31m_processes=33, raw_timing_samples=33*2*448,
                   t2_hz_reduction_percent=joint, t2_hz_share_of_native_gap_percent=share,
                   t7_70m_skips_off_reduction_percent=skip_reduction)
    (output / "scale-checks.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    verify_scale()
