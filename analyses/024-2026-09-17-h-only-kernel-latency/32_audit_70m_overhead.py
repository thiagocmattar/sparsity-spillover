"""Audit saved timings and dense-path work; no GPU execution or new experiment."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R35 = ROOT / "runs/035-2026-09-18-pythia70m-k050-port"
PROJECTIONS = ("qkv_projection", "mlp_w1", "mlp_w2", "attention_output_projection")


def gm(values):
    assert values and all(math.isfinite(x) and x > 0 for x in values)
    return math.exp(math.fsum(map(math.log, values)) / len(values))


def main():
    sources = {}

    def read(path):
        raw = path.read_bytes()
        sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    cohort = read(HERE / "data/paper-checkpoints.json")["checkpoints"]
    counts = read(HERE / "data/operation-bypass.json")["settings"]
    frozen = read(R35 / "artifacts/frozen-final-port.json")["sources"]
    bases = {}
    for size, width in (("14M", 128), ("70M", 512)):
        row, = [r for r in cohort if r["model"] == size and r["scope"] == "0"]
        counter, = [r for r in counts if r["checkpoint_key"] == row["checkpoint_key"]
                    and r["kind"] == "trained"]
        folder = (ROOT / counter["diagnostic_source"]).parent
        diag = read(folder / "diagnostics.json")
        assert diag["coverage"]["blocks"] == 338
        assert diag["coverage"]["excluded_tail_tokens"] == 1444
        times = {mode: {timer: [] for timer in ("host_ms", "cuda_ms")}
                 for mode in ("native_graph", "candidate_graph")}
        processes = []
        for rep in (1, 2, 3):
            attempt = folder.with_name(folder.name.replace("r1-", f"r{rep}-"))
            result, timing = [read(attempt / name) for name in ("result.json", "timing.json")]
            assert result["qualified"] and result["validation_blocks"] == 338
            if size == "70M":
                assert result["port_sources"] == frozen
            assert timing["indices"] == row["timing_indices"]
            record = {"replicate": rep, "device_uuid": result["runtime"]["device_uuid"]}
            for mode in times:
                samples = [s for s in timing["samples"] if s["mode"] == mode]
                assert len(samples) == 448
                assert {(s["input_index"], s["repeat"]) for s in samples} == {
                    (i, repeat) for i in range(64) for repeat in range(7)}
                assert all(s["output_shape"] == [1, 2048, 50304] for s in samples)
                for timer in times[mode]:
                    values = [s[timer] for s in samples]
                    times[mode][timer].extend(values)
                    record[f"{mode}_{timer}"] = gm(values)
            processes.append(record)
        assert len({p["device_uuid"] for p in processes}) == 1
        means = {mode: {timer: gm(values) for timer, values in timers.items()}
                 for mode, timers in times.items()}
        assert math.isclose(means["native_graph"]["host_ms"], row["native_latency_ms"], rel_tol=1e-12)
        assert math.isclose(means["candidate_graph"]["host_ms"], row["latency_ms"], rel_tol=1e-12)
        ops = counter["operations"]
        assert all(ops[n]["bypassed_mmas"] == ops[n]["simt_products"] == 0 for n in PROJECTIONS)
        # Independently reconcile instrumented h/z counts against their raw artifact.
        pooled = [sum(layer[i] for layer in diag["hybrid_counts_by_layer"].values())
                  for i in range(6)]
        assert pooled == [ops["mlp_w2"]["issued_mmas"], 0,
                          ops["attention_output_projection"]["issued_mmas"], 0, 0, 0]
        row_counts = {}
        for site in ("a", "m", "h", "z"):
            hist = [v for key, v in diag["active_features_per_row"].items()
                    if key.startswith(site + ".")]
            row_counts[site] = {"rows": sum(map(sum, hist)),
                                "rows_with_at_most_two_nonzeros": sum(sum(h[:3]) for h in hist)}
            assert row_counts[site]["rows"] == 338 * 2048 * 6
            assert row_counts[site]["rows_with_at_most_two_nonzeros"] == 0
        work = {n: {"issued_mmas": ops[n]["issued_mmas"],
                    "logical_products": ops[n]["logical_products"],
                    "issued_fma_equivalents_per_logical_product":
                    ops[n]["issued_mmas"] * 2048 / ops[n]["logical_products"]}
                for n in PROJECTIONS}
        # Static loop counts, not measured memory traffic or instruction timing.
        inspection = {"output_tiles_per_row_group": width // 128,
                      "row_groups_per_layer": 2048 // 8,
                      "h_plus_z_values_inspected_per_layer": 2048 * (5 * width) * (width // 128),
                      "h_plus_z_bf16_inspection_bytes_per_layer": 2 * 2048 * (5 * width) * (width // 128)}
        bases[size] = {"checkpoint_key": row["checkpoint_key"], "processes": processes,
                       "geometric_means_ms": means, "samples_per_mode": 1344,
                       "projection_work": work, "row_counts": row_counts,
                       "static_hz_inspection": inspection,
                       "all_projection_fma_equivalents_per_logical_product":
                       sum(ops[n]["issued_mmas"] * 2048 for n in PROJECTIONS) /
                       sum(ops[n]["logical_products"] for n in PROJECTIONS)}
    means = bases["70M"]["geometric_means_ms"]
    excess = {timer: means["candidate_graph"][timer] - means["native_graph"][timer]
              for timer in ("host_ms", "cuda_ms")}
    # Difference of geometric means is descriptive, not an additive time partition.
    excess["cuda_gap_over_host_gap"] = excess["cuda_ms"] / excess["host_ms"]
    for relative in ("kernel/candidate.py", "kernel/joint/candidate.py", "kernel/joint/joint.cu",
                     "kernel/projection/projection.cu", "kernel/norm/norm.cu",
                     "kernel/attention/candidate.py", "kernel/attention/sparse_gemm.h",
                     "02_benchmark.py", "20_numerical_isolation.py", "21_profile_native_attention.py"):
        path = R35 / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if relative.startswith("kernel/"):
            assert digest == frozen[relative]["sha256"], relative
        sources[path.relative_to(ROOT).as_posix()] = digest
    trace = read(R35 / "artifacts/native-attention-trace.json")
    kernels = [{"name": e["name"], "duration_us": e.get("dur")}
               for e in trace["traceEvents"] if e.get("cat") == "kernel"]
    related = read(ROOT / "runs/039-2026-09-19-pythia14m-am-port-fixture/results/am-load-avoidance.json")
    output = {"status": "existing-evidence audit; component latency attribution remains unmeasured",
              "bases": bases, "70M_excess": excess,
              "native_attention_profile": {"kernels": kernels,
                  "limitation": "One native synthetic-input attention call, not a paired full-model profile"},
              "related_14M_am_port_latency_ms": related["latencies_ms"],
              "sources_sha256": sources}
    (HERE / "data/70m-overhead-audit.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"sources": len(sources), "70M_excess": excess,
                      "base_host_ms": {size: b["geometric_means_ms"] for size, b in bases.items()}}, indent=2))


if __name__ == "__main__":
    main()
