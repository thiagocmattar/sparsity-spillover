"""Add the verified Run044 endpoint; preserve all previously published values."""
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "analyses/024-2026-09-17-h-only-kernel-latency"
RUN = ROOT / "runs/044-2026-09-20-pythia14m-hz-h-only-ol1-kappa05"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gm(values):
    assert values and all(math.isfinite(v) and v > 0 for v in values)
    return math.exp(statistics.mean(map(math.log, values)))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def collect():
    sources = {}

    def read(path):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return json.loads(path.read_text(encoding="utf-8"))

    previous = read(OLD / "data/14m-main-quality-sparsity-latency.json")
    complete = read(OLD / "data/all-model-quality-sparsity.json")
    assert len(previous["trained_points"]) == 40 and len(complete["trained_points"]) == 74
    assert sha(OLD / previous["output"]) == previous["output_sha256"]
    for document in [previous, complete]:
        for path, expected in document["sources_sha256"].items():
            assert sha(ROOT / path) == expected, path
    training = read(RUN / "artifacts/verification.json")
    latency = read(RUN / "latency/artifacts/verification.json")
    summary = read(RUN / "artifacts/summary.json")
    assert training["status"] == "verified" and training["condition_count"] == 1
    assert latency["all_required_artifacts_verified"] and latency["qualified_processes"] == 3
    for source in summary["sources"]:
        assert sha(RUN / source["path"]) == source["sha256"]
    trained = training["conditions"][0]
    condition = trained["condition"]
    assert condition["active_sites"] == condition["one_sided_sites"] == ["h", "z"]
    assert condition["pressure_sites"] == ["h"] and condition["gate_threshold"] == .5
    assert condition["pressure_method"] == "orthogonal_l1"
    assert condition["pressure_weight"] == condition["step_budget"] == 1
    attempt = RUN / "artifacts/attempts" / trained["attempt_id"]
    metrics = read(attempt / "metrics.json")
    logical = read(attempt / "diagnostics/logical_products.json")
    manifest = read(attempt / "manifest.json")
    counts = logical["measured"]
    coverage = complete["coverage"]
    for item in [metrics["validation"]["final"], logical["coverage"]]:
        assert all(item[k] == v for k, v in coverage.items())
    assert sum(p["zero_product_count"] for p in counts["per_operation"].values()) == counts["block_zero_product_count"]
    assert counts["model_product_count"] == counts["block_product_count"] + counts["lm_head_product_count"]
    assert manifest["seeds"] == {"model": 1234, "data_order": 1234}
    assert metrics["training"]["completed_steps"] == metrics["training"]["optimizer_step_count"] == 712
    assert metrics["training"]["input_tokens"] == 1493172224
    assert logical["architecture_maximum"] == previous["ceilings"]["2"]
    candidate, native, ratios, processes = [], [], [], []
    for replicate in range(1, 4):
        folder = RUN / "latency/artifacts/attempts" / f"scientific-c00-r{replicate}-001"
        result = read(folder / "result.json")
        quality = read(folder / "quality.json")
        timing = read(folder / "timing.json")
        assert result["status"] == "complete" and result["qualified"]
        assert result["validation_blocks"] == quality["blocks"] == 338
        assert quality["documents"] == 500 and quality["excluded_tail_tokens"] == 1444
        assert result["checkpoint"]["final_checkpoint_content_sha256"] == trained["checkpoint_content_sha256"]
        pairs = {}
        for sample in timing["samples"]:
            pair = pairs.setdefault((sample["repeat"], sample["input_index"]), {})
            assert sample["mode"] not in pair
            pair[sample["mode"]] = sample["host_ms"]
        assert len(pairs) == 448 and len(timing["indices"]) == 64
        for pair in pairs.values():
            assert set(pair) == {"native_graph", "candidate_graph"}
            candidate.append(pair["candidate_graph"])
            native.append(pair["native_graph"])
            ratios.append(pair["native_graph"] / pair["candidate_graph"])
        processes.append(gm([p["candidate_graph"] for p in pairs.values()]))
    summary_row = summary["conditions"][0]
    assert summary_row["kappa"] == .5 and summary_row["paired_samples"] == len(candidate) == 1344
    for key, value in [
        ("k050_geomean_host_ms", gm(candidate)),
        ("native_geomean_host_ms", gm(native)),
        ("native_relative_paired_geomean_speedup", gm(ratios)),
    ]:
        assert math.isclose(summary_row[key], value, rel_tol=1e-12)
    devices = {p["gpu_uuid"] for p in latency["scientific_processes"]}
    assert len(devices) == 1
    row = dict(
        model="14M", family="HZ-OL1-h", scope="hz", pressure="h", kappa=.5,
        local_pressure_weight=None, source_attempt=attempt.relative_to(ROOT).as_posix(),
        checkpoint_key=trained["checkpoint_content_sha256"],
        final_checkpoint_content_sha256=trained["checkpoint_content_sha256"],
        loss=metrics["validation"]["final"]["loss"],
        sparsity=100 * counts["block_zero_product_count"] / counts["model_product_count"],
        zero_product_count=counts["block_zero_product_count"], model_product_count=counts["model_product_count"],
        coverage=coverage, initial_parameter_sha256=manifest["initial_parameter_sha256"],
        training_schedule_hash=manifest["training_schedule_hash"], training_steps=712,
        training_tokens=1493172224, latency_ms=gm(candidate), process_latency_ms=processes,
        timing_session="Run044", timing_device_uuid=next(iter(devices)),
        timing_workload=previous["trained_points"][0]["timing_workload"],
        kernel="K050", qualified=True, native_same_checkpoint_latency_ms=gm(native),
        native_same_checkpoint_paired_speedup=gm(ratios),
        selected_site_exact_zero_fractions=trained["selected_site_exact_zero_fractions"],
        final_learning_rate=metrics["training"]["learning_rate_final"],
        logical_pass_loss=logical["coverage"]["loss"])
    assert row["loss"] == summary_row["training_validation_loss"]
    assert math.isclose(row["sparsity"], 100 * summary_row["R_model"], rel_tol=1e-12)
    main = copy.deepcopy(previous)
    main["trained_points"].append(row)
    for panel in main["panels"]:
        panel["trained_keys"].append(row["checkpoint_key"])
    hz_series = next(s for s in main["series"] if s["scope"] == "hz")
    hz_series["keys"].append(row["checkpoint_key"])
    assert len(main["trained_points"]) == len({r["checkpoint_key"] for r in main["trained_points"]}) == 41
    for field in ["initial_parameter_sha256", "training_schedule_hash"]:
        assert len({r[field] for r in main["trained_points"]}) == 1
    hz = sorted((r for r in main["trained_points"] if r["scope"] == "hz"), key=lambda r: r["kappa"])
    assert [r["kappa"] for r in hz] == [0, .01, .05, .1, .5]
    main.update(
        sources_sha256=sources,
        scope="41 trained 14M endpoints; previous 40 plus Run044. Ten recipes and all clipping coordinates unchanged.",
        dose_note="T1/Ph: lambda=.05,.1,.5,1. T2/Ph and T4/T7: kappa=0,.01,.05,.1,.5.",
        quality_view_note="All 41 trained points are visible; eight high-loss clipping points remain outside panel (a).",
        latency_note="K050 RTX5090 BF16 B1/T2048 full logits; 1344 pairs per endpoint. Runs029/033/041/044 are separate physical GPU/host sessions; small cross-session differences are descriptive.",
        observation="observations/001-run044-integration.md")
    for key in ["output", "output_sha256", "script", "script_sha256"]:
        main.pop(key, None)
    full = {k: copy.deepcopy(complete[k]) for k in [
        "trained_points", "clipping_points", "ceilings", "high_threshold_T7_all",
        "high_threshold_speedups", "coverage", "training_steps", "training_tokens", "cross_size_protocol"]}
    full["trained_points"].extend(hz)
    assert len(full["trained_points"]) == len({r["source_attempt"] for r in full["trained_points"]}) == 79
    assert [sum(r["model"] == size for r in full["trained_points"]) for size in ["14M", "70M", "410M"]] == [45, 22, 12]
    full["ceilings"]["14M"]["hz"] = logical["architecture_maximum"]
    full.update(
        title="Complete manuscript training endpoints including the five-point T2/Ph grid",
        sources_sha256=sources, loss_convention="Ordinary final-checkpoint FP16 validation for all 79 endpoints; retained FP16 clipping loss.",
        main_figure_checkpoint_keys=[r["checkpoint_key"] for r in main["trained_points"]],
        scope_note="The complete 45/22/12 cohort includes four historical naive-L1 14M ablations omitted from the 41-point main figure. Previous 74 endpoint records and all 60 clipping records are unchanged.",
        clipping_note="No post-hoc clipping was measured for T2/Ph, including Run044.",
        timing_note=main["latency_note"], observation="observations/001-run044-integration.md")
    assert full["trained_points"][:74] == complete["trained_points"]
    assert main["trained_points"][:40] == previous["trained_points"]
    write(HERE / "data/figure-data.json", main)
    write(HERE / "data/full-trained-results.json", full)
    print(json.dumps({"main_points": 41, "full_points": 79, "new_point": row}))


if __name__ == "__main__":
    collect()
