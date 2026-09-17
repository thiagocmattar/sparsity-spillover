"""Retained-data audit for the pressure-placement revision; no model execution."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
ROOT = ANALYSIS.parents[1]
KAPPAS = (0, .01, .05, .1, .5)
FAMILIES = ("A4", "A4-OL1-H", "A4-OL1", "A7", "A7-OL1-H", "A7-OL1")
SITES = ("a", "m", "h", "q_post", "k_post", "v", "z")
LABELS = {"A4": "4-site", "A4-OL1-H": "4-site + OL1(h)", "A4-OL1": "4-site + OL1(all)",
          "A7": "7-site", "A7-OL1-H": "7-site + OL1(h)", "A7-OL1": "7-site + OL1(all)"}


def module(name):
    spec = importlib.util.spec_from_file_location(name, ANALYSIS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load():
    sources = {}

    def read(path):
        sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text(encoding="utf-8"))

    approved = read(ROOT / "analyses/023-2026-09-17-14m-pressure-targets-paper-table/results.json")
    old = read(ROOT / "analyses/018-2026-09-08-results-materials/figure_data.json")
    runtime = read(ANALYSIS / "investigation/data/checkpoints.json")
    for path, digest in approved["sources_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    geometry = module("02_ol1_geometry")
    rows, histories = [], []
    for original in approved["rows"]:
        attempt = ROOT / original["source_attempt"]
        stats = read(attempt / "diagnostics/activation_statistics.json")
        logical = read(attempt / "diagnostics/logical_products.json")
        coverage = logical["coverage"]
        assert (coverage["sequences"], coverage["excluded_tail_tokens"]) == (338, 1444)
        counts = logical["measured"]
        assert sum(r["zero_product_count"] for r in counts["per_operation"].values()) == counts["block_zero_product_count"]
        assert np.isclose(100 * counts["block_zero_product_count"] / counts["model_product_count"], original["S_model_percent"])
        pooled, layers = {}, {}
        for site in SITES:
            selected = [r for r in stats["rows"] if r["name"].startswith(site + ".")]
            assert len(selected) == 6 and all(r["nonfinite"] == 0 for r in selected)
            saved = next(r for r in stats["pooled_by_site"] if r["name"] == site)
            zero = sum(r["exact_zero_count"] for r in selected)
            total = sum(r["total"] for r in selected)
            assert zero == saved["exact_zero_count"] and total == saved["total"]
            pooled[site] = {"zero": zero, "total": total, "percent": 100 * zero / total}
            layers[site] = [{"layer": int(r["name"].split("_")[-1]), "zero": r["exact_zero_count"], "total": r["total"]} for r in selected]
        qkv = [pooled[s] for s in ("q_post", "k_post", "v")]
        pooled["qkv"] = {"zero": sum(r["zero"] for r in qkv), "total": sum(r["total"] for r in qkv)}
        pooled["qkv"]["percent"] = 100 * pooled["qkv"]["zero"] / pooled["qkv"]["total"]
        row = {**original, "loss": original["reported_validation_loss"], "sites": pooled, "layers": layers,
               "counts": counts, "operations_pp": {k: 100 * v["zero_product_count"] / counts["model_product_count"] for k, v in counts["per_operation"].items()}}
        rows.append(row)
        if not original["realized_pressure_sites"]:
            continue
        path = attempt / "events.jsonl"
        sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        steps = [r for r in events if r.get("event") == "train"]
        assert [r["step"] for r in steps] == list(range(1, 713))
        for step in steps:
            geometry.audit_step(step, {"eps": 1e-12, "weight": 1, "step_budget": 1})
            # Run 012 predates capture-count logging: its realized h-only scope is
            # established by the separately retained Analysis 009 execution audit.
            if original["family"] != "A4-OL1-H":
                assert step["pressure_capture_tensor_count"] == 6 * len(original["realized_pressure_sites"])
        histories.append({"family": row["family"], "kappa": row["kappa"], "summary": geometry.summarize(steps),
                          "steps": [{"step": r["step"], "rho": geometry.opposing_component_ratio(r),
                                     "ratio": r["pressure_to_task_ratio_raw"], "cap": r["trust_scale"] < 1,
                                     "conflict": r["task_pressure_dot_before"] < 0} for r in steps]})
    assert len(rows) == 30 and len(histories) == 20
    contrasts = []
    pairs = [("A4-OL1-H", "A4"), ("A4-OL1", "A4-OL1-H"),
             ("A7-OL1-H", "A7"), ("A7-OL1", "A7-OL1-H"),
             ("A7", "A4"), ("A7-OL1-H", "A4-OL1-H")]
    for treatment, reference in pairs:
        for k in KAPPAS:
            a, = [r for r in rows if r["family"] == treatment and r["kappa"] == k]
            b, = [r for r in rows if r["family"] == reference and r["kappa"] == k]
            contrasts.append({"treatment": treatment, "reference": reference, "kappa": k,
                              "delta_loss": a["loss"] - b["loss"], "delta_s_pp": a["S_model_percent"] - b["S_model_percent"],
                              "delta_loss_ordinary": a["ordinary_final_validation_loss"] - b["ordinary_final_validation_loss"],
                              "delta_loss_logical": a["logical_diagnostic_validation_loss"] - b["logical_diagnostic_validation_loss"]})
    groups = {}
    for family in [f for f in FAMILIES if "OL1" in f]:
        steps = [s for h in histories if h["family"] == family for s in h["steps"]]
        groups[family] = {"steps": len(steps), "cap_percent": 100 * sum(s["cap"] for s in steps) / len(steps),
                          "conflict_percent": 100 * sum(s["conflict"] for s in steps) / len(steps),
                          "median_ratio": float(np.median([s["ratio"] for s in steps])),
                          "median_rho": float(np.median([s["rho"] for s in steps]))}
    for r in runtime:
        assert r["projection_on_candidate_gm_ms"] < r["full_candidate_gm_ms"]
    assert len(runtime) == 30
    # Restore the already qualified historical h-only checkpoints to deployment
    # evidence. The original 30-point ablation estimand remains unchanged.
    extractor_spec = importlib.util.spec_from_file_location("kernel_extract", ANALYSIS / "investigation/01_extract.py")
    extractor = importlib.util.module_from_spec(extractor_spec)
    extractor_spec.loader.exec_module(extractor)
    run = ROOT / "runs/029-2026-09-07-pythia14m-matched-kernel-retrospective"
    historical = read(run / "results/matched-retrospective-001.json")
    timing_hashes = {r["path"]: r["sha256"] for r in historical["sources"]}
    extra_runtime = []
    for point in historical["points"]:
        if point["candidate"] != "k050" or point["family"] != "A4+OL1@h":
            continue
        assert point["phase"] == "final" and point["qualified"] and len(point["replicates"]) == 3
        endpoint, = [r for r in rows if r["family"] == "A4-OL1-H" and r["counts"] == point["canonical_counts"]]
        timings = []
        for rep in point["replicates"]:
            assert rep["qualified"] and rep["status"] == "complete"
            path = run / "artifacts/attempts" / rep["attempt"] / "timing.json"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == timing_hashes[path.relative_to(run).as_posix()]
            timing = read(path)
            assert timing["indices"] == historical["timing_block_indices"]
            summary = extractor.timing_summary(timing)
            assert np.isclose(summary["paired_speedup"], rep["speedup"], rtol=1e-12)
            timings.append(summary)
        summary = {k: extractor.gm([r[k] for r in timings]) for k in timings[0]}
        extra_runtime.append({"condition": point["condition"], "family": "4-site", "recipe": "A4-OL1-H",
                              "pressure_scope": "h", "kappa": endpoint["kappa"], "validation_loss": endpoint["loss"],
                              "s_model_percent": endpoint["S_model_percent"], "full_candidate_gm_ms": summary["candidate_gm_ms"],
                              "native_baseline_gm_ms": summary["native_gm_ms"], "full_speedup": summary["paired_speedup"],
                              "qualified": True, "replicates": 3, "source_attempt": endpoint["source_attempt"]})
        added = extra_runtime[-1]
        for label, candidate in (("all_skips_off", "k050-no-skip"), ("projection_on", "k050-attention-dense")):
            control, = [p for p in historical["points"] if p["condition"] == point["condition"] and p["candidate"] == candidate and p["phase"] == "final"]
            assert control["qualified"] and len(control["replicates"]) == 3
            summaries = []
            for rep in control["replicates"]:
                assert rep["qualified"] and rep["status"] == "complete"
                path = run / "artifacts/attempts" / rep["attempt"] / "timing.json"
                assert hashlib.sha256(path.read_bytes()).hexdigest() == timing_hashes[path.relative_to(run).as_posix()]
                timing = read(path)
                assert timing["indices"] == historical["timing_block_indices"]
                summaries.append(extractor.timing_summary(timing))
            added[label + "_candidate_gm_ms"] = extractor.gm([r["candidate_gm_ms"] for r in summaries])
        diagnostic, = [r for r in historical["diagnostics"] if r["condition"] == point["condition"]]
        path = run / diagnostic["source"]["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == diagnostic["source"]["sha256"]
        added.update(extractor.diagnostic_fields(read(path), endpoint["counts"]))
        added["projection_sparse_gain"] = added["all_skips_off_candidate_gm_ms"] / added["projection_on_candidate_gm_ms"]
        added["attention_sparse_gain"] = added["projection_on_candidate_gm_ms"] / added["full_candidate_gm_ms"]
    assert len(extra_runtime) == 5 and {r["kappa"] for r in extra_runtime} == set(KAPPAS)
    combined = runtime + extra_runtime
    fits = {key: float(np.corrcoef([r[key] for r in combined], [r["projection_sparse_gain"] for r in combined])[0, 1]**2)
            for key in ("projection_mma_bypass_fraction", "projection_scalar_zero_fraction")}
    return {"sources": sources, "approved_source": approved["sources_sha256"], "coverage": approved["coverage"],
            "loss_policy": "Preserve author-approved Analysis 023 reported losses; h-only ordinary-final, other recipes logical-pass. Both full-validation, same final checkpoints; uniform alternatives retained per row/contrast.",
            "max_loss_pass_difference": approved["maximum_absolute_logical_minus_ordinary_loss"],
            "rows": rows, "contrasts": contrasts, "geometry": histories, "geometry_groups": groups,
            "original_trained": old["trained"], "ceilings": old["ceilings"], "runtime": runtime,
            "h_only_runtime": extra_runtime, "runtime_fits_35": fits, "training_endpoint_count": len(old["trained"]) + 10,
            "runtime_endpoint_count": len(combined), "ablation_endpoint_count": len(combined)}


if __name__ == "__main__":
    data = load()
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data/evidence.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(data["geometry_groups"], indent=2))
    print('Verified 64 training endpoints, 35 full-K050 and ablation endpoints, 14,240 OL1 steps.')
    print(data["runtime_fits_35"])
