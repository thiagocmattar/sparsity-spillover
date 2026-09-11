"""Reconcile retained K050 counters and raw timing pairs; no model execution."""

import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / "runs/029-2026-09-07-pythia14m-matched-kernel-retrospective"
R28 = RUN / "archive/root/runs/028-2026-09-06-pythia14m-all-site-sparse-kernels"
SOURCE = ROOT / "analyses/018-2026-09-08-results-materials/figure_data.json"
OPS = {"qkv": "qkv_projection", "qk": "qk_scores", "pv": "probability_value",
       "attn_out": "attention_output_projection", "ffn_up": "mlp_w1", "ffn_down": "mlp_w2"}
PROJECTIONS = ("qkv", "attn_out", "ffn_up", "ffn_down")
SITES = {"qkv": "a", "attn_out": "z", "ffn_up": "m", "ffn_down": "h"}
CANDIDATES = {"all_skips_off": "k050-no-skip", "projection_on": "k050-attention-dense", "full": "k050"}


def gm(values):
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def timing_summary(timing):
    modes = {mode: {} for mode in ("native_graph", "candidate_graph")}
    expected = {(repeat, index) for repeat in range(7) for index in range(64)}
    assert len(set(timing["indices"])) == 64
    for sample in timing["samples"]:
        key = sample["repeat"], sample["input_index"]
        target = modes[sample["mode"]]
        assert key not in target and sample["output_shape"] == [1, 2048, 50304]
        value = sample["host_ms"]
        assert math.isfinite(value) and value > 0
        target[key] = value
    assert all(set(values) == expected for values in modes.values())
    native = [modes["native_graph"][key] for key in sorted(expected)]
    candidate = [modes["candidate_graph"][key] for key in sorted(expected)]
    return {"native_gm_ms": gm(native), "candidate_gm_ms": gm(candidate),
            "paired_speedup": gm([n / c for n, c in zip(native, candidate)])}


def diagnostic_fields(diagnostic, canonical):
    assert diagnostic["status"] == "complete"
    assert diagnostic["coverage"] == dict(blocks=338, documents=500, input_tokens=692224, excluded_tail_tokens=1444)
    runtime = diagnostic["bf16_scalar_opportunity_lower_bound"]
    model = canonical["model_product_count"]
    assert runtime["model_products"] == model
    fields = {"model_products": model, "s_model_percent": 100 * canonical["block_zero_product_count"] / model}
    for short, name in OPS.items():
        c = canonical["per_operation"][name]
        b = runtime["per_operation"][name]
        assert c["product_count"] == b["product_count"]
        fields.update({f"{short}_products": c["product_count"],
                       f"{short}_zero_products": c["zero_product_count"],
                       f"{short}_scalar_zero_fraction": c["zero_product_count"] / c["product_count"],
                       f"{short}_model_contribution_pp": 100 * c["zero_product_count"] / model,
                       f"{short}_bf16_zero_products_lower_bound": b["zero_product_count"],
                       f"{short}_bf16_scalar_zero_fraction_lower_bound": b["zero_product_count"] / b["product_count"]})
    assert sum(fields[f"{op}_zero_products"] for op in OPS) == canonical["block_zero_product_count"]
    for label, operations in (("projection", PROJECTIONS), ("attention", ("qk", "pv"))):
        zero = sum(fields[f"{op}_zero_products"] for op in operations)
        count = sum(fields[f"{op}_products"] for op in operations)
        fields.update({f"{label}_products": count, f"{label}_zero_products": zero,
                       f"{label}_scalar_zero_fraction": zero / count,
                       f"{label}_model_contribution_pp": 100 * zero / model})
    for short in PROJECTIONS:
        b = runtime["per_operation"][OPS[short]]
        issued, bypassed = b["issued_mmas"], b["skipped_mmas"]
        potential = issued + bypassed
        padded = short in ("ffn_down", "attn_out")
        assert potential * (1024 if padded else 2048) == fields[f"{short}_products"]
        fields.update({f"{short}_mma_issued": issued, f"{short}_mma_bypassed": bypassed,
                       f"{short}_mma_potential": potential,
                       f"{short}_mma_bypass_fraction": bypassed / potential,
                       f"{short}_simt_products": b.get("simt_products", 0)})
        histograms = [hist for name, hist in diagnostic["active_features_per_row"].items()
                      if name.startswith(SITES[short] + ".")]
        assert len(histograms) == 6 and all(sum(h) == 338 * 2048 for h in histograms)
        rows = sum(sum(h) for h in histograms)
        fields.update({f"{short}_rows": rows,
                       f"{short}_zero_rows": sum(h[0] for h in histograms),
                       f"{short}_rows_le2": sum(sum(h[:3]) for h in histograms),
                       f"{short}_zero_row_fraction": sum(h[0] for h in histograms) / rows,
                       f"{short}_row_le2_fraction": sum(sum(h[:3]) for h in histograms) / rows})
        if padded:
            expected_simt_upper = sum(h[1] + 2 * h[2] for h in histograms) * 128
            assert b["simt_products"] <= expected_simt_upper
            fields[f"{short}_simt_opportunity_products"] = expected_simt_upper
            fields[f"{short}_simt_safety_gap_products"] = expected_simt_upper - b["simt_products"]
        else:
            n = 384 if short == "qkv" else 512
            assert bypassed % (n // 8) == 0
            fields[f"{short}_zero_16x16_fragments"] = bypassed // (n // 8)
            fields[f"{short}_total_16x16_fragments"] = potential // (n // 8)
    hybrid = [sum(v[i] for v in diagnostic["hybrid_counts_by_layer"].values()) for i in range(6)]
    assert hybrid == [fields[f"{op}_{metric}"] for op, metric in (
        ("ffn_down", "mma_issued"), ("ffn_down", "mma_bypassed"),
        ("attn_out", "mma_issued"), ("attn_out", "mma_bypassed"),
        ("ffn_down", "simt_products"), ("attn_out", "simt_products"))]
    fields["projection_mma_potential"] = sum(fields[f"{op}_mma_potential"] for op in PROJECTIONS)
    fields["projection_mma_bypassed"] = sum(fields[f"{op}_mma_bypassed"] for op in PROJECTIONS)
    fields["projection_mma_bypass_fraction"] = fields["projection_mma_bypassed"] / fields["projection_mma_potential"]
    fields["projection_simt_products"] = sum(fields[f"{op}_simt_products"] for op in PROJECTIONS)
    # Unpadded arithmetic-equivalent support proxy, not time saved or hardware FLOPs.
    bypass_equivalent = sum(fields[f"{op}_mma_bypassed"] * (1024 if op in ("attn_out", "ffn_down") else 2048)
                            for op in PROJECTIONS)
    fields["projection_arithmetic_avoidance_proxy_fraction"] = (bypass_equivalent - fields["projection_simt_products"]) / fields["projection_products"]
    fields["projection_pure_zero_only_bypass_fraction"] = None
    fields["projection_timing_subset_bypass_fraction"] = None
    fields["projection_mma_counter_method"] = "qkv/up: BF16 operand reconstruction; down/out: instrumented; includes SIMT substitution"
    attention = [sum(v[i] for v in diagnostic["attention_counts_by_layer"].values()) for i in range(4)]
    for op, offset in (("qk", 0), ("pv", 2)):
        issued, skipped = attention[offset:offset + 2]
        assert issued + skipped == 147456 * 338 * 6
        fields.update({f"{op}_mma_issued": issued, f"{op}_mma_skipped": skipped,
                       f"{op}_mma_potential": issued + skipped,
                       f"{op}_mma_skip_fraction": skipped / (issued + skipped)})
    fields["attention_mma_potential"] = sum(attention)
    fields["attention_mma_skipped"] = attention[1] + attention[3]
    fields["attention_mma_skip_fraction"] = fields["attention_mma_skipped"] / sum(attention)
    fields["attention_unmasked_only_mma_skip_fraction"] = None
    fields["attention_timing_subset_mma_skip_fraction"] = None
    return fields


def extract():
    records = {}

    def read(path, expected=None):
        content = path.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        if expected is not None:
            assert sha == expected, path
        records[path.relative_to(ROOT).as_posix()] = {"sha256": sha, "bytes": len(content)}
        return json.loads(content)

    source = read(SOURCE)
    retrospective_path = RUN / "results/matched-retrospective-001.json"
    raw = read(retrospective_path, source["sources"][retrospective_path.relative_to(ROOT).as_posix()])
    supplemental_path = ROOT / "manuscript/draft/supplementary-data/figure-data.json"
    supplemental_manifest = read(ROOT / "manuscript/draft/supplementary-data/SOURCES.json")
    supplemental = read(supplemental_path, supplemental_manifest["figure-data.json"]["sha256"])
    assert supplemental["runtime"] == source["runtime"]
    source_records = {item["path"]: item["sha256"] for item in raw["sources"]}
    selected = sorted([p for p in source["runtime"]["points"] if p["candidate"] == "k050"], key=lambda p: p["condition"])
    assert [p["condition"] for p in selected] == [f"c{i:02d}" for i in range(1, 31)]
    points = {(p["condition"], p["candidate"]): p for p in source["runtime"]["points"]}
    trained = {p["id"]: p for p in source["trained"]}
    diagnostics = {p["condition"]: p for p in raw["diagnostics"]}
    rows, layers = [], []
    for full in selected:
        condition = full["condition"]
        checkpoint = trained[full["evidence_id"]]
        family = "4-site" if full["family"].startswith("A4") else "7-site" if full["family"].startswith("A7") else "baseline/local"
        row = {"condition": condition, "recipe": full["family"], "family": family,
               "pressure": checkpoint["identity"]["pressure"]["method"],
               "kappa": checkpoint["dose"] if family != "baseline/local" else None,
               "pressure_weight": checkpoint["identity"]["pressure"].get("weight"),
               "checkpoint_evidence_id": full["evidence_id"], "validation_loss": checkpoint["loss"]}
        diagnostic_record = diagnostics[condition]["source"]
        diagnostic_path = RUN / diagnostic_record["path"]
        diagnostic = read(diagnostic_path, diagnostic_record["sha256"])
        row.update(diagnostic_fields(diagnostic, full["canonical_counts"]))
        row["diagnostic_source"] = diagnostic_path.relative_to(ROOT).as_posix()
        for layer, hybrid in diagnostic["hybrid_counts_by_layer"].items():
            layers.append({"condition": condition, "layer": int(layer),
                           **dict(zip(diagnostic["hybrid_counter_fields"], hybrid)),
                           **dict(zip(diagnostic["attention_counter_fields"], diagnostic["attention_counts_by_layer"][layer]))})
        for label, candidate in CANDIDATES.items():
            point = points[condition, candidate]
            assert point["qualified"] and len(point["replicates"]) == 3
            summaries = []
            for replicate in sorted(point["replicates"], key=lambda r: r["replicate"]):
                assert replicate["qualified"] and replicate["status"] == "complete"
                timing_path = RUN / "artifacts/attempts" / replicate["attempt"] / "timing.json"
                timing = read(timing_path, source_records[timing_path.relative_to(RUN).as_posix()])
                assert timing["indices"] == raw["timing_block_indices"]
                summary = timing_summary(timing)
                assert math.isclose(summary["paired_speedup"], replicate["speedup"], rel_tol=1e-12)
                summaries.append(summary)
            for field in summaries[0]:
                row[f"{label}_{field}"] = gm([s[field] for s in summaries])
            assert math.isclose(row[f"{label}_paired_speedup"], point["speedup"], rel_tol=1e-12)
        row["native_baseline_gm_ms"] = row["full_native_gm_ms"]
        row["fusion_only_speedup_common_native"] = row["native_baseline_gm_ms"] / row["all_skips_off_candidate_gm_ms"]
        row["projection_sparse_gain"] = row["all_skips_off_candidate_gm_ms"] / row["projection_on_candidate_gm_ms"]
        row["attention_sparse_gain"] = row["projection_on_candidate_gm_ms"] / row["full_candidate_gm_ms"]
        row["full_speedup"] = row["full_paired_speedup"]
        row["projection_sparse_gain_native_normalized"] = row["projection_on_paired_speedup"] / row["all_skips_off_paired_speedup"]
        row["attention_sparse_gain_native_normalized"] = row["full_paired_speedup"] / row["projection_on_paired_speedup"]
        assert math.isclose(row["fusion_only_speedup_common_native"] * row["projection_sparse_gain"] * row["attention_sparse_gain"], row["full_speedup"], rel_tol=1e-12)
        rows.append(row)
    # Pin exactly the source files inspected for counter and dispatch semantics.
    archive = read(RUN / "provenance/archive.json")
    archive_hashes = {r["snapshot"]["path"]: r["snapshot"]["sha256"] for r in archive["files"]}
    paths = [R28 / p for p in ("115_hybrid_diagnostics.py", "45_graph_forward.py", "candidates/k050/candidate.py",
             "candidates/k042/candidate.py", "candidates/k042/projection.cu", "candidates/k049/candidate.py",
             "candidates/k049/joint.cu", "candidates/k035/candidate.py", "candidates/k035/sparse_gemm.h", "candidates/k035/kernel.cu")]
    paths += [RUN / "archive/root/src/sparsity_research" / f for f in ("sites.py", "pythia.py")]
    paths += [RUN / "archive/root/runs/027-2026-09-06-pythia14m-kernel-sparsity-characterization/adapter.py"]
    paths += [RUN / "archive/root/runs/026-2026-09-06-pythia14m-fused-sparse-kernel/autoresearch/candidates/k019" / f
              for f in ("candidate.py", "kernel.cu")]
    for path in paths:
        content = path.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        assert sha == archive_hashes[path.relative_to(RUN).as_posix()], path
        records[path.relative_to(ROOT).as_posix()] = {"sha256": sha, "bytes": len(content)}
    replay_path = RUN / "replay.py"
    content = replay_path.read_bytes()
    records[replay_path.relative_to(ROOT).as_posix()] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    metadata = {"sources": records, "cohort": [r["condition"] for r in rows],
                "timing": "64 inputs x 7 paired passes x 3 fresh processes per implementation; geometric mean host latency",
                "diagnostic_coverage": "338 full-validation blocks, untimed BF16 pass; canonical scalar counts FP16",
                "supplementary_runtime_identical": True, "timing_block_indices": raw["timing_block_indices"]}
    return rows, layers, metadata


def main():
    rows, layers, metadata = extract()
    output = HERE / "data"
    output.mkdir(exist_ok=True)
    write_csv(output / "checkpoints.csv", rows)
    write_csv(output / "layer-counters.csv", layers)
    (output / "checkpoints.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Extracted {len(rows)} checkpoints, {len(rows[0])} columns, {len(layers)} layer records; {len(metadata['sources'])} source hashes recorded")


if __name__ == "__main__":
    main()
