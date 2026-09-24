"""Add the current paper's three-scale evidence without altering historical rows."""

import gzip
import hashlib
import json
import re
import csv
import io
import sys
from release_results import condition, clean_results


def extend_results(root, out, analysis, run, save, copy, provenance):
    def read(path, target):
        raw = path.read_bytes()
        provenance.append(dict(path=target, source=path.relative_to(root).as_posix(),
                               source_sha256=hashlib.sha256(raw).hexdigest(),
                               changes=["Select measured evidence; replace archive labels with scientific identifiers"]))
        return json.loads(raw)

    def write(target, value):
        raw = (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()
        save(target, gzip.compress(raw, mtime=0) if target.endswith(".gz") else raw)

    def csv_table(name, fields, rows, source):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        save("results/" + name + ".csv", stream.getvalue().encode())
        provenance.append(dict(path="results/" + name + ".csv", sources=[source],
                               changes=["Direct readable table of bundled measured values"]))

    sessions = {"Run054": "31m-base-t2", "Run055": "31m-seven-site-h-pressure",
                "Run045": "70m-main", "Run047": "70m-additional-endpoint",
                "Run029": "14m-main", "Run041": "14m-two-site",
                "Run044": "14m-two-site-high-threshold", "Run033": "14m-seven-site-h-pressure"}

    def clean(value):
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()
                    if k not in {"source_attempt", "source_sha256", "sources_sha256", "source_script_sha256",
                                 "source", "script", "output", "output_sha256", "script_sha256",
                                 "style_reference", "style_reference_sha256", "attempt", "gpu_uuid"}
                    and not (isinstance(v, str) and any(s in v for s in ("runs/", "analyses/", "/workspace/")))}
        if isinstance(value, str):
            for a, b in sessions.items():
                value = value.replace(a, b)
            return re.sub(r"opt073(?:-31m(?:-t7)?-v1)?", "specialized", value)
        return value

    endpoints = json.loads((out / "results/endpoints.json").read_text())
    originals = json.dumps(endpoints["trained_points"], sort_keys=True)
    timing_processes, diagnostics = [], []
    for number, filename, run_number, session in [
        (37, "31m-results.json", 54, "31m-base-t2"),
        (38, "31m-t7-results.json", 55, "31m-seven-site-h-pressure"),
    ]:
        cohort = read(analysis(number) / "data" / filename, "results/endpoints.json")
        for point in cohort["points"]:
            name = condition(point)
            attempt = root / point["source_attempt"]
            manifest = read(attempt / "manifest.json", "results/endpoints.json")
            metrics = read(attempt / "metrics.json", "results/31m-diagnostics.json")
            logical = read(attempt / "diagnostics/logical_products.json", "results/31m-diagnostics.json")
            measured = logical["measured"]
            row = {k: point[k] for k in ["model", "scope", "pressure", "kappa", "loss", "checkpoint_key",
                                       "coverage", "latency_ms", "implementation_latency_ms", "latency_process_ms"]}
            row.update(condition=name, local_pressure_weight=None,
                       sparsity=100 * measured["block_zero_product_count"] / measured["model_product_count"],
                       zero_product_count=measured["block_zero_product_count"], model_product_count=measured["model_product_count"],
                       kernel="specialized-31m", timing_session=session,
                       initial_parameter_sha256=manifest["initial_parameter_sha256"],
                       training_schedule_hash=manifest["data"]["training_schedule_hash"],
                       final_checkpoint_content_sha256=manifest["checkpoints"]["final"]["content_sha256"],
                       training_steps=manifest["completed_steps"], training_tokens=manifest["input_tokens"],
                       final_learning_rate=0.0001, logical_pass_loss=logical["coverage"]["loss"])
            endpoints["trained_points"].append(row)
            diagnostics.append(dict(condition=name, checkpoint_key=point["checkpoint_key"], logical_products=logical,
                                    validation=metrics["validation"], initialization=metrics["recipe_mapping"]["initialization"]))
            for process in point["processes"]:
                folder = run(run_number) / "latency/artifacts/attempts" / process["attempt"]
                target = "results/31m-timing-processes.json.gz"
                m = read(folder / "manifest.json", target)
                q = read(folder / "quality.json", target)
                t = read(folder / "timing.json", target)
                timing_processes.append(dict(condition=name, checkpoint_key=point["checkpoint_key"], session=session,
                    replicate=process["replicate"], quality=q, timing=t, runtime=clean(m["runtime"]),
                    qualification_config=clean(m["config"]), validation_sha256=m["cache_sha256"],
                    measured_source_identity_sha256=m["source_identity"]["content_sha256"]))
    assert json.dumps(endpoints["trained_points"][:84], sort_keys=True) == originals
    evidence = read(analysis(38) / "data/manuscript-evidence.json", "results/scale-evidence.json")
    write("results/scale-evidence.json", clean({k: evidence[k] for k in ["architecture", "matched_identity", "training",
        "tokens_per_parameter", "ceilings_31m", "rows_31m", "base_kernel_latency_ms", "base_native_latency_ms",
        "illustrative_kappa", "illustrative_selection_rule", "cross_scale_claims"]}))
    sys.path.insert(0, str(root / "src"))
    from sparsity_research.ceilings import architecture_ceiling
    from sparsity_research.sites import TOPOLOGIES, Topology
    TOPOLOGIES["HZ"] = Topology("HZ", ("h", "z"))
    endpoints["ceilings"]["31M"] = {}
    for scope, topology, paper in [("0", "A0", "T0"), ("1", "A1-H", "T1"), ("hz", "HZ", "T2"),
                                   ("4", "A4-Z", "T4"), ("7", "A7-Z-POST", "T7")]:
        ceiling = architecture_ceiling(topology, layers=6, hidden_size=256, ffn_size=1024,
                                       sequence_length=2048, vocabulary_size=50304)
        assert ceiling["reachable_product_count"] == evidence["ceilings_31m"][paper]["reachable_product_count"]
        assert ceiling["model_product_count"] == evidence["ceilings_31m"][paper]["model_product_count"]
        endpoints["ceilings"]["31M"][scope] = ceiling
    sys.path.remove(str(root / "src"))
    write("results/endpoints.json", endpoints)
    write("results/31m-diagnostics.json", diagnostics)
    write("results/31m-timing-processes.json.gz", timing_processes)
    scale = read(analysis(38) / "data/scale-figure.json", "results/scale-frontier.json")
    scale = clean(scale)
    for row in scale["executions"]:
        row["condition"] = condition(row)
        endpoint = next(r for r in endpoints["trained_points"] if r["condition"] == row["condition"])
        row.setdefault("timing_session", endpoint["timing_session"])
    write("results/scale-frontier.json", scale)
    csv_table("scale-frontier", ["condition", "model", "scope", "pressure", "kappa", "backend", "loss", "latency_ms",
                               "timing_session", "checkpoint_key"], scale["executions"], "results/scale-frontier.json")
    delta_rows = clean(evidence["rows_31m"])
    csv_table("31m-results", list(delta_rows[0]), delta_rows, "results/scale-evidence.json")
    endpoint_fields = ["condition", "model", "loss", "sparsity_percent", "latency_ms", "timing_session",
                       "zero_product_count", "model_product_count", "checkpoint_key"]
    csv_table("endpoints", endpoint_fields,
              [dict(r, sparsity_percent=r["sparsity"]) for r in endpoints["trained_points"]], "results/endpoints.json")

    source = read(analysis(33) / "data/three-panel.json", "results/14m-paper-figure.json")
    clean_points = clean_results("14m-figure", dict(series=[], trained_points=source["points"], clipping_points=[], ceilings={}),
                                {"endpoints": analysis(30) / "data/full-trained-results.json"})["trained_points"]
    write("results/14m-paper-figure.json", dict(points=clean_points,
          panels=clean(source["panels"]), scope=source["scope"], timing=source["timing"],
          loss=source["loss"], sparsity=source["sparsity"], verification=source["verification"]))

    hz = read(run(48) / "results/hz-latency.json", "results/14m-t2-execution-controls.json")
    hz = {k: hz[k] for k in ["latency", "effects", "processes", "all_logits_equal_to_eager",
                            "zero_patterns_and_values_identical", "diagnostic_counters"]}
    hz.update(condition="14M-T2-Ph-0.1", session="14m-t2-execution-controls",
              modes={"A": dict(skip_h=True, skip_z=True), "B": dict(skip_h=False, skip_z=True),
                     "C": dict(skip_h=True, skip_z=False), "D": dict(skip_h=False, skip_z=False)},
              counter_order=["h_mma_issued", "h_mma_bypassed", "z_mma_issued", "z_mma_bypassed", "h_simt_products", "z_simt_products"])
    engineering = read(run(48) / "results/engineering-comparison.json", "results/14m-t2-execution-controls.json")
    hz["engineering_comparison"] = clean(engineering)
    write("results/14m-t2-execution-controls.json", hz)

    figures = [
        ("04-14m-quality-savings-sparsity.pdf", "01-quality-savings-sparsity.pdf"),
        ("pythia-architecture-map.pdf", "02-architecture.pdf"),
        ("appendix-14m-layer-sparsity.pdf", "03-layer-sparsity.pdf"),
        ("18-14m-70m-ol1-geometry.pdf", "04-ol1-geometry.pdf"),
        ("02-14m-70m-latency-quality.pdf", "05-scale-frontier.pdf"),
        ("19-base-model-optimization.pdf", "06-base-training.pdf"),
    ]
    for original, public in figures:
        copy(root / "manuscript/draft/figures" / original, "figures/" + public)

    # Keep the exact plotting body used for Figure 5, with portable data/output paths.
    source = analysis(38) / "02_plot.py"
    text = source.read_text()
    header = text[:text.index("HERE=")]
    body = text[text.index("    markers="):text.index("    assert sha(old_pdf)", text.index("    markers="))]
    body = body.replace("HERE/'figures/01-base-t2-t7-scale-frontier.pdf'", "ROOT/'reproduced/scale-frontier.pdf'")
    setup = '''ROOT=Path(__file__).resolve().parents[1]
TITLE='Quality\\u2013latency trade-offs across model scales'

def main():
    data=json.loads((ROOT/'results/scale-frontier.json').read_text())
    executions=data['executions']
    refs=[p for p in executions if p['scope']=='0']
    families=[('hz','h'),('7','h')]
'''
    save("scripts/plot_scale.py", (header + setup + body + "\nif __name__=='__main__':main()\n").encode())
    provenance.append(dict(path="scripts/plot_scale.py", source=source.relative_to(root).as_posix(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        changes=["Retain exact Figure 5 drawing code; use bundled coordinates and local output"] ))
