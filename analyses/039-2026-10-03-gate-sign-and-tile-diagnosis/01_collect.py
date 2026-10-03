"""Audit retained gate, quality and tile evidence without evaluating a model."""

import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCES = {}


def read_source(relative, expected=None):
    path = ROOT / relative
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        assert digest == expected, f"Source identity changed: {relative}"
    SOURCES[path.relative_to(ROOT).as_posix()] = {"bytes": len(raw), "sha256": digest}
    return raw


def read_json(relative, expected=None):
    return json.loads(read_source(relative, expected))


quality = read_json("analyses/031-2026-09-20-quality-sites-quality/data/figure-data.json")
reconciliation = []
for relative, digest in quality["sources_sha256"].items():
    if not (ROOT / relative).exists():
        assert relative == "manuscript/draft/tables/operation-latency.tex"
        reconciliation.append({"path": relative, "status": "missing_historical_manuscript_table", "expected_sha256": digest})
        continue
    read_source(relative)
    actual = SOURCES[relative]["sha256"]
    if actual != digest:
        assert relative == "analyses/024-2026-09-17-h-only-kernel-latency/data/operation-latency.json"
        current = read_json(relative)
        assert current["rows"] == quality["bar_chart"]["rows"]
        reconciliation.append({"path": relative, "status": "hash_changed_measured_rows_unchanged",
                               "expected_sha256": digest, "current_sha256": actual})

operations = read_json("runs/037-2026-09-19-pythia14m-operation-latency/results/operation-latency.json")
for row in quality["bar_chart"]["rows"]:
    operation = row["operation"]
    saved = operations["latencies_ms"][f"without-{operation}"] - operations["latencies_ms"]["full"]
    assert math.isclose(row["saved_ms"], saved, abs_tol=1e-12)
    without, full = operations["process_ranges_ms"][f"without-{operation}"], operations["process_ranges_ms"]["full"]
    span = [1000 * (without[0] - full[1]), 1000 * (without[1] - full[0])]
    assert all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(span, row["cross_process_difference_span_us"]))

pairs = []
for scale, panel in quality["panels"].items():
    points = {(p["scope"], p["pressure"], p["kappa"]): p for p in panel["points"]}
    for kappa in (0.0, 0.01, 0.05, 0.1, 0.5):
        t2, t4 = points["hz", "h", kappa], points["4", "h", kappa]
        checked_identity = {}
        if scale == "14M":
            for field in ("initial_parameter_sha256", "training_schedule_hash", "training_steps", "training_tokens"):
                assert t2[field] == t4[field], (scale, kappa, field)
                checked_identity[field] = t2[field]
            for p in (t2, t4):
                assert p["coverage"]["sequences"] == 338
                assert p["coverage"]["excluded_tail_tokens"] == 1444
                assert math.isclose(p["sparsity"], 100 * p["zero_product_count"] / p["model_product_count"], abs_tol=1e-12)
        pairs.append({
            "scale": scale, "kappa": kappa,
            "t2_loss": t2["loss"], "t4_loss": t4["loss"],
            "t4_minus_t2_loss": t4["loss"] - t2["loss"],
            "t2_R_model_percent": t2["sparsity"], "t4_R_model_percent": t4["sparsity"],
            "t4_minus_t2_R_model_percentage_points": t4["sparsity"] - t2["sparsity"],
            "t2_checkpoint_key": t2["checkpoint_key"], "t4_checkpoint_key": t4["checkpoint_key"],
            "independently_checked_training_identity": checked_identity,
        })

density_root = "runs/031-2026-09-08-signed-activation-density/results/"
inventory = read_json(density_root + "inventory.json")
density_name = "histograms/14M_A0_None.json.gz"
density = json.loads(gzip.decompress(read_source(density_root + density_name, inventory[density_name]["sha256"])))
assert density["grid"]["bins"] == 16000
assert density["grid"]["lower"] == -8 and density["grid"]["upper"] == 8
assert density["coverage"]["sequences"] == 338
assert density["coverage"]["excluded_tail_tokens"] == 1444
m = density["sites"]["m"]
rows = [r for r in density["rows"] if r["name"].startswith("m.layer_")]
assert len(rows) == 6 and m["nonfinite"] == 0
assert sum(r["total"] for r in rows) == m["total"]
assert all(sum(r["histogram"][i] for r in rows) == m["histogram"][i] for i in range(16000))
negative = sum(m["histogram"][:8000]) + m["underflow"]
positive = sum(m["histogram"][8000:]) + m["overflow"]
assert negative + positive + m["exact_zero_count"] == m["total"]
baseline_m = {
    "source_checkpoint_key": density["source"]["key"], "coverage": density["coverage"],
    "precision": "FP32 parameters, FP16 CUDA autocast, eager",
    "elements": m["total"], "negative_count": negative, "positive_count": positive,
    "exact_zero_count": m["exact_zero_count"], "negative_fraction": negative / m["total"],
    "near_zero_counts_abs_le_epsilon": m["threshold_hits"],
    "near_zero_fraction_abs_le_0p01": m["threshold_hits"]["0.01"] / m["total"],
    "mean": m["sum"] / m["finite"], "rms": m["rms"],
    "minimum": m["minimum"], "maximum": m["maximum"],
    "limits": "Baseline m only. Trained pre-gate a/m/z values and removed energy are not reconstructed. Near-zero counters use <=, whereas gate rejection uses <.",
}

run39 = "runs/039-2026-09-19-pythia14m-am-port-fixture/"
structure = read_json(run39 + "results/mechanism.json")
read_source(run39 + structure["source"]["path"], structure["source"]["sha256"])
for site, row in structure["sites"].items():
    assert math.isclose(row["sparsity"], row["zeros"] / row["elements"], abs_tol=1e-12)
    if "tiles" in row:
        for size in (8, 16):
            t = row["tiles"]
            assert math.isclose(t[f"empty_fraction_{size}x16"], t[f"empty{size}"] / t[f"total{size}"], abs_tol=1e-12)

run56 = "runs/056-2026-09-25-pythia14m-mechanism-ablation/"
mechanisms = read_json(run56 + "results/mechanism-latency.json")
assert mechanisms["status"] == "qualified" and mechanisms["scientific_processes"] == 15
for source in mechanisms["sources"]:
    read_source(run56 + source["path"], source["sha256"])
latencies = mechanisms["latencies_ms"]
tile_saving = latencies["t00"] - latencies["t10"]
assert math.isclose(tile_saving, mechanisms["effects"]["tile_alone_ms"], abs_tol=1e-12)

gates = []
for path in sorted((ROOT / "runs").glob("*/config.yaml")):
    config = yaml.safe_load(read_source(path.relative_to(ROOT).as_posix()))
    model = config.get("model", {})
    if not isinstance(model, dict):
        continue
    per_site, uniform = model.get("site_gates"), model.get("site_gate")
    if per_site or uniform:
        gates.append({"path": path.relative_to(ROOT).as_posix(), "topology": model.get("topology_id"),
                      "site_gates": per_site, "site_gate": uniform})
read_source("src/sparsity_research/sites.py")
read_source("src/sparsity_research/pythia.py")

report = {
    "scope": "Retained-artifact audit only; no new model evaluation, training, kernel timing or manuscript change.",
    "quality_pairs": pairs, "baseline_m": baseline_m, "trained_structure": structure,
    "conditional_site_effects": quality["bar_chart"]["rows"],
    "tile_mechanism": {"latencies_ms": latencies, "effects": mechanisms["effects"],
                       "tile_alone_reduction_fraction_vs_padded_fallback": tile_saving / latencies["t00"],
                       "process_ranges_ms": mechanisms["process_ranges_ms"],
                       "limits": "14M T7/Pall kappa=.5, h/z jointly; padded custom fallback, not optimal dense execution."},
    "root_config_gate_audit": gates,
    "gate_audit_limits": "Root config.yaml declarations only; not an exhaustive search of executed attempt configs or archived candidates.",
    "historical_source_reconciliation": reconciliation,
    "sources": dict(sorted(SOURCES.items())),
}
target = HERE / "data" / "diagnosis.json"
if "--check" in sys.argv:
    assert json.loads(target.read_text(encoding="utf-8")) == report, "Saved audit differs from current sources"
else:
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Verified {len(pairs)} quality pairs, six pooled m layers, integer tile ratios and 15 retained timing processes; {len(SOURCES)} source hashes.")
print(f"Baseline m: {100 * baseline_m['negative_fraction']:.5f}% negative; {100 * baseline_m['near_zero_fraction_abs_le_0p01']:.5f}% within +/-0.01.")
print(f"Tile-only reduction versus padded fallback: {100 * tile_saving / latencies['t00']:.5f}%.")
