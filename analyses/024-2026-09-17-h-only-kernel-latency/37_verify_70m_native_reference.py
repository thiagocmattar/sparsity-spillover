"""Verify Figure 3's native reference directly from the retained paired timings."""
import hashlib
import json
import math
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data_path = HERE / "data/70m-quality-sparsity-native-latency.json"
    data = json.loads(data_path.read_text())
    original = json.loads((HERE / "data/14m-70m-quality-sparsity-latency.json").read_text())
    audit = json.loads((HERE / "data/kernel-appendix.json").read_text())
    expected = [r for r in original["trained_points"] if r["model"] == "70M"]
    assert data["trained_points"] == expected and len(expected) == 22
    native_samples = []
    for relative, digest in audit["sources_sha256"].items():
        if "runs/035-" not in relative or not relative.endswith("timing.json"):
            continue
        path = ROOT / relative
        assert sha(path) == digest
        timing = json.loads(path.read_text())
        rows = [s for s in timing["samples"] if s["mode"] == "native_graph"]
        assert len(rows) == 448
        assert {(s["input_index"], s["repeat"]) for s in rows} == {
            (i, r) for i in range(64) for r in range(7)}
        assert all(s["output_shape"] == [1, 2048, 50304] for s in rows)
        native_samples.extend(s["host_ms"] for s in rows)
    assert len(native_samples) == 1344
    native_ms = math.exp(math.fsum(map(math.log, native_samples)) / len(native_samples))
    assert native_ms == data["native_base_reference"]["native_ms"]
    for raw, plotted in zip(expected, data["plotted_points"]):
        assert all(plotted[k] == value for k, value in raw.items())
        assert plotted["displayed_latency_ms"] == (native_ms if raw["scope"] == "0" else raw["latency_ms"])
        assert plotted["execution"] == ("native PyTorch/SDPA" if raw["scope"] == "0" else "original 70M port")
    assert data["panels"][0]["trained_keys"] == data["panels"][1]["trained_keys"]
    assert len(data["panels"][0]["clipping_ids"]) == 20
    assert data["panels"][1]["clipping_ids"] == []
    for relative, digest in data["sources_sha256"].items():
        assert sha(ROOT / relative) == digest
    assert sha(HERE / data["script"]) == data["script_sha256"]
    manuscript = ROOT / "manuscript/draft"
    pdf = HERE / data["output"]
    assert sha(pdf) == data["output_sha256"] == sha(manuscript / "figures" / pdf.name)
    assert sha(data_path) == sha(manuscript / "supplementary-data" / data_path.name)
    doc = fitz.open(pdf)
    assert len(doc) == 1
    page = doc[0]
    assert "PyTorch base: 1.664 ms" in page.get_text()
    assert len(page.search_for("Post-hoc")) == 1
    assert all(page.rect.contains(fitz.Rect(b[:4])) for b in page.get_text("blocks"))
    assert all(doc.extract_font(f[0])[3] for f in page.get_fonts(full=True))
    exp = (manuscript / "experimental-study.tex").read_text()
    training = (manuscript / "training-results.tex").read_text()
    appendix = (manuscript / "kernel-appendix.tex").read_text()
    assert r"\ref{app:410m-results}" in exp and r"\ref{app:kernel-transfer}" in exp
    assert "13-14m-70m-sparsity-base-speedup.pdf" not in training
    assert "fig:base-model-speedup" not in training + appendix
    followup = data["followup_not_plotted"]
    assert followup["device_uuid"] != data["native_base_reference"]["device_uuid"]
    for row in followup["rows"]:
        if row["mode"] not in ("full", "opt002"):
            continue
        latency = row["geometric_means_ms"]["candidate_graph"]["host_ms"]
        assert f"{latency:.3f}" in appendix
        assert f"{followup['native_base_ms'] / latency:.3f}" in appendix
    result = dict(status="verified", trained_points=22, clipping_points_per_panel=[20, 0],
                  native_reference_ms=native_ms, primary_native_timing_samples=1344,
                  unchanged_source_measurements=True, separate_optimization_session=True,
                  appendix_table_checked=True, speedup_figure_removed=True,
                  pdf_sha256=sha(pdf), verifier_sha256=sha(Path(__file__)))
    (HERE / "data/70m-native-reference-verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
