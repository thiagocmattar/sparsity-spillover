"""Audit membership, immutable measurements, manuscript copies and rendered PDFs."""
import collections
import hashlib
import json
import math
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "analyses/024-2026-09-17-h-only-kernel-latency"
DRAFT = ROOT / "manuscript/draft"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    figure = read(HERE / "data/figure-data.json")
    full = read(HERE / "data/full-trained-results.json")
    previous = read(OLD / "data/14m-main-quality-sparsity-latency.json")
    old_full = read(OLD / "data/all-model-quality-sparsity.json")
    assert figure["trained_points"][:40] == previous["trained_points"]
    assert full["trained_points"][:74] == old_full["trained_points"]
    assert figure["clipping_points"] == previous["clipping_points"]
    assert full["clipping_points"] == old_full["clipping_points"]
    assert figure["limits"] == previous["limits"] and figure["layout"] == previous["layout"]
    rows = full["trained_points"]
    assert len(rows) == len({r["source_attempt"] for r in rows}) == 79
    assert dict(collections.Counter(r["model"] for r in rows)) == {"14M": 45, "70M": 22, "410M": 12}
    assert {r["checkpoint_key"] for r in figure["trained_points"]} == set(full["main_figure_checkpoint_keys"])
    assert len(full["main_figure_checkpoint_keys"]) == 41
    hz = sorted((r for r in rows if r["scope"] == "hz"), key=lambda r: r["kappa"])
    assert [r["kappa"] for r in hz] == [0, .01, .05, .1, .5]
    for row in rows:
        attempt = ROOT / row["source_attempt"]
        metrics = read(attempt / "metrics.json")
        logical = read(attempt / "diagnostics/logical_products.json")
        manifest = read(attempt / "manifest.json")
        assert row["loss"] == metrics["validation"]["final"]["loss"]
        assert row["final_checkpoint_content_sha256"] == manifest["checkpoints"]["final"]["content_sha256"]
        assert row["zero_product_count"] == logical["measured"]["block_zero_product_count"]
        assert row["model_product_count"] == logical["measured"]["model_product_count"]
        assert math.isclose(row["sparsity"], 100 * row["zero_product_count"] / row["model_product_count"], rel_tol=1e-12)
        assert all(metrics["validation"]["final"][k] == v for k, v in full["coverage"].items())
    for path, expected in full["sources_sha256"].items():
        assert sha(ROOT / path) == expected, path
    assert sha(HERE / figure["output"]) == figure["output_sha256"]
    assert sha(HERE / figure["script"]) == figure["script_sha256"]
    for folder, name in [
        ("figures", "22-14m-main-quality-sparsity-latency.pdf"),
        ("supplementary-data", "14m-main-quality-sparsity-latency.json"),
        ("supplementary-data", "all-model-quality-sparsity.json"),
    ]:
        record = read(DRAFT / folder / "SOURCES.json")[name]
        assert sha(DRAFT / folder / name) == sha(ROOT / record["source"]) == record["sha256"]
    assert (DRAFT / "tables/compact-results/14m-only.tex").read_bytes() == (HERE / "tables/14m-only.tex").read_bytes()
    table = (HERE / "tables/14m-only.tex").read_text()
    for row in hz:
        assert f"{row['kappa']:g} & {row['loss']:.4f} & {row['sparsity']:.3f}" in table
    introduction = (DRAFT / "introduction.tex").read_text()
    assert "same 41 checkpoints" in introduction and "Across 41 14M-parameter" in introduction
    assert r"$\kappa=0,.01,.05,.1,.5$ for $T_2/P_h$" in introduction
    appendix = (DRAFT / "results-appendix.tex").read_text()
    assert "report all 79" in appendix and r"$T_2/P_h$, $T_4/P_h$ or $T_7/P_h$" in appendix
    doc = fitz.open(DRAFT / "main.pdf")
    assert len(doc) == 21
    text = "\n".join(p.get_text() for p in doc)
    assert "same 41 checkpoints" in text and "all 79" in text
    assert "5.5363" in doc[15].get_text() and "5.339" in doc[15].get_text()
    assert "??" not in text
    font_records = {font[0]: font for page in doc for font in page.get_fonts(full=True)}
    assert all(font[2] != "Type3" and doc.extract_font(xref)[3] for xref, font in font_records.items())
    figure_doc = fitz.open(HERE / figure["output"])
    assert len(figure_doc) == 1
    assert all(figure_doc[0].rect.contains(fitz.Rect(block[:4])) for block in figure_doc[0].get_text("blocks"))
    result = dict(
        status="verified", main_checkpoints=41, complete_checkpoints=79,
        complete_by_size={"14M": 45, "70M": 22, "410M": 12},
        previous_main_rows_unchanged=40, previous_complete_rows_unchanged=74,
        old_clipping_records_unchanged=True, all_79_raw_losses_counts_and_checkpoint_identities_verified=True,
        t2_grid=[r["kappa"] for r in hz], source_hashes_checked=len(full["sources_sha256"]),
        main_figure_bounds_and_layout_unchanged=True, manuscript_copies_identical=True,
        manuscript_pages=21, figure1_page=2, complete_table_page=16, embedded_font_objects=len(font_records),
        manuscript_sha256=sha(DRAFT / "main.pdf"), figure_sha256=figure["output_sha256"],
        full_data_sha256=sha(HERE / "data/full-trained-results.json"),
        scripts_sha256={p.name: sha(p) for p in HERE.glob("*.py")})
    (HERE / "data/verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
