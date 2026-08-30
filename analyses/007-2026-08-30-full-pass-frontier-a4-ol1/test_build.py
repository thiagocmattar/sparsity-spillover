"""Focused tests for Analysis 007's immutable-source reduction."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("analysis_007_build", ANALYSIS_DIR / "01_build.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reduction_has_complete_coverage_and_expected_a4_ol1_grid() -> None:
    data = MODULE.build_figure_data()
    assert data["status"] == "complete_verified_figure_data"
    assert data["coverage"] == {
        "documents": 500,
        "sequences": 338,
        "input_tokens": 692_224,
        "excluded_tail_tokens": 1_444,
        "seed_count": 1,
    }
    rows = [row for row in data["trained_endpoints"] if row["series_id"] == "a4z_ol1"]
    assert [row["dose"] for row in rows] == [0.0, 0.01, 0.05, 0.1, 0.5]
    assert len(data["trained_endpoints"]) == 20


def test_site_and_logical_fractions_are_derived_from_integer_counts() -> None:
    data = MODULE.build_figure_data()
    rows = [row for row in data["trained_endpoints"] if row["series_id"] == "a4z_ol1"]
    for row in rows:
        logical = row["logical_product_counts"]
        assert math.isclose(
            logical["zero_product_count"] / logical["model_product_count"],
            row["R_model"],
            rel_tol=0.0,
            abs_tol=1e-16,
        )
        for site in MODULE.SELECTED_SITES:
            counts = row["site_exact_zero"][site]
            assert math.isclose(
                counts["exact_zero_count"] / counts["total_count"],
                counts["exact_zero_fraction"],
                rel_tol=0.0,
                abs_tol=1e-15,
            )
    assert rows[0]["site_exact_zero"]["a"] == {
        "exact_zero_count": 262_695_415,
        "total_count": 531_628_032,
        "exact_zero_fraction": 0.4941338665151502,
    }
    assert rows[0]["site_exact_zero"]["h"]["total_count"] == 2_126_512_128


def test_committed_outputs_match_the_source_reduction() -> None:
    expected = MODULE.build_figure_data()
    committed = json.loads((ANALYSIS_DIR / "figure_data.json").read_text(encoding="utf-8"))
    assert committed == expected
    assert (ANALYSIS_DIR / "tables.md").read_text(encoding="utf-8") == MODULE.table_markdown(expected)
    pdf = ANALYSIS_DIR / "figures" / "01-full-pass-frontier-with-a4-ol1.pdf"
    assert pdf.read_bytes().startswith(b"%PDF-")
    assert pdf.stat().st_size > 20_000
