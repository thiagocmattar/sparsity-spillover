"""Focused tests for Analysis 008's A7/A7-OL1 immutable-source reduction."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("analysis_008_build", ANALYSIS_DIR / "01_build.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reduction_has_complete_coverage_and_expected_a7_grids() -> None:
    data = MODULE.build_figure_data()
    assert data["status"] == "complete_verified_figure_data"
    assert data["coverage"] == {
        "documents": 500,
        "sequences": 338,
        "input_tokens": 692_224,
        "excluded_tail_tokens": 1_444,
        "seed_count": 1,
    }
    a7_rows = [
        row
        for row in data["trained_endpoints"]
        if row["series_id"] == "a7z_post_mixed_threshold"
    ]
    a7_ol1_rows = [
        row
        for row in data["trained_endpoints"]
        if row["series_id"] == "a7z_post_mixed_threshold_ol1"
    ]
    expected_kappas = [0.0, 0.01, 0.05, 0.1, 0.5]
    assert [row["dose"] for row in a7_rows] == expected_kappas
    assert [row["dose"] for row in a7_ol1_rows] == expected_kappas
    assert len(data["trained_endpoints"]) == 30
    assert len(data["a7_matched_a4_comparison"]) == 5
    assert len(data["a7_ol1_matched_a7_comparison"]) == 5


def test_site_and_logical_fractions_are_derived_from_integer_counts() -> None:
    data = MODULE.build_figure_data()
    rows = [
        row
        for row in data["trained_endpoints"]
        if row["series_id"]
        in {"a7z_post_mixed_threshold", "a7z_post_mixed_threshold_ol1"}
    ]
    for row in rows:
        logical = row["logical_product_counts"]
        assert math.isclose(
            logical["zero_product_count"] / logical["model_product_count"],
            row["R_model"],
            rel_tol=0.0,
            abs_tol=1e-16,
        )
        for site in MODULE.ALL_SITES:
            counts = row["site_exact_zero"][site]
            assert math.isclose(
                counts["exact_zero_count"] / counts["total_count"],
                counts["exact_zero_fraction"],
                rel_tol=0.0,
                abs_tol=1e-15,
            )
    assert rows[0]["site_exact_zero"]["a"] == {
        "exact_zero_count": 258_316_282,
        "total_count": 531_628_032,
        "exact_zero_fraction": 0.48589665414783845,
    }
    assert rows[0]["site_exact_zero"]["attention_output"]["exact_zero_count"] == 129
    assert rows[0]["site_exact_zero"]["h"]["total_count"] == 2_126_512_128
    assert rows[5]["site_exact_zero"]["a"] == {
        "exact_zero_count": 258_278_995,
        "total_count": 531_628_032,
        "exact_zero_fraction": 0.4858265167627579,
    }
    assert rows[5]["site_exact_zero"]["attention_output"]["exact_zero_count"] == 126
    assert rows[5]["pressure_sites"] == list(MODULE.ACTIVE_SITES)
    assert rows[5]["step_budget"] == 1.0


def test_table_is_single_and_contains_every_requested_column() -> None:
    text = MODULE.table_markdown(MODULE.build_figure_data())
    assert text.count("| kappa |") == 1
    for name in (
        "Variant",
        "Validation loss",
        "R_model (%)",
        "a zero (%)",
        "m zero (%)",
        "h zero (%)",
        "q_post zero (%)",
        "k_post zero (%)",
        "v zero (%)",
        "z zero (%)",
        "attention_output zero (%)",
    ):
        assert name in text
    data_rows = [line for line in text.splitlines() if line.startswith("| 0")]
    assert len(data_rows) == 10
    assert "| 0 | A7 |" in data_rows[0]
    assert "| 0 | A7-OL1 |" in data_rows[1]


def test_committed_outputs_match_the_source_reduction() -> None:
    expected = MODULE.build_figure_data()
    committed = json.loads((ANALYSIS_DIR / "figure_data.json").read_text(encoding="utf-8"))
    assert committed == expected
    assert (ANALYSIS_DIR / "tables.md").read_text(encoding="utf-8") == MODULE.table_markdown(expected)
    pdf = ANALYSIS_DIR / "figures" / "01-full-pass-frontier-with-a7.pdf"
    assert pdf.read_bytes().startswith(b"%PDF-")
    assert pdf.stat().st_size > 20_000


def test_figure_uses_one_panel_and_plain_ol1_labels() -> None:
    source = (ANALYSIS_DIR / "01_build.py").read_text(encoding="utf-8")
    assert "inset_axes" not in source
    assert MODULE.TRAINED_STYLES["a4z_ol1"]["label"] == "A4-OL1"
    assert MODULE.TRAINED_STYLES["a7z_post_mixed_threshold_ol1"]["label"] == "A7-OL1"
