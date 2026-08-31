"""Focused tests for Analysis 009's immutable-source comparison."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("analysis_009_build", ANALYSIS_DIR / "01_build.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reduction_has_matched_complete_grids_and_realized_pressure_sites() -> None:
    data = MODULE.build_figure_data()
    assert data["status"] == "complete_verified_analysis"
    assert data["coverage"] == {
        "documents": 500,
        "sequences": 338,
        "input_tokens": 692_224,
        "excluded_tail_tokens": 1_444,
        "seed_count": 1,
    }
    h_only = [row for row in data["series"] if row["series_id"] == "run012_h_only"]
    baseline = [row for row in data["series"] if row["series_id"] == "run011_a4"]
    four_site = [
        row for row in data["series"] if row["series_id"] == "run015_four_site"
    ]
    expected = [0.0, 0.01, 0.05, 0.1, 0.5]
    assert [row["kappa"] for row in baseline] == expected
    assert [row["kappa"] for row in h_only] == expected
    assert [row["kappa"] for row in four_site] == expected
    assert all(row["realized_pressure_sites"] == [] for row in baseline)
    assert all(row["realized_pressure_sites"] == ["h"] for row in h_only)
    assert all(
        row["realized_pressure_sites"] == ["a", "m", "h", "z"]
        for row in four_site
    )
    assert len(data["matched_comparison"]) == 5
    assert len(data["series"]) == 15


def test_matched_manifest_contracts_and_count_first_fractions_reconcile() -> None:
    data = MODULE.build_figure_data()
    by_series = {
        series_id: sorted(
            (row for row in data["series"] if row["series_id"] == series_id),
            key=lambda row: row["kappa"],
        )
        for series_id in ("run011_a4", "run012_h_only", "run015_four_site")
    }
    for historical, corrected in zip(
        by_series["run012_h_only"], by_series["run015_four_site"], strict=True
    ):
        assert historical["matched_contract_sha256"] == corrected["matched_contract_sha256"]
    for baseline, historical, corrected in zip(
        by_series["run011_a4"],
        by_series["run012_h_only"],
        by_series["run015_four_site"],
        strict=True,
    ):
        assert (
            baseline["matched_gate_contract_sha256"]
            == historical["matched_gate_contract_sha256"]
            == corrected["matched_gate_contract_sha256"]
        )
    for row in data["series"]:
        logical = row["logical_product_counts"]
        assert math.isclose(
            logical["zero_product_count"] / logical["model_product_count"],
            row["R_model"],
            rel_tol=0.0,
            abs_tol=1e-16,
        )
        assert (
            abs(logical["diagnostic_minus_terminal_validation_loss"])
            <= MODULE.MAX_DIAGNOSTIC_LOSS_DELTA
        )
        for site in MODULE.ALL_SITES:
            counts = row["site_exact_zero"][site]
            assert math.isclose(
                counts["exact_zero_count"] / counts["total_count"],
                counts["exact_zero_fraction"],
                rel_tol=0.0,
                abs_tol=1e-15,
            )
    assert by_series["run012_h_only"][0]["site_exact_zero"]["a"] == {
        "exact_zero_count": 262_695_415,
        "total_count": 531_628_032,
        "exact_zero_fraction": 0.4941338665151502,
    }
    assert by_series["run015_four_site"][0]["site_exact_zero"]["a"] == {
        "exact_zero_count": 366_241_969,
        "total_count": 531_628_032,
        "exact_zero_fraction": 0.688906428846852,
    }
    assert all(row["site_exact_zero"]["h"]["total_count"] == 2_126_512_128 for row in data["series"])


def test_realization_audit_guards_the_h_only_bug_and_four_site_fix() -> None:
    data = MODULE.build_figure_data()
    audit = data["realization_audit"]
    assert audit["run011_a4"] == {
        "declared_pressure_sites": [],
        "realized_pressure_sites": [],
        "pressure_method": "none",
    }
    assert audit["run012_h_only"]["realized_pressure_sites"] == ["h"]
    assert 'ActivationCapture(model, ["h"]' in audit["run012_h_only"]["capture_expression"]
    assert audit["run015_four_site"]["realized_pressure_sites"] == ["a", "m", "h", "z"]
    assert audit["run015_four_site"]["pressure_capture_tensor_count_per_microbatch"] == 24
    assert audit["run015_four_site"]["pressure_capture_names_sha256"] == MODULE.EXPECTED_CAPTURE_HASH


def test_observation_contains_the_exact_generated_zero_mass_table() -> None:
    observation = (
        ANALYSIS_DIR / "observations" / "O001-h-only-vs-four-site-a4-ol1.md"
    ).read_text(encoding="utf-8")
    assert MODULE.table_markdown(MODULE.build_figure_data()).strip() in observation


def test_committed_outputs_match_sources_and_pdf_contract() -> None:
    expected = MODULE.build_figure_data()
    committed = json.loads((ANALYSIS_DIR / "figure_data.json").read_text(encoding="utf-8"))
    assert committed == expected
    pdf = ANALYSIS_DIR / "figures" / "01-rmodel-vs-validation-loss.pdf"
    assert pdf.read_bytes().startswith(b"%PDF-")
    assert pdf.stat().st_size > 15_000
    source = (ANALYSIS_DIR / "01_build.py").read_text(encoding="utf-8")
    assert "inset_axes" not in source
    assert MODULE.SERIES["run011_a4"]["label"] == "A4 without OL1 (Run 011)"
