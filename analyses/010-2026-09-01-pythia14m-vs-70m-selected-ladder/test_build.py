"""Focused source-reduction and output tests for Analysis 010."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("analysis_010_build", ANALYSIS_DIR / "01_build.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reduction_has_complete_grids_and_coverage() -> None:
    data = MODULE.build_figure_data()
    assert data["status"] == "complete_verified_analysis"
    assert len(data["trained_endpoints"]) == 20
    assert len(data["teal_points"]) == 40
    assert data["coverage"] == {
        "documents": 500,
        "sequences": 338,
        "input_tokens": 692_224,
        "excluded_tail_tokens": 1_444,
        "complete_block_coverage": True,
        "seed_count": 1,
    }
    for scale in ("14M", "70M"):
        for family in ("A4-OL1", "A7-OL1"):
            rows = [row for row in data["trained_endpoints"] if row["scale"] == scale and row["family"] == family]
            assert [row["kappa"] for row in rows] == [0.0, 0.01, 0.05, 0.1, 0.5]
        for control in ("A0", "A1-H"):
            rows = [row for row in data["teal_points"] if row["scale"] == scale and row["control"] == control]
            assert [row["target_sparsity"] for row in rows] == [index / 10 for index in range(10)]


def test_integer_counts_and_boundary_metrics_reconcile() -> None:
    data = MODULE.build_figure_data()
    for row in data["trained_endpoints"]:
        logical = row["logical_counts"]
        assert math.isclose(
            logical["zero_product_count"] / logical["model_product_count"],
            row["R_model"],
            rel_tol=0.0,
            abs_tol=1e-16,
        )
        assert 0 <= row["conflict_steps"] <= 712
        assert 0 <= row["projection_steps"] <= 712
        for counts in row["site_exact_zero"].values():
            assert math.isclose(
                counts["exact_zero_count"] / counts["total_count"],
                counts["exact_zero_fraction"],
                rel_tol=0.0,
                abs_tol=1e-15,
            )
    for row in data["teal_points"]:
        logical = row["logical_counts"]
        assert math.isclose(
            logical["zero_product_count"] / logical["model_product_count"],
            row["R_model"],
            rel_tol=0.0,
            abs_tol=1e-16,
        )


def test_persistence_checks_capture_the_matched_crossover() -> None:
    data = MODULE.build_figure_data()
    for scale in ("14M", "70M"):
        checks = data["persistence_checks"][scale]
        assert checks["A4_dominates_A7_at_kappa_0"] is True
        assert checks["A7_dominates_A4_at_kappa_0p5"] is True
        assert checks["A7_kappa_0p1_delta_R_model_percentage_points_from_kappa_0"] > 4.0
    assert data["persistence_checks"]["14M"]["A7_kappa_0p1_delta_validation_loss_from_kappa_0"] < 0
    assert data["persistence_checks"]["70M"]["A7_kappa_0p1_delta_validation_loss_from_kappa_0"] < 0.01


def test_committed_outputs_match_source_reduction() -> None:
    expected = MODULE.build_figure_data()
    committed = json.loads((ANALYSIS_DIR / "figure_data.json").read_text(encoding="utf-8"))
    assert committed == expected
    assert (ANALYSIS_DIR / "tables.md").read_text(encoding="utf-8") == MODULE.table_markdown(expected)
    pdf = ANALYSIS_DIR / "figures" / "01-pythia14m-vs-70m-selected-ladder.pdf"
    assert pdf.read_bytes().startswith(b"%PDF-")
    assert pdf.stat().st_size > 15_000


def test_figure_contract_is_pdf_only_single_absolute_frontier() -> None:
    source = (ANALYSIS_DIR / "01_build.py").read_text(encoding="utf-8")
    assert "plt.subplots(1, 2" not in source
    assert 'axis.set_ylim(4.05, y_max)' in source
    assert "y_max = 6.0" in source
    assert "Final control checkpoints" not in source
    assert "Delta R_model from target 0" not in source
    for color in ("#CC79A7", "#222222", "#6F4C9B", "#56B4E9"):
        assert color in source
    assert ".png" not in source.lower()
