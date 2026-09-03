"""Focused verification for Analysis 011."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("analysis011_build", HERE / "01_build.py")
assert SPEC and SPEC.loader
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def test_complete_grids_and_paired_coverage() -> None:
    data = BUILD.build_figure_data()
    assert data["status"] == "complete_verified_analysis"
    assert len(data["trained_endpoints"]) == 30
    assert len(data["teal_points"]) == 60
    assert data["coverage"] == {
        "documents": 500,
        "sequences": 338,
        "input_tokens": 692_224,
        "excluded_tail_tokens": 1_444,
        "complete_block_coverage": True,
        "seed_count_per_scale": 1,
    }
    for scale in BUILD.SCALES:
        for family in BUILD.FAMILIES:
            rows = [
                row
                for row in data["trained_endpoints"]
                if row["scale"] == scale and row["family"] == family
            ]
            assert tuple(row["kappa"] for row in rows) == BUILD.KAPPAS
        for control in BUILD.CONTROLS:
            rows = [
                row
                for row in data["teal_points"]
                if row["scale"] == scale and row["control"] == control
            ]
            assert tuple(row["target_sparsity"] for row in rows) == BUILD.TARGETS


def test_run019_endpoints_and_control_identity() -> None:
    data = BUILD.build_figure_data()
    rows = {
        (row["family"], row["kappa"]): row
        for row in data["trained_endpoints"]
        if row["scale"] == "410M"
    }
    assert abs(rows[("A4-OL1", 0.5)]["validation_loss"] - 5.1909658185829075) < 1e-12
    assert abs(rows[("A4-OL1", 0.5)]["R_model"] - 0.715914446480748) < 1e-15
    assert abs(rows[("A7-OL1", 0.5)]["validation_loss"] - 5.120691668705122) < 1e-12
    assert abs(rows[("A7-OL1", 0.5)]["R_model"] - 0.806155139186339) < 1e-15
    controls = {
        row["control"]: row
        for row in data["teal_points"]
        if row["scale"] == "410M" and row["target_sparsity"] == 0.0
    }
    assert abs(controls["A0"]["validation_loss"] - 4.547456437666741) < 1e-12
    assert abs(controls["A1-H"]["R_model"] - 0.21049725280441106) < 1e-15
    assert set(controls["A0"]["site_exact_zero"]) == set(BUILD.SITES)


def test_count_reconciliation_and_expected_persistence_change() -> None:
    data = BUILD.build_figure_data()
    for row in data["trained_endpoints"] + data["teal_points"]:
        counts = row["logical_counts"]
        assert abs(
            counts["zero_product_count"] / counts["model_product_count"]
            - row["R_model"]
        ) < 1e-16
        for site_row in row["site_exact_zero"].values():
            assert abs(
                site_row["exact_zero_count"] / site_row["total_count"]
                - site_row["exact_zero_fraction"]
            ) < 1e-15
    checks = data["persistence_checks"]
    assert checks["14M"]["A4_dominates_A7_at_kappa_0"] is True
    assert checks["70M"]["A4_dominates_A7_at_kappa_0"] is True
    assert checks["410M"]["A7_dominates_A4_at_kappa_0"] is True
    assert all(checks[scale]["A7_dominates_A4_at_kappa_0p5"] for scale in BUILD.SCALES)


def test_markdown_contains_complete_410m_and_teal_tables() -> None:
    data = BUILD.build_figure_data()
    text = BUILD.table_markdown(data)
    assert "## Pythia-410M selected endpoints" in text
    assert "| A0 | p=0 | 4.547456 | 0.0110 |" in text
    assert "| A7-OL1 | kappa=0.5 | 5.120692 | 80.6155 |" in text
    assert text.count("## A0 post-hoc TEAL frontier") == 1
    assert text.count("## A1-H post-hoc TEAL frontier") == 1
    assert "| 0.9 | 410M |" in text
