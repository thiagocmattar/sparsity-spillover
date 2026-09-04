from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("paper_synthesis", HERE / "01_build.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_complete_source_grids_and_coverage() -> None:
    data = MODULE.build_data()
    assert data["status"] == "complete_verified"
    assert data["coverage"]["sequences"] == 338
    assert data["coverage"]["input_tokens"] == 692_224
    assert len(data["trained_endpoints"]) == 30
    assert len(data["teal_points"]) == 60
    assert len(data["endpoint_microstructure"]) == 12


def test_baseline_exposure_and_lr_selection() -> None:
    data = MODULE.build_data()
    exposure = {row["scale"]: row for row in data["baseline_exposure"]}
    assert exposure["14M"]["tokens_per_parameter"] > exposure["70M"]["tokens_per_parameter"] > exposure["410M"]["tokens_per_parameter"]
    assert exposure["70M"]["validation_loss"] < exposure["410M"]["validation_loss"]
    assert data["run021_selection"]["selected_peak_learning_rate"] == 3e-4


def test_410m_trained_endpoint_reversal() -> None:
    data = MODULE.build_data()
    deltas = data["within_family_deltas"]
    for family in ("A4-OL1", "A7-OL1"):
        row = MODULE.select(deltas, kind="trained", scale="410M", family=family, dose=0.5)
        assert row["delta_R_model_pp"] > 30
        assert row["delta_loss"] < 0
    for scale in ("14M", "70M"):
        for family in ("A4-OL1", "A7-OL1"):
            row = MODULE.select(deltas, kind="trained", scale=scale, family=family, dose=0.5)
            assert row["delta_R_model_pp"] > 0
            assert row["delta_loss"] > 0


def test_operation_contributions_reconcile() -> None:
    data = MODULE.build_data()
    for row in data["endpoint_microstructure"]:
        assert abs(sum(row["operation_contributions"].values()) - row["R_model"]) < 1e-10
        assert all(0 <= value <= 1 for value in row["site_exact_zero"].values())


def test_generated_contract(tmp_path: Path) -> None:
    data = MODULE.build_data()
    MODULE.render_absolute(data, tmp_path / "absolute.pdf")
    MODULE.render_deltas(data, tmp_path / "deltas.pdf")
    MODULE.render_site_heatmap(data, tmp_path / "sites.pdf")
    MODULE.render_operation_contributions(data, tmp_path / "operations.pdf")
    for name in ("absolute.pdf", "deltas.pdf", "sites.pdf", "operations.pdf"):
        path = tmp_path / name
        assert path.read_bytes().startswith(b"%PDF")
        assert path.stat().st_size > 10_000
