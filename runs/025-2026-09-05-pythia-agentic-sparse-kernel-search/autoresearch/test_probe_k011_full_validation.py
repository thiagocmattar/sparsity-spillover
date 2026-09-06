from pathlib import Path
import importlib.util

import pytest


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "probe_k011_full_validation.py"
SPEC = importlib.util.spec_from_file_location("run025_probe_k011_full_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_development_boundary_accepts_only_predeclared_14m_rows(monkeypatch):
    monkeypatch.setattr(
        MODULE,
        "read_json",
        lambda _path: {
            "partition_boundary": {"development_conditions": ["14m/a0"]}
        },
    )
    manifest = {
        "checkpoints": [
            {"id": "14m/a0", "size": "14m", "partition": "development"},
            {"id": "14m/heldout", "size": "14m", "partition": "untuned"},
            {"id": "70m/a0", "size": "70m", "partition": "development"},
        ]
    }
    assert MODULE.development_condition(manifest, "14m/a0")["id"] == "14m/a0"
    with pytest.raises(ValueError):
        MODULE.development_condition(manifest, "14m/heldout")
    with pytest.raises(ValueError):
        MODULE.development_condition(manifest, "70m/a0")


def test_site_selection_preserves_topology_contract():
    topology = {"active_sites": ["a", "q_post", "z"]}
    assert MODULE.select_sites("active", topology) == frozenset(("a", "z"))
    assert MODULE.select_sites("h", topology) == frozenset(("h",))
    assert MODULE.select_sites("all", topology) == MODULE.LINEAR_SITES


def test_source_requires_complete_validation_and_records_heldout_boundary():
    text = SOURCE.read_text(encoding="utf-8")
    assert "if not args.full_validation" in text
    assert "blocks != 338 or tail != 1444" in text
    assert "partition\") != \"development\"" in text
    assert "held-out checkpoint conditions forbidden" in text
