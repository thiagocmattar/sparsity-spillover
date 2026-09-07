from pathlib import Path
import importlib.util

import pytest


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CANDIDATE = load("run025_k015_candidate_test", HERE / "candidates/k015/candidate.py")
PROBE = load("run025_k015_probe_test", HERE / "probe_k015_models.py")
SELECT = load("run025_k015_select_test", HERE / "select_k015_complete_validation.py")


def test_variant_contract_and_identity():
    layers, sites = CANDIDATE.variant("all-hz")
    assert layers == frozenset(range(6))
    assert sites == frozenset(("h", "z"))
    assert CANDIDATE.implementation_for("all-hz") == "k015-all-hz"
    with pytest.raises(ValueError):
        CANDIDATE.variant("all-active")


def test_probe_boundary_is_a7_development_only():
    manifest = {"checkpoints": [
        {"id": "70m/a7-0", "size": "70m", "partition": "development"},
        {"id": "70m/a7-0p01", "size": "70m", "partition": "untuned"},
    ]}
    assert PROBE.development_condition(manifest, "70m/a7-0")["id"] == "70m/a7-0"
    with pytest.raises(ValueError):
        PROBE.development_condition(manifest, "70m/a7-0p01")


def test_selector_prefers_fewer_modules_inside_tolerance():
    rows = [
        {"variant": "a", "score": 1.100, "adapted_linear_count": 12, "eligible": True},
        {"variant": "b", "score": 1.096, "adapted_linear_count": 6, "eligible": True},
    ]
    assert SELECT.choose(rows)["variant"] == "b"
