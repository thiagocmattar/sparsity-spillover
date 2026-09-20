from pathlib import Path
import importlib.util

import pytest


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "probe_k014_models.py"
SPEC = importlib.util.spec_from_file_location("run025_probe_k014_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_implementation_mapping_is_predeclared():
    assert MODULE.parse_implementation("k014-suffix3") == "suffix3"
    with pytest.raises(ValueError):
        MODULE.parse_implementation("k014-prefix1")


def test_development_boundary_rejects_other_sizes_and_untuned():
    manifest = {
        "checkpoints": [
            {"id": "70m/a1h", "size": "70m", "partition": "development"},
            {"id": "70m/a4-0p01", "size": "70m", "partition": "untuned"},
            {"id": "14m/a1h", "size": "14m", "partition": "development"},
        ]
    }
    assert MODULE.development_condition(manifest, "70m/a1h")["id"] == "70m/a1h"
    with pytest.raises(ValueError):
        MODULE.development_condition(manifest, "70m/a4-0p01")
    with pytest.raises(ValueError):
        MODULE.development_condition(manifest, "14m/a1h")


def test_probe_inherits_complete_validation_contract():
    base_text = (HERE / "probe_models.py").read_text(encoding="utf-8")
    assert "blocks != 338 or tail != 1444" in base_text
    assert "Fixed complete-validation logit/loss gate failed" in base_text
