from pathlib import Path
import importlib.util

import pytest


SOURCE = Path(__file__).resolve().parent / "candidates/k014/candidate.py"
SPEC = importlib.util.spec_from_file_location("run025_k014_candidate_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_masks_are_static_and_cover_the_predeclared_ladder():
    assert list(MODULE.MASKS) == ["suffix2", "suffix3", "suffix4", "suffix5", "all6", "odd"]
    assert MODULE.selected_layers("suffix2") == frozenset((4, 5))
    assert MODULE.selected_layers("all6") == frozenset(range(6))
    assert MODULE.implementation_for("odd") == "k014-odd"
    with pytest.raises(ValueError):
        MODULE.selected_layers("prefix1")
