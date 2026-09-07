import importlib.util
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "run025_probe_k010_heldout", HERE / "probe_k010_heldout.py"
)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def test_condition_accepts_only_untouched_interior_partition():
    wanted = {"id": "410m/a4-0p05", "partition": "untuned"}
    assert P.heldout_condition({"checkpoints": [wanted]}, wanted["id"]) is wanted
    with pytest.raises(ValueError, match="untouched"):
        P.heldout_condition(
            {"checkpoints": [{"id": "410m/a4-0p5", "partition": "development"}]},
            "410m/a4-0p5",
        )


def test_freeze_verifies_current_sources_and_declared_boundary():
    freeze = P.verify_freeze("410m/a7-0p1")
    assert freeze["policy"]["runtime_density_dispatch"] is False
    with pytest.raises(ValueError, match="outside"):
        P.verify_freeze("410m/a7-0p5")
