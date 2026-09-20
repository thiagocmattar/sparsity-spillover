from pathlib import Path
import importlib.util


SOURCE = Path(__file__).resolve().parent / "candidates/k016/candidate.py"
SPEC = importlib.util.spec_from_file_location("run025_k016_candidate_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_policy_is_topology_dispatched_and_kappa_independent():
    assert set(MODULE.POLICIES) == {"A0", "A1-H", "A4-Z", "A7-Z-POST"}
    assert MODULE.POLICIES["A0"] == {"layers": (), "sites": ()}
    assert MODULE.POLICIES["A4-Z"]["layers"] == (3, 4, 5)
    assert MODULE.POLICIES["A7-Z-POST"]["sites"] == ("h", "z")
    assert "kappa" not in repr(MODULE.POLICIES).lower()
