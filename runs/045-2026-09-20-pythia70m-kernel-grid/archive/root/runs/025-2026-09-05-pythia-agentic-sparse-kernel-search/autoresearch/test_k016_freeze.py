from pathlib import Path
import hashlib
import json


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
FREEZE = HERE / "candidates/k016/FROZEN.json"


def test_k016_freeze_sources_and_boundary_are_exact():
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert frozen["candidate_id"] == "k016"
    assert frozen["policy"]["kappa_dispatch"] is False
    assert frozen["heldout_boundary"]["fresh_holdout_available"] is False
    assert len(frozen["heldout_boundary"]["conditions"]) == 6
    for row in frozen["sources"]:
        path = RUN / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


def test_k016_freeze_uses_one_policy_per_topology():
    policy = json.loads(FREEZE.read_text(encoding="utf-8"))["policy"]
    assert set(policy["topology_policies"]) == {"A0", "A1-H", "A4-Z", "A7-Z-POST"}
    assert policy["topology_policies"]["A7-Z-POST"]["sites"] == ["h", "z"]
