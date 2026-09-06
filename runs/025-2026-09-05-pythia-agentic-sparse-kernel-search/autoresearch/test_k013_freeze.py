from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
FREEZE_PATH = HERE / "candidates/k013/FROZEN.json"


def test_k013_freeze_source_records_are_exact():
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    assert freeze["candidate_id"] == "k013"
    for item in freeze["sources"]:
        path = RUN / item["path"]
        assert path.is_file()
        assert path.stat().st_size == item["bytes"]
        assert sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_k013_freeze_preserves_heldout_boundary_and_static_policy():
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    assert freeze["policy"]["layer_indices"] == [3, 4, 5]
    assert freeze["policy"]["runtime_density_dispatch"] is False
    assert freeze["policy"]["checkpoint_identity_dispatch"] is False
    assert freeze["heldout_boundary"]["heldout_result_artifacts_present_remote_at_freeze"] == 0
    assert freeze["heldout_boundary"]["selection_artifact_heldout_inspected"] is False
    assert len(freeze["heldout_boundary"]["conditions"]) == 6
    assert len(freeze["selection_evidence"]["development_evaluation"]) == 6
