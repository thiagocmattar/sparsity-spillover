import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN = HERE.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_k009_and_k012_freezes_match_sources_and_partition_boundaries():
    manifest = json.loads((RUN / "prelaunch/input_manifest.json").read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in manifest["checkpoints"]}
    for candidate, size in (("k009", "70m"), ("k012", "14m")):
        freeze = json.loads(
            (HERE / f"candidates/{candidate}/FROZEN.json").read_text(encoding="utf-8")
        )
        assert freeze["candidate_id"] == candidate
        development = [row["condition"] for row in freeze["selection_evidence"]["development_evaluation"]]
        heldout = freeze["heldout_boundary"]["conditions"]
        assert len(development) == len(set(development)) == 6
        assert len(heldout) == len(set(heldout)) == 6
        assert all(by_id[item]["size"] == size and by_id[item]["partition"] == "development" for item in development)
        assert all(by_id[item]["size"] == size and by_id[item]["partition"] == "untuned" for item in heldout)
        assert not set(development).intersection(heldout)
        for row in freeze["sources"]:
            path = RUN / row["path"]
            assert path.is_file() and not path.is_symlink()
            assert path.stat().st_size == row["bytes"]
            assert digest(path) == row["sha256"]
