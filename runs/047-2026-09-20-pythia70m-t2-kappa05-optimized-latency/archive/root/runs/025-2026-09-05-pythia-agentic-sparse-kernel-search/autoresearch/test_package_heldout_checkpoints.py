import importlib.util
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
SPEC = importlib.util.spec_from_file_location(
    "run025_package_heldout", HERE / "package_heldout_checkpoints.py"
)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def row(identifier, partition="untuned"):
    return {
        "id": identifier,
        "size": identifier.split("/", 1)[0],
        "partition": partition,
        "files": [],
        "provenance": [],
    }


def complete_rows():
    return [row(f"{size}/{suffix}") for size in P.SIZES for suffix in P.SUFFIXES]


def test_selects_exactly_eighteen_untouched_interior_checkpoints():
    rows = complete_rows()
    manifest = {"checkpoints": rows + [row("410m/a0", "development")]}
    assert P.heldout_rows(manifest) == sorted(rows, key=lambda item: item["id"])


def test_rejects_missing_or_extra_untuned_checkpoint():
    rows = complete_rows()[:-1]
    with pytest.raises(ValueError, match="frozen design"):
        P.heldout_rows({"checkpoints": rows})
    rows = complete_rows() + [row("410m/a0")]
    with pytest.raises(ValueError, match="frozen design"):
        P.heldout_rows({"checkpoints": rows})


def test_deduplicates_records_and_rejects_conflicts():
    item = {"path": "x", "bytes": 1, "sha256": "a"}
    rows = [{"files": [item], "provenance": [dict(item)]}]
    assert P.unique_records(rows) == [item]
    rows[0]["provenance"][0]["bytes"] = 2
    with pytest.raises(ValueError, match="Conflicting"):
        P.unique_records(rows)
