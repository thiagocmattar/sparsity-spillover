import importlib.util
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
SPEC = importlib.util.spec_from_file_location(
    "run025_package_size_development", HERE / "package_size_development.py"
)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def row(identifier, partition="development"):
    size = identifier.split("/", 1)[0]
    return {
        "id": identifier,
        "size": size,
        "partition": partition,
        "files": [],
        "provenance": [],
    }


def test_selects_only_frozen_size_development_endpoints():
    wanted = [
        row("70m/a0"),
        row("70m/a1h"),
        row("70m/a4-0"),
        row("70m/a4-0p5"),
        row("70m/a7-0"),
        row("70m/a7-0p5"),
    ]
    manifest = {
        "checkpoints": wanted
        + [row("70m/a4-0p1", "untuned"), row("410m/a0")]
    }
    assert P.development_rows(manifest, "70m") == sorted(
        wanted, key=lambda item: item["id"]
    )


def test_rejects_missing_or_extra_development_endpoint():
    manifest = {
        "checkpoints": [
            row("70m/a0"),
            row("70m/a1h"),
            row("70m/a4-0"),
            row("70m/a4-0p5"),
            row("70m/a7-0"),
        ]
    }
    with pytest.raises(ValueError, match="frozen design"):
        P.development_rows(manifest, "70m")


def test_deduplicates_identical_records_and_rejects_conflicts():
    first = {"path": "x", "bytes": 1, "sha256": "a"}
    rows = [{"files": [first], "provenance": [dict(first)]}]
    assert P.unique_records(rows) == [first]
    rows[0]["provenance"][0]["bytes"] = 2
    with pytest.raises(ValueError, match="Conflicting"):
        P.unique_records(rows)
