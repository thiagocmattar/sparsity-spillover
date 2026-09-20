"""Small append-only trial record; no model API, Pod credentials, or scheduler."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from run025_common import record, verify_record


def exclusive_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def begin(directory, *, proposal, source_paths, evaluator_paths, root):
    """The caller explicitly supplies identity/settings/accounting; unknowns stay null."""
    required = {"candidate_id", "parent_id", "trajectory_id", "hypothesis", "agent", "budget"}
    if not required.issubset(proposal):
        raise ValueError(f"Missing trial fields: {sorted(required - proposal.keys())}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    sources = [record(path, root) for path in source_paths]
    snapshots = []
    for path, row in zip(source_paths, sources):
        destination = directory / "sources" / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        snapshot = record(destination, root)
        if snapshot["sha256"] != row["sha256"]:
            raise ValueError("Source changed while snapshotting")
        snapshots.append(snapshot)
    value = {"proposal": proposal, "sources": sources, "snapshots": snapshots,
             "evaluator": [record(path, root) for path in evaluator_paths]}
    exclusive_json(directory / "proposal.json", value)
    return value


def finish(directory, *, result, root):
    directory = Path(directory)
    proposal = json.loads((directory / "proposal.json").read_text(encoding="utf-8"))
    for row in proposal["sources"] + proposal["snapshots"] + proposal["evaluator"]:
        verify_record(row, root)
    required = {"status", "decision", "elapsed_seconds", "gpu_usd", "agent_usd", "token_usage", "evidence"}
    if not required.issubset(result):
        raise ValueError(f"Missing result fields: {sorted(required - result.keys())}")
    exclusive_json(directory / "result.json", result)
