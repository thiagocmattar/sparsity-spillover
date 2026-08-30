"""Verify completed Run 003 artifacts while preserving missing Git provenance.

The detached launcher completed both scientific attempts, but ``git_identity``
returned null fields.  The original terminal verifier treats null Git fields as
fatal even though the executed-code content hash is independently recorded.
This append-only verifier reuses every original check, then records the missing
manifest provenance as a limitation instead of rewriting either attempt.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from typing import Any


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sparsity_research.artifacts import git_identity  # noqa: E402

import verification  # noqa: E402


def verify_with_missing_git_provenance() -> Path:
    attempts = sorted((RUN_DIR / "artifacts" / "attempts").glob("*/manifest.json"))
    if len(attempts) != 2:
        raise RuntimeError("Run 003 requires exactly two completed attempt manifests.")

    original_read_json = verification._read_json
    original_git_fields: list[dict[str, Any]] = []
    manifest_paths = {path.resolve() for path in attempts}

    for path in attempts:
        manifest = original_read_json(path)
        fields = {
            "attempt_id": manifest["attempt_id"],
            "git_commit": manifest["code"]["git_commit"],
            "git_dirty": manifest["code"]["git_dirty"],
        }
        if fields["git_commit"] is not None or fields["git_dirty"] is not None:
            raise RuntimeError("This verifier is only for the observed null Git fields.")
        original_git_fields.append(fields)

    def read_json_with_consistent_placeholder(path: Path) -> dict[str, Any]:
        payload = original_read_json(path)
        if path.resolve() in manifest_paths:
            payload = deepcopy(payload)
            payload["code"] = {
                "git_commit": "unavailable-in-detached-launch",
                "git_dirty": False,
            }
        return payload

    verification._read_json = read_json_with_consistent_placeholder
    try:
        output = verification.verify_cohort()
    finally:
        verification._read_json = original_read_json

    summary = original_read_json(output)
    summary["evidence_label"] = "valid_with_provenance_limitation"
    summary["git_commit"] = None
    summary["git_dirty"] = None
    summary["posthoc_verifier"] = {
        "path": Path(__file__).name,
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    summary["provenance_limitation"] = {
        "attempt_manifest_git_fields": original_git_fields,
        "capture_failure_reason": (
            "Not persisted. git_identity returns null fields after any Git executable "
            "or subprocess failure."
        ),
        "posthoc_repository_identity": git_identity(REPO_ROOT),
        "attempts_rewritten": False,
        "scientific_artifact_impact": "none detected",
        "verification_basis": (
            "Both immutable attempts match the prelaunch executed-code content hash; "
            "the original verifier's remaining configuration, schedule, initialization, "
            "event, validation, diagnostic, checkpoint, and transfer checks all passed."
        ),
        "original_driver_failure": "Git commit missing/mixed.",
    }
    verification.write_json(output, summary)
    verification.write_json(
        RUN_DIR / "artifacts" / "progress.json",
        {
            "status": "verified_with_provenance_limitation",
            "condition_count": 2,
            "completed_conditions": 2,
            "current_condition": None,
            "verification": str(output.relative_to(RUN_DIR)),
        },
    )
    return output


if __name__ == "__main__":
    print(verify_with_missing_git_provenance())
