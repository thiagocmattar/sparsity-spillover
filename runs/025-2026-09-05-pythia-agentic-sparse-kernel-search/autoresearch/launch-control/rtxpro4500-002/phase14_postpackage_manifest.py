"""Write a stable hash record for the already-created Run 025 evidence archive."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ARCHIVE = Path("/workspace/run025-rtxpro4500-002-evidence.tar.gz")
OUTPUT = Path("/workspace/run025-rtxpro4500-002-postpackage.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)
    record = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": {
            "path": str(ARCHIVE),
            "bytes": ARCHIVE.stat().st_size,
            "sha256": sha256(ARCHIVE),
        },
        "purpose": (
            "Stable post-package identity for the immutable evidence archive; "
            "generated only after Phase 13 and all benchmark processes exited."
        ),
    }
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)


if __name__ == "__main__":
    main()
