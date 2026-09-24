"""Verify uploaded source/initialization and data before installing or training."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    for name in ("deployment.json","data-transfer.json"):
        manifest = json.loads((ROOT/name).read_text())
        for row in manifest["files"]:
            path = (ROOT/row["path"]).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError("Input path escapes repository")
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle,"sha256").hexdigest()
            if digest != row["sha256"] or path.stat().st_size != row["bytes"]:
                raise RuntimeError("Input mismatch: "+row["path"])
        print(name,"verified",len(manifest["files"]),"files")


if __name__ == "__main__":
    main()
