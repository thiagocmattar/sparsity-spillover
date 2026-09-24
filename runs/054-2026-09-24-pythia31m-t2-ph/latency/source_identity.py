"""Hash the exact local specialization and archived runtime dependencies."""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE = HERE.parents[1]/"045-2026-09-20-pythia70m-kernel-grid"
ROOT_FILES = ("io_utils.py", "replay.py", "frozen_replay.py", "controls.py", "hz_adapter.py", "hz_topology.py")


def source_paths():
    paths = [*HERE.glob("*.py"), HERE/"port-provenance.json", *[SOURCE/p for p in ROOT_FILES]]
    for folder in (HERE/"kernel", SOURCE/"archive"):
        folder = Path("\\\\?\\"+str(folder)) if os.name == "nt" else folder
        for base, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", "artifacts", ".git")]
            paths.extend(Path(base)/name for name in files if not name.endswith(".pyc"))
    return sorted(set(paths))


def identity():
    provenance = json.loads((HERE/"port-provenance.json").read_text())
    for name, row in provenance["files"].items():
        payload = (HERE/"kernel"/name).read_bytes().replace(b"\r\n",b"\n")
        if hashlib.sha256(payload).hexdigest() != row["port_sha256"]:
            raise RuntimeError("Port drift: "+name)
    rows = []
    for path in source_paths():
        # All declared runtime dependencies are text; ignore checkout newline policy.
        payload = path.read_bytes().replace(b"\r\n",b"\n")
        relative = Path(str(path).removeprefix("\\\\?\\")).relative_to(REPO).as_posix()
        rows.append(dict(path=relative, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
    return dict(files=rows, content_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
                text_normalization="CRLF to LF")
