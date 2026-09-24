"""Build and independently check the verified supplement against 100,000,000 bytes."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "handoff"
ARCHIVE = ROOT / "handoff.zip"
LIMIT = 100_000_000


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    manifest = (SOURCE / "MANIFEST.json").read_bytes()
    verification = json.loads((HERE / "verification.json").read_text())
    assert verification["status"] == "passed"
    assert verification["manifest_sha256"] == sha(manifest), "Package changed since isolated verification"
    records = json.loads(manifest)["files"]
    records.append(dict(path="MANIFEST.json", bytes=len(manifest), sha256=sha(manifest)))
    assert {r["path"] for r in records} == {p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob("*") if p.is_file()}
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for r in sorted(records, key=lambda r: r["path"]):
            path = (SOURCE / r["path"]).resolve()
            assert path.is_relative_to(SOURCE.resolve())
            raw = path.read_bytes()
            assert len(raw) == r["bytes"] and sha(raw) == r["sha256"], path
            entry = zipfile.ZipInfo("handoff/" + r["path"], date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, raw, compresslevel=9)
    size = ARCHIVE.stat().st_size
    assert size < LIMIT, (size, LIMIT)
    scratch = Path(tempfile.mkdtemp(prefix="handoff-zip-review-", dir=ROOT / "tmp"))
    expected = {"handoff/" + r["path"]: r for r in records}
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        assert len(archive.infolist()) == len(expected)
        assert set(archive.namelist()) == set(expected)
        for name, r in expected.items():
            target = (scratch / name).resolve()
            assert target.is_relative_to(scratch.resolve())
            raw = archive.read(name)
            assert sha(raw) == r["sha256"] and len(raw) == r["bytes"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    env = dict(os.environ, PYTHONPATH=str(scratch / "handoff/src"), PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    checks = []
    for action in ("verify", "results"):
        result = subprocess.run([sys.executable, "scripts/reproduce.py", action], cwd=scratch / "handoff",
                                env=env, text=True, capture_output=True, timeout=60)
        checks.append(dict(action=action, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        assert result.returncode == 0, checks[-1]
    summary = dict(status="passed", archive="handoff.zip", bytes=size, decimal_MB=size/1_000_000,
                   MiB=size/2**20, maximum_bytes=LIMIT, remaining_bytes=LIMIT-size,
                   fraction_of_limit=size/LIMIT, files=len(records), uncompressed_bytes=sum(r["bytes"] for r in records),
                   sha256=sha(ARCHIVE.read_bytes()), manifest_sha256=sha(manifest), crc_check="passed",
                   every_extracted_file_hash="passed", checks=checks,
                   extracted_directory=scratch.relative_to(ROOT).as_posix())
    (HERE / "zip-verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
