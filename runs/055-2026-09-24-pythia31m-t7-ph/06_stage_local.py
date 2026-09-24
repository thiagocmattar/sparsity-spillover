"""Explicit hash-verified copy into isolated WSL, without Windows drive mounts."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
from run_config import REPO_ROOT, RUN_DIR, scientific_source_paths, write_json

LINUX = "/home/researcher/sparsity-spillover/run055-workspace"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", action="store_true")
    p.add_argument("--latency", action="store_true")
    args = p.parse_args()
    paths = scientific_source_paths()+list((RUN_DIR/"prelaunch/initialization").glob("*"))
    if args.latency:
        source = RUN_DIR.parent/"045-2026-09-20-pythia70m-kernel-grid"
        paths += [source/name for name in ("io_utils.py", "replay.py", "frozen_replay.py",
                  "controls.py", "hz_adapter.py", "hz_topology.py")]
        for folder in (source/"archive", RUN_DIR/"latency"):
            extended = Path("\\\\?\\"+str(folder.absolute()))
            for base, dirs, files in os.walk(extended):
                dirs[:] = [d for d in dirs if d not in ("__pycache__", "artifacts")]
                paths += [Path(base)/name for name in files if not name.endswith(".pyc")]
    if args.data:
        for split in ("train", "validation"):
            paths += list((REPO_ROOT/"data/tokenized/minipile-pythia-14m-full"/split).glob("*"))
    rows = []
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        normal = Path(str(path).removeprefix("\\\\?\\"))
        rows.append(dict(path=normal.relative_to(REPO_ROOT).as_posix(), bytes=path.stat().st_size, sha256=digest))
    write_json(RUN_DIR/"prelaunch/local-transfer.json", dict(linux_root=LINUX, files=rows))
    subprocess.run(["wsl.exe", "-d", "SparsityGPU", "--exec", "mkdir", "-p", LINUX], check=True)
    proc = subprocess.Popen(["wsl.exe", "-d", "SparsityGPU", "--exec", "tar", "-xf", "-", "-C", LINUX], stdin=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdin, mode="w|") as archive:
        for row in rows:
            archive.add(Path("\\\\?\\"+str(REPO_ROOT/row["path"])), arcname=row["path"], recursive=False)
        payload = json.dumps(rows).encode()
        item = tarfile.TarInfo("input-transfer.json")
        item.size = len(payload)
        archive.addfile(item, io.BytesIO(payload))
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("WSL transfer failed")
    verification = '''import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
rows=json.loads((root/'input-transfer.json').read_text())
for r in rows:
 p=root/r['path']
 with p.open('rb') as f: h=hashlib.file_digest(f,'sha256').hexdigest()
 assert p.stat().st_size==r['bytes'] and h==r['sha256'],r['path']
print(json.dumps({'verified_files':len(rows),'bytes':sum(r['bytes'] for r in rows)}))
'''
    result = subprocess.run(["wsl.exe", "-d", "SparsityGPU", "--exec", "python3", "-", LINUX],
                            input=verification, text=True, capture_output=True, check=True)
    write_json(RUN_DIR/"prelaunch/local-transfer-receipt.json", json.loads(result.stdout))
    print(result.stdout)


if __name__ == "__main__":
    main()
