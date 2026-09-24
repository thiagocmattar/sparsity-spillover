"""Check an already downloaded archive, extract safely, and verify its evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from sparsity_research.artifacts import verify_transfer_inventory
from run_config import RUN_DIR, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    args = p.parse_args()
    receipt = json.loads(args.receipt.read_text())
    with args.archive.open("rb") as handle:
        digest = hashlib.file_digest(handle,"sha256").hexdigest()
    if digest != receipt["archive"]["sha256"] or args.archive.stat().st_size != receipt["archive"]["bytes"]:
        raise RuntimeError("Archive identity mismatch")
    with tarfile.open(args.archive) as archive:
        archive.extractall(RUN_DIR,filter="data")
    verify_transfer_inventory(RUN_DIR,receipt)
    subprocess.run([sys.executable,str(RUN_DIR/"03_verify.py")],check=True)
    grid = json.loads((RUN_DIR/"latency/artifacts"/(receipt["tag"]+"-grid.json")).read_text())["outcomes"]
    if len(grid) != 18 or not all(r["qualified"] and r["returncode"]==0 for r in grid):
        raise RuntimeError("Final latency cohort incomplete or unqualified")
    for row in grid:
        folder = RUN_DIR/"latency/artifacts/attempts"/row["attempt"]
        manifest = json.loads((folder/"manifest.json").read_text())
        quality = json.loads((folder/"quality.json").read_text())
        if not manifest["scientific_measurement"] or quality["blocks"] != 338:
            raise RuntimeError("Final result cannot be a calibration or partial qualification")
    write_json(RUN_DIR/"prelaunch"/("retrieved-"+receipt["tag"]+".json"),
               dict(status="verified",files=len(receipt["files"]),bytes=receipt["total_bytes"],archive_sha256=digest))
    print("Full training and latency evidence verified locally")


if __name__ == "__main__":
    main()
