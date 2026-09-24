"""Seal terminal training/latency evidence for hash-verified retrieval."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from run_config import RUN_DIR, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag",required=True)
    args = p.parse_args()
    if not args.tag.replace("-","").isalnum():
        raise ValueError("Simple unique tag required")
    pipeline = json.loads((RUN_DIR/"artifacts/pipeline"/args.tag/"status.json").read_text())
    if pipeline["status"] not in ("completed","failed"):
        raise RuntimeError("Do not seal files that are still being written")
    roots = [RUN_DIR/"artifacts",RUN_DIR/"latency/artifacts",RUN_DIR/"latency/logs"]
    roots += list((RUN_DIR/"prelaunch").glob("calibration-"+args.tag+"*"))
    files = sorted({p for root in roots for p in root.rglob("*") if p.is_file()})
    rows = []
    for path in files:
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle,"sha256").hexdigest()
        rows.append(dict(path=path.relative_to(RUN_DIR).as_posix(),bytes=path.stat().st_size,sha256=digest))
    manifest = dict(files=rows,total_bytes=sum(r["bytes"] for r in rows),tag=args.tag)
    receipt = RUN_DIR/"prelaunch"/("retrieval-"+args.tag+".json")
    write_json(receipt,manifest)
    archive_path = RUN_DIR/"prelaunch"/("retrieval-"+args.tag+".tar")
    with tarfile.open(archive_path,"x") as archive:
        for row in rows:
            archive.add(RUN_DIR/row["path"],arcname=row["path"],recursive=False)
        archive.add(receipt,arcname="retrieval-inventory.json",recursive=False)
    with archive_path.open("rb") as handle:
        manifest["archive"] = dict(path=archive_path.name,bytes=archive_path.stat().st_size,
                                   sha256=hashlib.file_digest(handle,"sha256").hexdigest())
    write_json(receipt,manifest)
    print(json.dumps(manifest["archive"]))


if __name__ == "__main__":
    main()
