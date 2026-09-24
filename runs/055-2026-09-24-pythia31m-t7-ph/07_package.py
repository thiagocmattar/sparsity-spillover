"""Package explicit committed source and immutable inputs; never include credentials."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from run_config import REPO_ROOT, RUN_DIR, scientific_source_paths, git_identity, write_json
sys.path.insert(0,str(RUN_DIR/"latency"))
from source_identity import source_paths, identity


def make_archive(name, paths, repository):
    rows = []
    selected = {}
    for path in paths:
        normal = Path(str(path).removeprefix("\\\\?\\")).relative_to(REPO_ROOT).as_posix()
        selected[normal] = path
    for relative,path in sorted(selected.items()):
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle,"sha256").hexdigest()
        rows.append(dict(path=relative,bytes=path.stat().st_size,sha256=digest))
    manifest = dict(repository=repository, files=rows, total_bytes=sum(r["bytes"] for r in rows))
    target = RUN_DIR/"prelaunch"/(name+".tar")
    with tarfile.open(target,"w") as archive:
        for relative,path in sorted(selected.items()):
            archive.add(path,arcname=relative,recursive=False)
        payload = json.dumps(manifest,indent=2).encode()
        item = tarfile.TarInfo("data-transfer.json" if name=="input-data" else "deployment.json")
        item.size = len(payload)
        archive.addfile(item,io.BytesIO(payload))
    with target.open("rb") as handle:
        manifest["archive"] = dict(path=target.name,bytes=target.stat().st_size,
                                   sha256=hashlib.file_digest(handle,"sha256").hexdigest())
    write_json(RUN_DIR/"prelaunch"/(name+"-receipt.json"),manifest)
    print(json.dumps(manifest["archive"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",action="store_true")
    p.add_argument("--training-only",action="store_true",help="Omit unused kernel/vendor files on training Pods")
    args = p.parse_args()
    if subprocess.run(["git","status","--porcelain","--",str(RUN_DIR)],cwd=REPO_ROOT,
                      check=True,capture_output=True,text=True).stdout.strip():
        raise RuntimeError("Commit the run before packaging its source")
    repository = git_identity()
    repository["packaged_scientific_files_committed"] = True
    identity()  # Validate the shape-port provenance before packaging.
    paths = scientific_source_paths()+list((RUN_DIR/"prelaunch/initialization").glob("*"))
    if not args.training_only:
        paths += source_paths()
    paths += [REPO_ROOT/"pyproject.toml", REPO_ROOT/"README.md", RUN_DIR/"00_setup_remote.sh"]
    make_archive("input-training" if args.training_only else "input-source",paths,repository)
    if args.data:
        paths = [REPO_ROOT/"data/tokenized/minipile-pythia-14m-full"/split/name
                 for split in ("train","validation") for name in ("metadata.json","tokens.int32.bin")]
        make_archive("input-data",paths,repository)


if __name__ == "__main__":
    main()
