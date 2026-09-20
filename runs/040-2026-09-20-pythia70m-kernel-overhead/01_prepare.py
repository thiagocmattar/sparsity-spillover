"""Retain the approved two endpoints, development blocks and frozen sources."""
import argparse
import json
import shutil
import tarfile
from io_utils import RUN, REPO, fs, read, write, record, verify, sha

BASE = REPO / "runs/035-2026-09-18-pythia70m-k050-port"


def prepare():
    if (RUN / "artifacts/attempts").exists():
        raise RuntimeError("Inputs are immutable after execution begins")
    copies = []

    def copy(source, relative, expected=None, source_root=BASE):
        if expected and record(source, source_root) != expected:
            raise ValueError(f"Source identity mismatch: {source}")
        dest = RUN / relative
        assert dest.resolve().is_relative_to(RUN)
        fs(dest.parent).mkdir(parents=True, exist_ok=True)
        if not fs(dest).exists() or sha(dest) != sha(source):
            shutil.copyfile(fs(source), fs(dest))
        copies.append({"source": record(source, REPO), "copy": record(dest)})

    archive = read(BASE / "provenance/archive.json")
    for row in archive["files"]:
        item = row["snapshot"]
        copy(BASE / item["path"], item["path"], item)
    write(RUN / "provenance/archive.json", archive)
    manifest = read(BASE / "provenance/inputs.json")
    selected = [r for r in manifest["checkpoints"] if r["id"] in ("c00", "c21")]
    assert len(selected) == 2
    assert selected[0]["family"] == "A0"
    assert selected[1]["family"] == "A7+OL1@h" and selected[1]["dose"] == .5
    for row in [manifest["validation"]] + [f for c in selected for f in c["files"] + c["provenance"]]:
        copy(BASE / row["path"], row["path"], row)
    dev = read(REPO / "runs/027-2026-09-06-pythia14m-kernel-sparsity-characterization/prelaunch/inputs.json")["inputs"]["development"]
    copy(REPO / dev["path"], "inputs/development.int32.bin", dev, REPO)
    write(RUN / "provenance/inputs.json", {**manifest, "checkpoints": selected,
          "development": record(RUN / "inputs/development.int32.bin"),
          "development_selection": "first16 complete blocks of the retained training development cache"})
    frozen = read(BASE / "artifacts/frozen-final-port.json")
    for row in frozen["sources"].values():
        copy(BASE / row["path"], row["path"], row)
    write(RUN / "provenance/frozen-port.json", frozen)
    copy(BASE / "provenance/pip-freeze.txt", "provenance/pip-freeze.txt")
    write(RUN / "provenance/reuse.json", {"base_run": BASE.relative_to(REPO).as_posix(), "files": copies})
    verify_inputs()


def verify_inputs():
    reuse = read(RUN / "provenance/reuse.json")
    for item in reuse["files"]:
        verify(item["copy"])
    print(f"Verified {len(reuse['files'])} immutable source/input copies", flush=True)


def bundle(tag):
    verify_inputs()
    assert tag.isalnum()
    target = RUN / "bundles" / f"input-{tag}.tar"
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        raise FileExistsError("Use a new bundle tag")
    paths = list(RUN.glob("*.py")) + list(RUN.glob("*.sh")) + [RUN / "config.json", RUN / "README.md"]
    paths += [RUN / item["copy"]["path"] for item in read(RUN / "provenance/reuse.json")["files"]]
    paths += [p for p in (RUN / "provenance").glob("*") if p.is_file()]
    paths = sorted(set(paths))
    write(target.with_name(f"input-{tag}-inventory.json"), {"files": [record(p) for p in paths]})
    with tarfile.open(target, "w") as tar:
        for p in paths:
            tar.add(fs(p), arcname=p.relative_to(RUN).as_posix(), recursive=False)
    write(target.with_suffix(".receipt.json"), record(target))
    print(json.dumps(record(target)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "verify", "bundle"))
    parser.add_argument("--tag", default="001")
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "verify": verify_inputs()
    else: bundle(args.tag)
