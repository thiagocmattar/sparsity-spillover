"""Build one explicit, hash-verified run025 input/code transfer, exclusively."""
from pathlib import Path
import argparse
import sys
import tarfile

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN))
from run025_common import ROOT, record, read_json, verify_record, write_json, select_conditions
import importlib.util
spec = importlib.util.spec_from_file_location("run025_package", RUN / "02_package.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--sizes", nargs="*", choices=["14m", "70m", "410m"], default=[])
    parser.add_argument("--code", action="store_true")
    args = parser.parse_args()
    if not args.name.replace("-", "").replace("_", "").isalnum():
        parser.error("Simple unique bundle name required")
    manifest = read_json(RUN / "prelaunch/input_manifest.json")
    rows = package.source_files()
    rows += [manifest[key] for key in ("validation", "validation_metadata", "development")]
    selected = [r for r in select_conditions(manifest, "development") if r["id"].split("/")[0] in args.sizes]
    for row in selected:
        rows += row["files"] + row["provenance"]
    if args.code:
        rows.append(record(RUN / "prelaunch/transfer-primitive.json"))
        rows += [record(p) for p in RUN.joinpath("autoresearch").rglob("*")
                 if p.is_file() and p.suffix in {".py", ".cu", ".sh", ".json", ".md"}
                 and not (set(p.parts) & {"artifacts", "bundles", "__pycache__"})]
    rows = list({row["path"]: row for row in rows}.values())
    for row in rows:
        verify_record(row)
    folder = RUN / "autoresearch/bundles" / args.name
    folder.mkdir(parents=True, exist_ok=False)
    inventory = folder / "inventory.json"
    write_json(inventory, {"name": args.name, "files": rows,
                          "conditions": [row["id"] for row in selected],
                          "bytes": sum(row["bytes"] for row in rows),
                          "excludes": ["credentials", "optimizer states", "interior checkpoint weights"]})
    archive_path = folder / "payload.tar.gz"
    with tarfile.open(archive_path, "x:gz", compresslevel=1) as archive:
        for row in rows + [record(inventory)]:
            archive.add(ROOT / row["path"], arcname=row["path"], recursive=False)
    result = record(archive_path)
    write_json(folder / "archive.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
