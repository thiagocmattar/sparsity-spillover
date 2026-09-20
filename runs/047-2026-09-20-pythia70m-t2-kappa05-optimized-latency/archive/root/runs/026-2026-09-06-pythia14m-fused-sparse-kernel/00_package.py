"""Explicit 14M-only source/checkpoint transfer; excludes credentials and caches."""
from pathlib import Path
import argparse
import sys
import tarfile

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
SOURCE = ROOT/'runs/025-2026-09-05-pythia-agentic-sparse-kernel-search'
sys.path.insert(0, str(SOURCE))
from run025_common import record, read_json, verify_record, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name')
    p.add_argument('--verify', type=Path)
    p.add_argument('--code-only', action='store_true')
    a = p.parse_args()
    if a.verify:
        inventory = read_json(a.verify)
        for row in inventory['files']:
            verify_record(row)
        print(f"Verified {len(inventory['files'])} files, {inventory['bytes']} bytes", flush=True)
        return
    if not a.name or not a.name.replace('-', '').isalnum():
        p.error('Simple unique name required')
    files = list((ROOT/'src').rglob('*.py'))
    files += [SOURCE/q for q in ['config.json', 'prelaunch/input_manifest.json',
              'run025_common.py', 'measurement.py', 'autoresearch/dense_probe/probe.py',
              'autoresearch/candidates/k001/candidate.py', 'autoresearch/candidates/k001/kernel.cu',
              'upstream/LICENSE']]
    files += [f for f in RUN.rglob('*') if f.is_file() and f.suffix in {'.py','.cu','.sh','.md','.json','.ps1'}
              and not set(f.relative_to(RUN).parts)&{'bundles','artifacts','runtime','launch-control','retrieved'}]
    inventory_rows = [record(f) for f in files]
    if not a.code_only:
        manifest = read_json(SOURCE/'prelaunch/input_manifest.json')
        selected = read_json(RUN/'config.json')['condition']
        checkpoint = next(r for r in manifest['checkpoints'] if r['id'] == selected)
        inventory_rows += checkpoint['files'] + checkpoint['provenance']
        inventory_rows += [manifest[k] for k in ['development','validation','validation_metadata']]
    inventory_rows = list({r['path']: r for r in inventory_rows}.values())
    for row in inventory_rows:
        verify_record(row)
    dest = RUN/'bundles'/a.name
    dest.mkdir(parents=True, exist_ok=False)
    path = dest/'inventory.json'
    write_json(path, {'files': inventory_rows, 'bytes': sum(r['bytes'] for r in inventory_rows)})
    with tarfile.open(dest/'payload.tar.gz', 'x:gz', compresslevel=1) as archive:
        for row in inventory_rows + [record(path)]:
            archive.add(ROOT/row['path'], arcname=row['path'], recursive=False)
    write_json(dest/'archive.json', record(dest/'payload.tar.gz'))
    print(read_json(dest/'archive.json'), flush=True)


if __name__ == '__main__':
    main()
