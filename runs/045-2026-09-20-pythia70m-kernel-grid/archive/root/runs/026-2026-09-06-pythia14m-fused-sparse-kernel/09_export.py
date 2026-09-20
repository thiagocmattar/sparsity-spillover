"""Snapshot completed evidence with hashes; never traverse runtime environments."""
from pathlib import Path
import argparse
import sys
import tarfile

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
sys.path.insert(0,str(ROOT/'runs/025-2026-09-05-pythia-agentic-sparse-kernel-search'))
from run025_common import read_json,write_json,record,verify_record


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True)
    p.add_argument('--terminal',action='store_true', help='Caller has verified all GPU workers terminal')
    a=p.parse_args()
    if not a.name.replace('-','').isalnum(): p.error('Simple unique name required')
    paths=[]
    for attempt in sorted((RUN/'autoresearch/artifacts').glob('*')):
        result=attempt/'result.json'
        if not result.exists() or read_json(result)['status']=='running':
            continue
        paths.extend(f for f in attempt.rglob('*') if f.is_file())
    # These preflight records are terminal before search starts. Active phase
    # logs are deliberately excluded; collect them at their own closeout.
    for name in ['primitive.json','primitive-joint.json','pip-freeze.txt',
                 'nvcc-version.txt','calibration.log','calibration2.log']:
        f=RUN/'runtime'/name
        if f.is_file(): paths.append(f)
    if a.terminal:
        for name in ['primitive-rope.json','graph-first-divergence.json',
                     'phase2.log','phase3.log','phase4.log','phase5.log',
                     'final.log','diagnostics.log']:
            f=RUN/'runtime'/name
            if not f.is_file(): raise ValueError(f'Missing terminal evidence: {f}')
            paths.append(f)
        paths.extend(f for f in (RUN/'runtime/diagnostics').glob('*.json') if f.is_file())
    progress=RUN/'autoresearch/progress.csv'
    if progress.exists(): paths.append(progress)
    rows=[record(f) for f in paths]
    dest=RUN/'bundles'/a.name
    dest.mkdir(parents=True,exist_ok=False)
    inventory=dest/'inventory.json'
    write_json(inventory,{'files':rows,'bytes':sum(r['bytes'] for r in rows)})
    archive_path=dest/'evidence.tar.gz'
    with tarfile.open(archive_path,'x:gz',compresslevel=1) as archive:
        for row in rows+[record(inventory)]:
            verify_record(row)
            archive.add(ROOT/row['path'],arcname=row['path'],recursive=False)
    receipt=record(archive_path)
    write_json(dest/'archive.json',receipt)
    print(receipt,flush=True)


if __name__=='__main__':main()
