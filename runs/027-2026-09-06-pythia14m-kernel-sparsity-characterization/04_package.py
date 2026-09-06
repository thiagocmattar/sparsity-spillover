"""Explicit model-only/source transport, terminal export, and SHA256 verification."""
import argparse
import tarfile
from pathlib import Path
from run027_common import ROOT,RUN,R25,R26,read_json,write_json,record,verify_record


def main():
    p=argparse.ArgumentParser(); p.add_argument('--name'); p.add_argument('--code-only',action='store_true')
    p.add_argument('--export',action='store_true'); p.add_argument('--verify',type=Path)
    a=p.parse_args()
    if a.verify:
        manifest=read_json(a.verify)
        for row in manifest['files']: verify_record(row)
        print(f'Verified {len(manifest["files"])} files / {manifest["bytes"]} bytes');return
    if not a.name or not a.name.replace('-','').isalnum():p.error('Unique simple name required')
    paths=[]
    if a.export:
        for folder in (RUN/'artifacts').iterdir():
            if not (folder/'result.json').exists(): continue
            if read_json(folder/'result.json')['status']=='running':continue
            paths.extend(f for f in folder.rglob('*') if f.is_file())
        paths.extend(f for f in (RUN/'runtime').glob('*') if f.is_file() and f.suffix in {'.json','.log','.txt'})
    else:
        paths+=list((ROOT/'src').rglob('*.py'))
        paths += [R25/name for name in ['run025_common.py','measurement.py','config.json','autoresearch/dense_probe/probe.py']]
        paths += [R26/f'autoresearch/candidates/{k}/{name}' for k in ['k018','k019'] for name in ['candidate.py','kernel.cu']]
        paths += [f for f in RUN.glob('*') if f.is_file() and f.suffix in {'.py','.cu','.sh','.ps1','.json','.md'}]
        paths += list((RUN/'prelaunch').glob('*.json'))
    rows=[record(f) for f in paths]
    if not a.code_only and not a.export:
        data=read_json(RUN/'prelaunch/inputs.json')
        rows+=list(data['inputs'].values())+[data['cohort_source']]
        for checkpoint in data['checkpoints']: rows+=checkpoint['files']+checkpoint['provenance']
    rows=list({r['path']:r for r in rows}.values())
    for row in rows:verify_record(row)
    dest=RUN/'bundles'/a.name; dest.mkdir(parents=True,exist_ok=False)
    write_json(dest/'inventory.json',{'files':rows,'bytes':sum(r['bytes'] for r in rows)})
    archive=dest/'payload.tar.gz'
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for row in rows+[record(dest/'inventory.json')]:tar.add(ROOT/row['path'],arcname=row['path'],recursive=False)
    receipt=record(archive);write_json(dest/'archive.json',receipt);print(receipt,flush=True)


if __name__=='__main__':main()
