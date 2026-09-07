"""Explicit input/output bundles with portable byte identities (no credentials)."""
import argparse
from pathlib import Path
import tarfile
import shutil
from io_utils import RUN, fs, read, write, record, verify


def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['code','inputs','outputs','verify'])
    p.add_argument('--tag',default='001');a=p.parse_args()
    if not a.tag.replace('-','').isalnum():raise ValueError('Simple tag required')
    if a.kind=='verify':
        for row in read(RUN/'provenance/archive.json')['files']:verify(row['snapshot'])
        inputs=read(RUN/'provenance/inputs.json')
        for row in [inputs['validation']]+[f for c in inputs['checkpoints'] for f in c['files']+c['provenance']]:verify(row)
        print('All archived sources and inputs verified');return
    if a.kind=='inputs':
        value=read(RUN/'provenance/inputs.json');rows=[value['validation']]+[f for c in value['checkpoints'] for f in c['files']]
    elif a.kind=='code':
        paths=list(RUN.glob('*.py'))+list(RUN.glob('*.sh'))+list(RUN.glob('*.json'))+list((RUN/'tests').glob('*.py'))
        paths+=list((RUN/'provenance').rglob('*'))
        paths+=list((RUN/'upstream-reference').rglob('*'))
        rows=[record(p) for p in paths if fs(p).is_file()]
        # Use the manifest for deep Windows paths: ordinary rglob/is_file can
        # silently omit these even when extended-path copies succeeded.
        rows += [f for c in read(RUN/'provenance/inputs.json')['checkpoints'] for f in c['provenance']]
        rows += [r['snapshot'] for r in read(RUN/'provenance/archive.json')['files']]
    else:
        # Only completed leaves are immutable and eligible for incremental export.
        paths=[]
        for result in (RUN/'artifacts/attempts').glob('*/result.json'):
            paths+=list(result.parent.iterdir())
        for phase in ['smoke','calibration','scientific']:
            for pth in (RUN/'artifacts'/phase).glob('*'):
                if pth.suffix in {'.json','.log'}:paths.append(pth)
        paths += list((RUN/'runtime').glob('*.txt'))+list((RUN/'runtime').glob('*.log'))
        stage=RUN/'bundles'/f'output-snapshot-{a.tag}'
        stage.mkdir(parents=True,exist_ok=False)
        rows=[]
        for source in paths:
            if source.is_file():
                target=stage/source.relative_to(RUN);target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,target)
                rows.append(record(target,stage))
    # Duplicate provenance entries are collapsed, not silently overwritten in tar.
    rows=list({r['path']:r for r in rows}.values())
    folder=RUN/'bundles';folder.mkdir(exist_ok=True)
    inventory=folder/f'{a.kind}-{a.tag}-inventory.json'
    bundle=folder/f'{a.kind}-{a.tag}.tar'
    if bundle.exists():raise ValueError('Use a fresh bundle tag')
    root=stage if a.kind=='outputs' else RUN
    for row in rows:verify(row,root)
    write(inventory,{'kind':a.kind,'files':rows,'bytes':sum(r['bytes'] for r in rows)})
    with tarfile.open(bundle,'w') as tar:
        for row in rows:tar.add(fs(root/row['path']),arcname=row['path'],recursive=False)
        tar.add(inventory,arcname=inventory.relative_to(RUN).as_posix())
    receipt=record(bundle);write(bundle.with_suffix('.receipt.json'),receipt)
    print(receipt)


if __name__=='__main__':main()
