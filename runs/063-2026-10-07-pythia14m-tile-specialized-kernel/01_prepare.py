"""Seal the two original tile models and byte-identical Run061 dependencies."""
import argparse
import shutil
import tarfile
from io_utils import RUN, REPO, fs, read, write, record, verify

BASE=REPO/'runs/061-2026-10-04-pythia14m-corner-tile-latency'

def prepare():
    if (RUN/'artifacts/attempts').exists():
        raise RuntimeError('Inputs are immutable after execution begins')
    copies=[]
    def copy(row):
        source=verify(row,BASE); target=RUN/row['path']
        fs(target.parent).mkdir(parents=True,exist_ok=True)
        shutil.copyfile(fs(source),fs(target))
        if record(target)!=row:raise ValueError('Copy identity mismatch')
        copies.append({'source':record(source,REPO),'copy':row})
    archive=read(BASE/'provenance/archive.json')
    for row in archive['files']:copy(row['snapshot'])
    write(RUN/'provenance/archive.json',archive)
    inputs=read(BASE/'provenance/inputs.json')
    inputs['checkpoints']=[c for c in inputs['checkpoints'] if c['family']=='tile']
    expected={v[0] for v in read(RUN/'config.json')['pair_conditions'].values()}
    if len(inputs['checkpoints'])!=2 or {c['id'] for c in inputs['checkpoints']}!=expected:
        raise ValueError('Exactly two approved tile checkpoints required')
    copy(inputs['validation'])
    for c in inputs['checkpoints']:
        for row in c['files']+c['provenance']:copy(row)
    write(RUN/'provenance/inputs.json',inputs)
    write(RUN/'provenance/reuse.json',{'files':copies,'frozen_base':BASE.relative_to(REPO).as_posix(),
          'generic_sha256':record(BASE/'joint.cu',BASE)['sha256'],
          'adaptation':'generic.cu byte-identical Run061; specialized joint.cu; no gate or weight change'})
    print('Prepared',len(copies),'verified immutable inputs')

def verify_all():
    rows=read(RUN/'provenance/reuse.json')['files']
    for row in rows:verify(row['copy'])
    if record(RUN/'generic.cu')['sha256']!=read(RUN/'provenance/reuse.json')['generic_sha256']:
        raise ValueError('Generic consumer source changed')
    print('Verified',len(rows),'input identities and generic consumer')

def bundle(tag):
    verify_all(); target=RUN/'bundles'/f'input-{tag}.tar'; target.parent.mkdir(exist_ok=True)
    if target.exists():raise ValueError('Fresh bundle tag required')
    paths=list(RUN.glob('*.py'))+list(RUN.glob('*.cu'))+list(RUN.glob('*.sh'))+[RUN/'config.json']
    paths += [RUN/r['copy']['path'] for r in read(RUN/'provenance/reuse.json')['files']]
    paths += [p for p in (RUN/'provenance').glob('*') if p.is_file()]
    paths=sorted(set(paths)); write(target.with_suffix('.inventory.json'),{'files':[record(p) for p in paths]})
    with tarfile.open(fs(target),'w') as archive:
        for path in paths:archive.add(fs(path),arcname=path.relative_to(RUN).as_posix(),recursive=False)
    write(target.with_suffix('.receipt.json'),record(target)); print(record(target))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','verify','bundle']);parser.add_argument('--tag',default='001')
    args=parser.parse_args();{'prepare':prepare,'verify':verify_all,'bundle':lambda:bundle(args.tag)}[args.action]()
