"""Pin two 14M references and create a portable, hash-verified launch inventory."""
import argparse, copy, shutil, tarfile
from pathlib import Path
from support import RUN, BASE, read, write, sha, fs


def record(path,root):
    return {'path':path.relative_to(root).as_posix(),'bytes':fs(path).stat().st_size,'sha256':sha(path)}


def prepare():
    if (RUN/'artifacts').exists():raise RuntimeError('Executed inputs are immutable')
    source=RUN.parent/'041-2026-09-20-pythia14m-hz-h-only-ol1/latency'
    original=read(source/'provenance/inputs.json');out={'checkpoints':[],'source_manifest_sha256':sha(source/'provenance/inputs.json')}
    for cid,dose in [('d05',.05),('d10',.1)]:
        old=next(c for c in original['checkpoints'] if c['dose']==dose)
        row=copy.deepcopy(old);row['id']=cid;row['checkpoint']='inputs/reference14/'+cid
        row['source_id']=old['id'];row['source_manifest']=(source/'provenance/inputs.json').relative_to(RUN.parents[1]).as_posix()
        for kind in ('files','provenance'):
            row[kind]=[]
            for i,item in enumerate(old[kind]):
                p=source/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
                name=Path(item['path']).name
                target=RUN/row['checkpoint']/name if kind=='files' else RUN/f'provenance/reference14/{cid}/{i:02d}-{name}'
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
                row[kind].append(record(target,RUN))
        out['checkpoints'].append(row)
    write(RUN/'provenance/reference14.json',out)
    write(RUN/'provenance/reference-policy.json',{
        'dense':read(RUN/'provenance/prior-selection.json')['dense'],
        'policies':{'prior_c':read(RUN/'provenance/prior-selection.json')['policies']['sparse_c']},
        'ablations':{},'purpose':'Immutable pre-search 70M reference; not a new candidate selection'})
    verify()


def inventory():
    # Reuse the already sealed Run049 archive and inputs at their original
    # relative paths. No logs, credentials, full training cache or GPU artifacts.
    old=[BASE/r['copy']['path'] for r in read(BASE/'provenance/reuse.json')['files']]
    old += [BASE/r['path'] for r in read(BASE/'provenance/source-freeze.json')['files']]
    old += [BASE/'provenance'/n for n in ('inputs.json','reuse.json','source-freeze.json','archive.json','pip-freeze.txt')]
    new=list(RUN.glob('*.py'))+list(RUN.glob('*.cu'))+list(RUN.glob('*.sh'))+[RUN/'config.json',RUN/'README.md']
    new += [p for folder in ('inputs','provenance','candidate14') for p in (RUN/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    return sorted(set(old+new))


def verify():
    for row in read(BASE/'provenance/reuse.json')['files']:
        item=row['copy'];p=BASE/item['path'];assert record(p,BASE)==item,p
    for row in read(BASE/'provenance/source-freeze.json')['files']:
        assert record(BASE/row['path'],BASE)==row,row['path']
    for c in read(RUN/'provenance/reference14.json')['checkpoints']:
        for item in c['files']+c['provenance']:assert record(RUN/item['path'],RUN)==item
    train=read(RUN/'provenance/inputs.json')['training'];assert record(RUN/train['path'],RUN)==train
    print('Verified prior source/input records, both 14M checkpoints, and training prefix')


def bundle():
    verify();root=RUN.parent
    paths=inventory();rows=[record(p,root) for p in paths]
    dest=RUN/'bundles/input-001.tar.gz';dest.parent.mkdir(exist_ok=True)
    if dest.exists():raise FileExistsError(dest)
    write(RUN/'prelaunch/transfer-inventory.json',{'files':rows,'bytes':sum(r['bytes'] for r in rows)})
    with tarfile.open(dest,'w:gz',compresslevel=1) as archive:
        for p in paths:archive.add(fs(p),arcname=p.relative_to(root).as_posix(),recursive=False)
    write(RUN/'prelaunch/bundle.json',record(dest,RUN))
    print({'files':len(rows),'input_bytes':sum(r['bytes'] for r in rows),'archive_bytes':dest.stat().st_size})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','verify','bundle'));a=p.parse_args()
    {'prepare':prepare,'verify':verify,'bundle':bundle}[a.action]()
