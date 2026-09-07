"""Small source overlay; original model/data bundle is independently reused."""
import argparse
import tarfile
from common import RUN, ROOT, R27, R25, R26, record, read_json, write_json, verify_record


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--name')
    p.add_argument('--verify')
    a=p.parse_args()
    if a.verify:
        rows=read_json(a.verify)['files']
        for row in rows:verify_record(row)
        print({'verified_files':len(rows),'bytes':sum(r['bytes'] for r in rows)})
        return
    if not a.name or not a.name.replace('-','').isalnum():p.error('Unique simple bundle name required')
    paths=[f for f in RUN.glob('*') if f.is_file() and f.suffix in {'.py','.json','.md','.sh','.ps1'}]
    paths += [f for f in (RUN/'candidates').rglob('*') if f.is_file() and (f.suffix in {'.py','.cu','.h','.cuh','.md','.json'} or f.name.startswith('LICENSE'))]
    paths += [f for f in R27.glob('*') if f.is_file() and f.suffix in {'.py','.cu','.json'}]
    paths += [R27/'prelaunch/inputs.json']
    paths += [R25/name for name in ['run025_common.py','measurement.py','config.json','autoresearch/dense_probe/probe.py']]
    paths += [R26/f'autoresearch/candidates/{k}/{name}' for k in ['k018','k019'] for name in ['candidate.py','kernel.cu']]
    paths += list((ROOT/'src').rglob('*.py'))
    rows=[record(f) for f in sorted(set(paths))]
    dest=RUN/'bundles'/a.name;dest.mkdir(parents=True,exist_ok=False)
    write_json(dest/'inventory.json',{'files':rows,'bytes':sum(r['bytes'] for r in rows)})
    with tarfile.open(dest/'payload.tar.gz','x:gz',compresslevel=1) as tar:
        for row in rows+[record(dest/'inventory.json')]:tar.add(ROOT/row['path'],arcname=row['path'],recursive=False)
    result=record(dest/'payload.tar.gz');write_json(dest/'archive.json',result);print(result)


if __name__=='__main__':main()
