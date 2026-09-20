"""Export closed trial folders and explicitly named closed runtime files."""
import argparse
import tarfile
from common import RUN, ROOT, read_json, write_json, record


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True)
    p.add_argument('--runtime',nargs='*',default=[])
    a=p.parse_args()
    if not a.name.replace('-','').isalnum():p.error('Simple unique name required')
    paths=[]
    for folder in (RUN/'artifacts').iterdir():
        if not (folder/'result.json').exists():continue
        if read_json(folder/'result.json')['status'] not in {'complete','failed'}:continue
        paths += [f for f in folder.rglob('*') if f.is_file()]
    for name in a.runtime:
        if '/' in name or '\\' in name:p.error('Runtime basenames only')
        paths.append(RUN/'runtime'/name)
    rows=[record(f) for f in sorted(set(paths))]
    dest=RUN/'bundles'/a.name;dest.mkdir(parents=True,exist_ok=False)
    write_json(dest/'inventory.json',{'files':rows,'bytes':sum(r['bytes'] for r in rows)})
    with tarfile.open(dest/'payload.tar.gz','x:gz',compresslevel=1) as tar:
        for row in rows+[record(dest/'inventory.json')]:tar.add(ROOT/row['path'],arcname=row['path'],recursive=False)
    receipt=record(dest/'payload.tar.gz');write_json(dest/'archive.json',receipt);print(receipt)


if __name__=='__main__':main()
