"""Restore timestamps on unchanged copied files so Ninja reuses verified objects."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

RUN=Path('/workspace/run037')
OUT=RUN/'artifacts/infrastructure/003-local-runtime'

def restore(item):
    source,target,record=item
    if hashlib.file_digest(target.open('rb'),'sha256').hexdigest()!=record['sha256']:
        return {'path':str(target),'restored':False,'reason':'File changed since verified copy'}
    shutil.copystat(source,target)
    return {'path':str(target),'restored':True}

if __name__=='__main__':
    report=json.loads((OUT/'verified-copy.json').read_text())
    items=[]
    for tree in report['trees']:
        source=Path(tree['source']+'-network-original')
        target=Path(tree['destination'])
        items.extend((source/r['path'],target/r['path'],r) for r in tree['files'])
    with ThreadPoolExecutor(max_workers=12) as pool:
        rows=list(pool.map(restore,items))
    (OUT/'metadata-restoration.json').write_text(json.dumps({'files':rows},indent=2)+'\n')
    print(json.dumps({'restored':sum(r['restored'] for r in rows),'changed':sum(not r['restored'] for r in rows)}),flush=True)
