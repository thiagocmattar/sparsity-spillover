"""Retrieve immutable report inputs while the terminal checkpoint archive seals."""
import hashlib
from importlib import import_module
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_config import RUN_DIR as RUN,write_json
remote=import_module('10_remote')

if __name__=='__main__':
    client,lease=remote.connect()
    selected=[]
    try:
        with client.open_sftp() as sftp:
            root=remote.REMOTE+'/runs/'+RUN.name
            with sftp.open(root+'/transfer/training-inventory-001.json') as f: inventory=json.loads(f.read())
            for row in inventory['files']:
                relative=Path(row['path'])
                if not (row['path']=='artifacts/verification.json' or relative.name in ('metrics.json','activation_statistics.json')):
                    continue
                target=RUN/relative
                if not target.resolve().is_relative_to(RUN.resolve()):raise ValueError('Unscoped report input')
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    part=target.with_suffix(target.suffix+'.part')
                    sftp.get(root+'/'+row['path'],str(part))
                    with part.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
                    if part.stat().st_size!=row['bytes'] or sha!=row['sha256']:raise ValueError('Report input transfer differs')
                    part.replace(target)
                with target.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
                if sha!=row['sha256']:raise ValueError('Existing report input differs')
                selected.append(row)
        write_json(RUN/'prelaunch/report-input-retrieval-001.json',{'pod_id':lease['pod']['id'],
            'files':selected,'hashes_verified':True,'checkpoint_transfer_still_required':True})
        print(json.dumps({'report_inputs_verified':len(selected),'checkpoint_transfer_still_required':True}),flush=True)
    finally:client.close()
