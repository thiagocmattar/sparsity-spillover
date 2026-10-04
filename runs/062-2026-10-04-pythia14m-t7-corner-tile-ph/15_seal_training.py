"""Archive only terminal, verified evidence and required diagnostics/checkpoints."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from run_config import RUN_DIR as RUN,write_json,load_config,condition_specs

def record(path):
    with path.open('rb') as handle:sha=hashlib.file_digest(handle,'sha256').hexdigest()
    return {'path':path.relative_to(RUN).as_posix(),'bytes':path.stat().st_size,'sha256':sha}

if __name__=='__main__':
    verification=json.loads((RUN/'artifacts/verification.json').read_text())
    if verification['status']!='verified' or verification['condition_count']!=len(condition_specs(load_config())):
        raise RuntimeError('Complete verified cohort required')
    target=RUN/'transfer';target.mkdir(exist_ok=True)
    archive=target/'training-001.tar.gz'
    if archive.exists():raise FileExistsError('Preserve sealed outputs')
    logs=RUN/'prelaunch/remote-control';logs.mkdir(exist_ok=True)
    for path in Path('/workspace/run062-control').glob('*'):
        if path.is_file() and path.suffix in ('.log','.exit','.txt'):
            shutil.copyfile(path,logs/path.name)
    paths=sorted([p for p in (RUN/'artifacts').rglob('*') if p.is_file()]+
        [p for p in logs.rglob('*') if p.is_file()]+list((RUN/'prelaunch').glob('remote-preflight-*.json')))
    inventory={'files':[record(p) for p in paths]}
    inventory['total_bytes']=sum(r['bytes'] for r in inventory['files'])
    inventory_path=target/'training-inventory-001.json'
    write_json(inventory_path,inventory)
    with tarfile.open(archive,'w:gz',compresslevel=1) as bundle:
        for path in paths+[inventory_path]:bundle.add(path,arcname=path.relative_to(RUN).as_posix(),recursive=False)
    write_json(target/'training-receipt-001.json',record(archive))
    print(json.dumps({'archive':record(archive),'files':len(paths),'payload_bytes':inventory['total_bytes']}),flush=True)
