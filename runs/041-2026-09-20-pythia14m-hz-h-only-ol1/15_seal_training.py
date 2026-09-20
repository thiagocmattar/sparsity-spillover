"""Seal terminal Run041 training evidence for hash-verified local retrieval."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

RUN=Path(__file__).resolve().parent


def record(path):
    with path.open('rb') as handle:
        digest=hashlib.file_digest(handle,'sha256').hexdigest()
    return {'path':path.relative_to(RUN).as_posix(),
            'bytes':path.stat().st_size,'sha256':digest}


def main():
    control=Path('/workspace/run041-control')
    if not (control/'training-verified').exists():
        raise RuntimeError('Training must finish and verify before sealing')
    verification=json.loads((RUN/'artifacts/verification.json').read_text())
    assert verification['status']=='verified' and verification['condition_count']==4
    target=RUN/'transfer';target.mkdir(exist_ok=True)
    archive=target/'training-001.tar.gz'
    if archive.exists():raise FileExistsError('Preserve previous sealed outputs')
    logs=RUN/'prelaunch/remote-control';logs.mkdir(exist_ok=True)
    for path in control.iterdir():
        if path.is_file() and path.suffix in {'.log','.exit','.txt'}:
            shutil.copyfile(path,logs/path.name)
    paths=[p for p in (RUN/'artifacts').rglob('*') if p.is_file()]
    paths+=list(logs.glob('*'))+list((RUN/'prelaunch').glob('remote-preflight-*.json'))
    paths=sorted(set(paths))
    inventory={'files':[record(p) for p in paths]}
    inventory_path=target/'training-inventory-001.json'
    inventory_path.write_text(json.dumps(inventory,indent=2)+'\n')
    with tarfile.open(archive,'w:gz',compresslevel=1) as bundle:
        for path in paths+[inventory_path]:
            bundle.add(path,arcname=path.relative_to(RUN).as_posix(),recursive=False)
    receipt=record(archive)
    (target/'training-receipt-001.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'archive':receipt,'files':len(paths),
        'payload_bytes':sum(r['bytes'] for r in inventory['files'])}),flush=True)


if __name__=='__main__':main()
