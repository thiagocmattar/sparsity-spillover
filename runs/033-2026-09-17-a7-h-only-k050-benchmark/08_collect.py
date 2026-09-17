"""Archive terminal outputs, runtime records and source identities for retrieval."""
import hashlib
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent


def record(path):
    return {'path':path.relative_to(HERE).as_posix(),'bytes':path.stat().st_size,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    assert not (HERE/'artifacts/controller.lock').exists(), 'Controller is still active'
    summary=json.loads((HERE/'artifacts/scientific/summary-001.json').read_text())
    assert len(summary)==15 and all(r['status']=='complete' for r in summary)
    target=HERE/'transfer';target.mkdir(exist_ok=True)
    archive=target/'output-001.tar.gz'
    assert not archive.exists(), 'Never overwrite a retrieval archive'
    paths=[p for folder in ['artifacts','provenance'] for p in (HERE/folder).rglob('*') if p.is_file()]
    paths += [p for p in (HERE/'runtime').iterdir() if p.is_file() and p.suffix in ['.json','.txt','.log','.exit']]
    paths += list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'config.json']
    paths=sorted(set(paths))
    inventory={'files':[record(p) for p in paths]}
    inventory_path=target/'inventory-001.json'
    inventory_path.write_text(json.dumps(inventory,indent=2)+'\n')
    with tarfile.open(archive,'w:gz') as bundle:
        for path in paths+[inventory_path]:
            bundle.add(path,arcname=path.relative_to(HERE).as_posix(),recursive=False)
    receipt=record(archive)
    (target/'receipt-001.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'archive':receipt,'files':len(paths),'payload_bytes':sum(r['bytes'] for r in inventory['files'])}))


if __name__=='__main__':main()
