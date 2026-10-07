"""Bundle exact prior input members and this run's independent instrumentation."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).parent;BASE=ROOT.parent/'063-2026-10-07-pythia14m-tile-specialized-kernel'
dest=ROOT/'bundles/input-001.tar.gz';dest.parent.mkdir(exist_ok=True)
if dest.exists():raise ValueError('Fresh bundle required')
rows=[]
with tarfile.open(dest,'w:gz',compresslevel=3) as out:
    with tarfile.open(BASE/'bundles/input-002.tar') as source:
        for member in source:
            if not member.isfile():raise ValueError('Regular frozen files required')
            data=source.extractfile(member).read();name='dependencies/run063/'+member.name
            item=tarfile.TarInfo(name);item.size=len(data);out.addfile(item,io.BytesIO(data))
            rows.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    paths=sorted(list(ROOT.glob('*.py'))+list(ROOT.glob('*.cu'))+list(ROOT.glob('*.sh'))+[ROOT/'config.json'])
    for path in paths:
        data=path.read_bytes();name=path.name;item=tarfile.TarInfo(name);item.size=len(data);out.addfile(item,io.BytesIO(data))
        rows.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    data=json.dumps({'files':rows},indent=2).encode();item=tarfile.TarInfo('prelaunch/bundle-inventory.json');item.size=len(data);out.addfile(item,io.BytesIO(data))
(ROOT/'prelaunch').mkdir(exist_ok=True)
receipt={'bytes':dest.stat().st_size,'sha256':hashlib.file_digest(dest.open('rb'),'sha256').hexdigest(),'files':len(rows),'payload_bytes':sum(r['bytes'] for r in rows)}
(ROOT/'prelaunch/bundle-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
