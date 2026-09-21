"""Infrastructure only: copy exact-version, RECORD-verified CUDA distributions."""
import base64
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil

RUN = Path('/workspace/run048')
destination = Path('/tmp/run048-local-runtime/venv/lib/python3.12/site-packages')
assert destination.is_dir()
pins = dict(line.strip().split('==') for line in (RUN/'provenance/pip-freeze.txt').read_text().splitlines() if '==' in line)
rows = []
for name, version in pins.items():
    if not name.startswith('nvidia-'):
        continue
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        continue
    if dist.version != version:
        continue
    base = Path(dist.locate_file('')).resolve()
    assert base == Path('/usr/local/lib/python3.12/dist-packages')
    for item in dist.files or []:
        source = Path(dist.locate_file(item)).resolve()
        target = (destination/str(item)).resolve()
        assert source.is_relative_to(base) and target.is_relative_to(destination)
        if not source.is_file():
            continue
        if item.hash:
            h = hashlib.new(item.hash.mode)
            with source.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024**2), b''):
                    h.update(block)
            assert base64.urlsafe_b64encode(h.digest()).decode().rstrip('=') == item.hash.value, source
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists(), target
        shutil.copy2(source, target)
        h = hashlib.sha256()
        with target.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2), b''):
                h.update(block)
        rows.append({'distribution':name,'version':version,'path':str(item),
                     'bytes':target.stat().st_size,'sha256':h.hexdigest(),
                     'record_hash_verified':bool(item.hash)})
    print('Copied exact CUDA distribution',name,version,flush=True)
result = {'status':'verified','files':rows,'bytes':sum(r['bytes'] for r in rows),
          'source':'Pinned base image; matching distribution versions and RECORD hashes'}
(RUN/'provenance/image-cuda-reuse.json').write_text(json.dumps(result,indent=2)+'\n')
print('Verified/copy complete',len(rows),result['bytes'],flush=True)
