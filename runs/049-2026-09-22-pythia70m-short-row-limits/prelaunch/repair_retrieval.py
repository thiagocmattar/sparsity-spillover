"""Repair only empty optional-control imports created by the first retriever."""
import hashlib
import json
from pathlib import Path
import shutil

RUN=Path(__file__).resolve().parent.parent
changes=[]
for name in ('local-runtime-cache.json','local-archive-cache.json'):
    local=RUN/'provenance'/name
    if not local.exists() or local.stat().st_size!=0:continue
    saved=RUN/'retrieval/001/empty-control-imports'/name
    saved.parent.mkdir(exist_ok=True);shutil.copy2(local,saved)
    original=RUN/'retrieval/001/extracted/provenance'/name
    latest=RUN/'retrieval/002/extracted/provenance'/name
    if original.exists() and latest.exists():
        assert original.read_bytes()==latest.read_bytes()
        shutil.copy2(latest,local)
        changes.append({'path':str(local.relative_to(RUN)),'action':'restored identical inventoried bytes','sha256':hashlib.sha256(latest.read_bytes()).hexdigest()})
    else:
        assert not original.exists() and not latest.exists()
        local.unlink()
        changes.append({'path':str(local.relative_to(RUN)),'action':'removed spurious empty optional file; preserved in retrieval'})
(RUN/'prelaunch/retrieval-repair-001.json').write_text(json.dumps({'reason':'Missing optional SFTP reads had created empty local files; canonical provenance is now never overwritten by optional control snapshots','changes':changes},indent=2)+'\n')
print(changes)
