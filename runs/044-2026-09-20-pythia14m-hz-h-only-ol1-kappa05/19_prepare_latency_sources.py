"""Copy and verify the unchanged Run041 K050 source archive."""
import hashlib
import json
import os
from pathlib import Path
import shutil

RUN = Path(__file__).resolve().parent
SOURCE = RUN.parent/'041-2026-09-20-pythia14m-hz-h-only-ol1/latency'
DEST = RUN/'latency'

def fs(path):
    return Path('\\\\?\\' + str(path.resolve())) if os.name == 'nt' else path

def main():
    archive = json.loads((SOURCE/'provenance/archive.json').read_text())
    for item in archive['files']:
        row = item['snapshot']
        source = fs(SOURCE/row['path'])
        payload = source.read_bytes()
        assert len(payload) == row['bytes']
        assert hashlib.sha256(payload).hexdigest() == row['sha256']
        target = fs(DEST/row['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == payload
        else:
            shutil.copyfile(source, target)
    print(json.dumps({'verified_archived_files':len(archive['files']),
                      'source':str(SOURCE), 'kernel_changes':0}))

if __name__ == '__main__':
    main()
