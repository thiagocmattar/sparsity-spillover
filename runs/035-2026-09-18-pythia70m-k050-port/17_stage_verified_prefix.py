"""Accept only hash-complete sentinel files from an in-progress input archive.

This enables untimed qualification preparation while the remaining checkpoints
transfer. Full archive and cohort verification still precede the final grid.
"""
import hashlib
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent

def main():
    rows=json.loads((HERE/'provenance/inputs.json').read_text())['checkpoints']
    selected=[r for r in rows if r['id'] in {'c00','c03','c06','c11'}]
    with tarfile.open(HERE/'relay-bundle-001/input-001.tar') as bundle:
        members={m.name:m for m in bundle}
        accepted=[]
        for row in selected:
            complete=True
            for f in row['files']:
                m=members.get(f['path'])
                if not m or not m.isfile() or m.size!=f['bytes']:complete=False;continue
                content=bundle.extractfile(m).read()
                if hashlib.sha256(content).hexdigest()!=f['sha256']:complete=False;continue
                target=HERE/f['path'];assert target.resolve().is_relative_to(HERE/'inputs/checkpoints')
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
            if complete:accepted.append(row['id'])
    report={'hash_complete_sentinels':accepted,'full_archive_verification_still_required':True}
    (HERE/'runtime/verified-prefix.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
