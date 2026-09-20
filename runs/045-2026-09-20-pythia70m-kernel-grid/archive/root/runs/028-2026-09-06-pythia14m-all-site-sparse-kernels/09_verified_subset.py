"""Recover only named, hash-verified inputs from an incomplete transport archive.

This never marks the archive complete. No source code is extracted. It allows
development on already received checkpoint files while a slow transfer finishes.
"""
import argparse
import shutil
import tarfile
from common import RUN, ROOT, R27, read_json, write_json, verify_record


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',required=True)
    p.add_argument('--conditions',nargs='+',required=True)
    p.add_argument('--receipt',required=True)
    a=p.parse_args()
    manifest=read_json(R27/'prelaunch/inputs.json')
    checkpoints=[next(r for r in manifest['checkpoints'] if r['id']==cid) for cid in a.conditions]
    records=list(manifest['inputs'].values())
    for c in checkpoints:records+=c['files']+c['provenance']
    expected={r['path']:r for r in records}
    pending=set(expected)
    for name in list(pending):
        try:verify_record(expected[name]);pending.remove(name)
        except (ValueError,FileNotFoundError):pass
    extraction_error=None
    try:
        with tarfile.open(a.archive,'r|gz') as archive:
            for entry in archive:
                if entry.name not in pending:continue
                row=expected[entry.name]
                if not entry.isfile() or entry.size!=row['bytes']:raise ValueError('Unexpected archive member')
                target=(ROOT/entry.name).resolve()
                if not target.is_relative_to(ROOT.resolve()):raise ValueError('Escaping archive path')
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_suffix(target.suffix+'.transport-partial')
                with archive.extractfile(entry) as src,temporary.open('wb') as dst:shutil.copyfileobj(src,dst)
                # Verify bytes before installing any input at its canonical path.
                from common import sha256
                if temporary.stat().st_size!=row['bytes'] or sha256(temporary)!=row['sha256']:
                    raise ValueError('Incomplete/corrupt selected member')
                temporary.replace(target)
                verify_record(row);pending.remove(entry.name)
                if not pending:break
    except (EOFError,tarfile.ReadError,OSError) as exc:
        extraction_error=str(exc)
    verified=[]
    for checkpoint in checkpoints:
        try:
            for row in list(manifest['inputs'].values())+checkpoint['files']+checkpoint['provenance']:verify_record(row)
            verified.append(checkpoint['id'])
        except (ValueError,FileNotFoundError):pass
    receipt={'archive_complete_claimed':False,'requested':a.conditions,'verified_conditions':verified,
        'verified_files':[expected[name] for name in expected if name not in pending],
        'missing_files':sorted(pending),'extraction_error':extraction_error}
    write_json(RUN/'launch-control'/a.receipt,receipt)
    print({'verified_conditions':verified,'missing_files':len(pending),'extraction_error':extraction_error})


if __name__=='__main__':main()
