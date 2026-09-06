"""Install only input/provenance files from the fully verified original archive."""
import argparse
import hashlib
import tarfile
import shutil
from common import ROOT,RUN,R27,read_json,write_json,verify_record

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--receipt',required=True);a=p.parse_args()
    expected='044dd4115f1bd4066dba11ae4e1b8305e3e939f6376b38562186b1df95ef9f19'
    digest=hashlib.sha256()
    with open(a.archive,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=expected:raise ValueError('Whole archive hash mismatch')
    manifest=read_json(R27/'prelaunch/inputs.json')
    records=list(manifest['inputs'].values())
    for checkpoint in manifest['checkpoints']:records+=checkpoint['files']+checkpoint['provenance']
    selected={r['path']:r for r in records}
    missing=set(selected)
    with tarfile.open(a.archive,'r|gz') as archive:
        for entry in archive:
            if entry.name not in missing:continue
            row=selected[entry.name]
            target=(ROOT/entry.name).resolve()
            if not target.is_relative_to(ROOT.resolve()) or not entry.isfile() or entry.size!=row['bytes']:raise ValueError('Invalid member')
            try:verify_record(row)
            except (FileNotFoundError,ValueError):
                target.parent.mkdir(parents=True,exist_ok=True)
                temp=target.with_suffix(target.suffix+'.transport-partial')
                with archive.extractfile(entry) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst)
                from common import sha256
                if sha256(temp)!=row['sha256']:raise ValueError('Input member hash mismatch')
                temp.replace(target)
            verify_record(row);missing.remove(entry.name)
    if missing:raise ValueError('Input archive missing expected members')
    receipt={'status':'verified','archive_sha256':expected,'conditions':[r['id'] for r in manifest['checkpoints']],
             'verified_files':list(selected.values()),'source_code_extracted':False}
    write_json(RUN/'launch-control'/a.receipt,receipt)
    print({'status':'verified','conditions':len(receipt['conditions']),'files':len(selected)},flush=True)

if __name__=='__main__':main()
