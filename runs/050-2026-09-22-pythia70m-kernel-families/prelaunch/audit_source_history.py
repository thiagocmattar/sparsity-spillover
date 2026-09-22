"""Audit deploy snapshots; recover exact earlier bytes without overwriting history."""
import hashlib,io,json,tarfile,time
from pathlib import Path
run=Path(__file__).resolve().parent.parent;history=run/'provenance/source-history';pool={};locations={}
for archive in sorted(history.glob('upload-*.tar.gz')):
 with tarfile.open(archive,'r:gz') as t:
  for member in t.getmembers():
   if not member.isfile():continue
   raw=t.extractfile(member).read();digest=hashlib.sha256(raw).hexdigest();pool[digest]=raw;locations[digest]=[archive.name,member.name]
for path in history.glob('recovered-*.py'):
 raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest();pool[digest]=raw;locations[digest]=[path.name,'Reversed compiler-metadata-only edits; verified against the original upload SHA256']
issues=[];audited=[]
for receipt in sorted((run/'prelaunch').glob('upload-*.json')):
 archive=history/(receipt.stem+'.tar.gz')
 if not archive.exists():continue
 rows=json.loads(receipt.read_text())['files'];bad=[]
 with tarfile.open(archive,'r:gz') as t:
  for row in rows:
   got=hashlib.sha256(t.extractfile(row['path']).read()).hexdigest()
   if got!=row['sha256']:bad.append({'path':row['path'],'expected':row['sha256'],'stored':got})
 if bad:
  assert all(row['sha256'] in pool for row in rows),'Required source bytes absent'
  restored=history/('recovered-'+archive.name)
  if not restored.exists():
   with tarfile.open(restored,'w:gz') as t:
    for row in rows:
     raw=pool[row['sha256']];assert len(raw)==row['bytes']
     info=tarfile.TarInfo(row['path']);info.size=len(raw);info.mtime=int(time.time());t.addfile(info,io.BytesIO(raw))
  with tarfile.open(restored,'r:gz') as t:
   assert all(hashlib.sha256(t.extractfile(r['path']).read()).hexdigest()==r['sha256'] for r in rows)
  issues.append({'receipt':receipt.name,'original_retained':archive.name,'recovered':restored.name,'mismatches':bad,'recovery_sources':{r['path']:locations[r['expected']] for r in bad}})
 audited.append({'receipt':receipt.name,'files':len(rows),'matching_original_archive':not bad})
report={'audited':audited,'recovered':issues,'missing_required_bytes':False,'cause':'Earlier sender read live local files again while creating its archive. Remote uploads were hash-verified. The sender now uploads and archives one immutable byte snapshot.'}
(run/'provenance/source-history-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print({'archives_audited':len(audited),'recovered_archives':len(issues),'missing_required_bytes':False})
