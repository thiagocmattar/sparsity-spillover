"""Reconcile every remote attempt artifact with its locally retained bytes."""
import hashlib,json,shlex
from pathlib import Path
from transport import connect,execute
run=Path(__file__).resolve().parent.parent
code='''import pathlib,hashlib,json
r=pathlib.Path('/workspace/run051');root=r/'artifacts'
rows=[{'path':p.relative_to(r).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.rglob('*')) if p.is_file()]
print(json.dumps({'files':rows,'attempt_directories':[p.name for p in sorted(root.iterdir()) if p.is_dir()],'large_root_files':[{'path':p.name,'bytes':p.stat().st_size} for p in r.iterdir() if p.is_file() and p.stat().st_size>4*1024**2]}))
'''
c=connect()
try:manifest=json.loads(execute(c,'python3 -c '+shlex.quote(code),timeout=180))
finally:c.close()
missing=[];different=[]
for row in manifest['files']:
 path=run/row['path']
 if not path.exists():missing.append(row['path'])
 elif path.stat().st_size!=row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:different.append(row['path'])
out=run/'retrieval';out.mkdir(exist_ok=True)
(out/'all-artifacts-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
summary={'remote_files':len(manifest['files']),'remote_bytes':sum(r['bytes'] for r in manifest['files']),
         'attempt_directories':manifest['attempt_directories'],'missing':missing,'different':different,'large_root_files':manifest['large_root_files'],
         'manifest_sha256':hashlib.sha256((out/'all-artifacts-manifest.json').read_bytes()).hexdigest(),
         'scope':'Every regular file in /workspace/run051/artifacts, including failed and incomplete attempts. Checkpoints/cache identities are retained unchanged through Run049. Execution-control logs are retained under prelaunch.'}
(run/'results/retention-audit.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2));assert not missing and not different
