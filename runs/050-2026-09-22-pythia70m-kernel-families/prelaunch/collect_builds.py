"""Retain this run's compiled CUDA extensions and their resource reports."""
import shlex
from transport import connect,execute
code='''import pathlib,shutil,subprocess,hashlib,json
out=pathlib.Path('/workspace/run050/artifacts/builds-001');out.mkdir(exist_ok=False)
root=pathlib.Path('/workspace/run049/runtime/extensions');rows=[]
for folder in sorted(root.glob('r050_*')):
 if not list(folder.glob('*.so')):continue
 dest=out/folder.name;dest.mkdir()
 for src in sorted(folder.iterdir()):
  if src.suffix!='.so' and src.name!='build.ninja':continue
  target=dest/src.name;shutil.copyfile(src,target)
  rows.append({'source':str(src),'path':target.relative_to(out).as_posix(),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size})
  if src.suffix=='.so':
   result=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(src)],capture_output=True,text=True)
   (dest/(src.stem+'-resources.txt')).write_text(result.stdout+result.stderr)
   assert result.returncode==0
(out/'result.json').write_text(json.dumps({'status':'complete','files':rows,'purpose':'Retained compiled extension images and cuobjdump register/local-memory reports; no kernel execution'},indent=2)+'\\n')
print(json.dumps({'files':len(rows),'bytes':sum(r['bytes'] for r in rows)}))
'''
c=connect()
try:print(execute(c,'python3 -c '+shlex.quote(code),timeout=120))
finally:c.close()
