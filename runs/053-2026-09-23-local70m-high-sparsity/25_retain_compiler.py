"""Retain compiler additions and local extension binaries after a terminal stage."""
import argparse
import subprocess
from local_support import RUN, load

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);a=parser.parse_args()
    assert a.tag.replace('-','').isalnum()
    assert load(RUN/'prelaunch'/('worker-'+a.tag+'.json'))['status'] in ('complete','failed')
    code='''import hashlib,json,pathlib,shutil,sys
root=pathlib.Path(sys.argv[1]);tag=sys.argv[2]
base=root/'artifacts/attempts/wider-outputs-001-compiler-cache/inventory.json'
prior={row['path']:row['sha256'] for row in json.loads(base.read_text())['files']}
dst=root/'artifacts/attempts'/(tag+'-compiler-delta');assert not dst.exists()
rows=[];unchanged=0
for name,src in [('triton',pathlib.Path('/opt/sparsity-gpu/triton-cache')),('extensions',pathlib.Path('/opt/sparsity-gpu/extensions'))]:
 assert src.is_dir()
 for path in sorted(src.rglob('*')):
  if path.is_symlink() or not path.is_file():continue
  path.resolve().relative_to(src.resolve())
  rel=path.relative_to(src).as_posix()
  with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  if name=='triton' and prior.get(rel)==digest:
   unchanged+=1;continue
  target=dst/name/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
  rows.append({'path':target.relative_to(dst).as_posix(),'bytes':target.stat().st_size,'sha256':digest})
dst.mkdir(parents=True,exist_ok=True)
with base.open('rb') as f:base_sha=hashlib.file_digest(f,'sha256').hexdigest()
(dst/'inventory.json').write_text(json.dumps({'scope':'Triton additions/changes since verified wider-output snapshot; complete local C++ extension cache. Includes controls and test kernels.', 'base_inventory':base.relative_to(root).as_posix(),'base_inventory_sha256':base_sha,'unchanged_triton_files':unchanged,'files':rows},indent=2)+'\\n')
print(json.dumps({'retained_files':len(rows),'bytes':sum(r['bytes'] for r in rows),'unchanged_triton_files':unchanged}))
'''
    linux='/home/researcher/sparsity-spillover/'+RUN.name
    process=subprocess.run(['wsl.exe','-d','SparsityGPU','--exec','/opt/sparsity-gpu/venv/bin/python','-',linux,a.tag],
                           input=code.encode(),capture_output=True,check=True)
    print(process.stdout.decode())

if __name__=='__main__':main()
