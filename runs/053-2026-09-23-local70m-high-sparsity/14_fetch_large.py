"""Retrieve terminal attempts without bidirectional tar pipe backpressure."""
import argparse
import json
import subprocess
import tarfile
from local_support import RUN, load, sha, write

LINUX = '/home/researcher/sparsity-spillover/' + RUN.name
PYTHON = '/opt/sparsity-gpu/venv/bin/python'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    assert args.tag.replace('-', '').isalnum()
    assert load(RUN/'prelaunch'/('worker-'+args.tag+'.json'))['status'] in ('complete', 'failed')
    code = '''import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);tag=sys.argv[2]
rows=[]
for folder in sorted((root/'artifacts/attempts').glob(tag+'-*')):
 for path in sorted(folder.rglob('*')):
  if not path.is_file(): continue
  assert not path.is_symlink()
  path.resolve().relative_to(root.resolve())
  with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
  rows.append({'path':path.relative_to(root).as_posix(),'bytes':path.stat().st_size,'sha256':digest})
print(json.dumps({'files':rows}))
'''
    response = subprocess.run(['wsl.exe','-d','SparsityGPU','--exec',PYTHON,'-',LINUX,args.tag],
                              input=code.encode(),capture_output=True,check=True)
    inventory = json.loads(response.stdout)
    assert inventory['files']
    # Only the five short directory arguments cross the process boundary.
    # The old file-list stdin writer could block before stdout was drained.
    folders = sorted({'/'.join(row['path'].split('/')[:3]) for row in inventory['files']})
    process = subprocess.Popen(['wsl.exe','-d','SparsityGPU','--exec','tar','-cf','-','-C',LINUX,'--',*folders],
                               stdin=subprocess.DEVNULL,stdout=subprocess.PIPE)
    with tarfile.open(fileobj=process.stdout,mode='r|') as archive:
        archive.extractall(RUN,filter='data')
    assert process.wait() == 0
    for row in inventory['files']:
        path = RUN/row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],row['path']
    write(RUN/'results'/(args.tag+'-retrieval.json'),inventory)
    print(json.dumps({'verified_files':len(inventory['files']),'bytes':sum(r['bytes'] for r in inventory['files'])}))

if __name__ == '__main__': main()
