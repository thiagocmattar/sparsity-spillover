"""Deploy corrected fixture using verified immutable inputs already on the Pod."""
import hashlib
import json
from pathlib import Path
import tarfile
import remote

RUN=Path(__file__).resolve().parent.parent

if __name__=='__main__':
    target=RUN/'bundles/correction-002.tar.gz'
    assert not target.exists()
    paths=list(RUN.glob('*.py'))+list(RUN.glob('*.sh'))+[RUN/'config.json']
    paths+=list((RUN/'candidate').glob('*.cu'))+list((RUN/'provenance').glob('*'))
    with tarfile.open(target,'w:gz') as tar:
        for p in sorted(paths):tar.add(p,arcname=p.relative_to(RUN).as_posix(),recursive=False)
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    c=remote.connect()
    try:
        with c.open_sftp() as sftp:sftp.put(str(target),'/workspace/run039-correction-002.tar.gz',callback=remote.progress())
        command="""python3 - <<'PY'
import hashlib,json,shutil,subprocess,tarfile,time
from pathlib import Path
old=Path('/workspace/run038');r=Path('/workspace/run039')
bundle=Path('/workspace/run039-correction-002.tar.gz')
assert hashlib.sha256(bundle.read_bytes()).hexdigest()=='DIGEST'
assert not r.exists();r.mkdir()
for name in ['archive','inputs']:shutil.copytree(old/name,r/name)
with tarfile.open(bundle) as tar:tar.extractall(r,filter='data')
(r/'runtime').mkdir()
for name in ['pip-freeze.txt','nvcc.txt','nvidia-smi-initial.txt']:shutil.copyfile(old/'runtime'/name,r/'runtime'/name)
subprocess.run(['/opt/run038-venv/bin/python',str(r/'01_prepare.py'),'verify'],check=True)
record={'epoch':time.time(),'reuse':'Run038 frozen source/input trees and pinned /opt/run038-venv','correction_sha256':'DIGEST','original_deadline_retained':True}
(r/'runtime/setup-reuse.json').write_text(json.dumps(record,indent=2)+'\\n')
(r/'runtime/setup-complete').touch();(r/'runtime/setup-001.exit').write_text('0\\n')
print(json.dumps(record))
PY""".replace('DIGEST',digest)
        output=remote.execute(c,command)
        print(output)
        (RUN/'prelaunch/deploy-002.txt').write_text(output)
    finally:c.close()
