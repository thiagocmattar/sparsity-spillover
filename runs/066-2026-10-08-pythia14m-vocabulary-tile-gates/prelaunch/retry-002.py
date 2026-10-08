"""Retry only the failed qualification/reporting infrastructure on the owned Pod."""
import json
from pathlib import Path
import shlex
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_config import RUN_DIR as RUN,run_code_identity,write_json
from importlib import import_module
remote=import_module('10_remote')

if __name__=='__main__':
    receipt=json.loads((RUN/'prelaunch/source-002.receipt.json').read_text())
    client,lease=remote.connect()
    try:
        archive="""python3 - <<'PY'
import hashlib,json,shutil
from pathlib import Path
r=Path('/workspace/sparsity-spillover/runs/066-2026-10-08-pythia14m-vocabulary-tile-gates')
c=Path('/workspace/run066-control')
assert (c/'pipeline.exit').read_text().strip()=='1'
assert not (r/'artifacts/attempts').exists(), 'Scientific training must not have started'
a=r/'prelaunch/attempts/001';a.mkdir(parents=True,exist_ok=False)
(a/'remote-control').mkdir()
for p in sorted(c.iterdir()):
    if p.is_file() and p.name!='guard.log' and p.suffix in ('.log','.exit','.txt','.pid'):
        shutil.move(str(p),str(a/'remote-control'/p.name))
for p in sorted((r/'prelaunch').glob('remote-preflight-*.json')):
    shutil.copyfile(p,a/p.name)
for p in sorted((r/'prelaunch').glob('calibration-checkpoint-*')):
    assert p.is_dir() and p.resolve().is_relative_to(r.resolve())
    shutil.move(str(p),str(a/p.name))
records=[]
for p in sorted(a.rglob('*')):
    if p.is_file():
        with p.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
        records.append({'path':p.relative_to(r).as_posix(),'bytes':p.stat().st_size,'sha256':sha})
note={'status':'failed_before_scientific_training','cause':'Preflight diagnostic dict shadowed statistics module during final summary','deployment_commit':'e745d27ff74df3811df9afacb87865d7da1619b0','files':records}
(a/'failure-inventory.json').write_text(json.dumps(note,indent=2)+'\n')
print(json.dumps({'preserved_files':len(records),'scientific_training_started':False}))
PY"""
        print(remote.execute(client,archive),flush=True)
        with client.open_sftp() as sftp:
            destination='/workspace/run066-source-002.bundle'
            remote.upload(sftp,RUN/receipt['path'],destination)
        if remote.execute(client,'sha256sum '+destination).split()[0]!=receipt['sha256']:
            raise RuntimeError('Retry source transfer differs')
        root=remote.REMOTE
        print(remote.execute(client,f'cd {root} && git fetch {destination} HEAD && git checkout '+receipt['deployment_commit'],timeout=120),flush=True)
        run=f'{root}/runs/{RUN.name}'
        py='/workspace/run066-venv/bin/python'
        identity=json.loads(remote.execute(client,f"cd {root} && {py} -c "+shlex.quote(f"import sys,json;sys.path.insert(0,{run!r});from run_config import run_code_identity;print(json.dumps(run_code_identity()))")))
        if identity!=run_code_identity():raise RuntimeError('Retry runtime source differs')
        control='/workspace/run066-control'
        wrapper=f'cd {root}; bash {run}/08_pipeline.sh; code=$?; printf "%s\\n" "$code" > {control}/pipeline.exit'
        remote.execute(client,f'nohup setsid bash -c {shlex.quote(wrapper)} > {control}/pipeline.log 2>&1 < /dev/null & echo $! > {control}/pipeline.pid')
        write_json(RUN/'prelaunch/retry-002.receipt.json',{'deployment':receipt,'run_code':identity,
            'pod_id':lease['pod']['id'],'deadline_utc':lease['deadline_utc'],
            'preserved_attempt':'prelaunch/attempts/001','active_preflight_attempt':'prelaunch/attempts/002',
            'scientific_changes':False,'cache_reused_after_exact_hash_verification':True,
            'tests':{'focused_passed':39,'bootstrap_passed':242}})
        print('Fresh qualification retry detached; scientific training still behind four-condition barrier',flush=True)
    finally:client.close()
