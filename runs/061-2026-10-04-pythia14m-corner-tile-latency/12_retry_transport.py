"""Append-only module-resolution retry on the original owned Pod and deadline."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import tarfile
from io_utils import RUN, module, read, record, sha, write

transport = module('run061_transport', RUN / '06_transfer.py')
transport.CONTROL = '/workspace/run061-control-002'
REMOTE, CONTROL = transport.REMOTE, transport.CONTROL


def start(client, lease):
    transport.execute(client, f'test ! -e {CONTROL} && mkdir {CONTROL}')
    remote_boot = REMOTE + '/infrastructure/retry-002'
    transport.execute(client, f'mkdir -p {remote_boot}')
    with client.open_sftp() as sftp:
        for source, target in [(RUN / '08_deadline_guard.py', CONTROL + '/guard.py'),
                               (RUN / 'infrastructure/retry-002/sitecustomize.py', remote_boot + '/sitecustomize.py')]:
            sftp.put(str(source), target)
            if transport.execute(client, f'sha256sum {shlex.quote(target)}').split()[0] != sha(source):
                raise RuntimeError('Retry transport hash mismatch')
        with sftp.open(CONTROL + '/guard-settings.json', 'w') as handle:
            handle.write(json.dumps({'pod_id': lease['pod']['id'], 'name': lease['pod']['name'],
                                    'deadline_epoch': lease['deadline_epoch'], 'control': CONTROL}))
    transport.execute(client, f'nohup python3 -u {CONTROL}/guard.py {CONTROL}/guard-settings.json > {CONTROL}/guard.log 2>&1 < /dev/null &')
    wrapper = (f'cd {REMOTE}; export PYTHONPATH={shlex.quote(remote_boot)}; '
               f'bash {REMOTE}/05_execute.sh smoke {lease["deadline_epoch"]} 002 || exit $?; '
               f'bash {REMOTE}/05_execute.sh scientific {lease["deadline_epoch"]} 002')
    wrapper = f'({wrapper}); code=$?; printf "%s\\n" "$code" > {CONTROL}/pipeline.exit'
    transport.execute(client, f'nohup setsid bash -c {shlex.quote(wrapper)} > {CONTROL}/pipeline.log 2>&1 < /dev/null & echo $! > {CONTROL}/pipeline.pid')
    write(RUN / 'prelaunch/retry-002.json', {'status': 'launched', 'pod_id': lease['pod']['id'],
          'deadline_epoch': lease['deadline_epoch'], 'source': record(RUN / 'infrastructure/retry-002/sitecustomize.py'),
          'reason': 'legacy diagnostics import shadowed the sealed Run061 module',
          'scientific_sources_changed': False, 'new_controller_tag': '002',
          'failed_attempt_preserved': 'scientific-k0p1-r1-001'})
    print('Retry002 launched with unchanged scientific sources and original deadline')


def retrieve(client):
    command = f"""python3 - <<'PY'
from pathlib import Path
import hashlib,json,shutil,tarfile
run=Path('{REMOTE}');control=Path('{CONTROL}')
if not (control/'pipeline.exit').exists():raise RuntimeError('Retry pipeline still active')
out=run/'artifacts/control'
for name in ['pipeline.log','pipeline.exit','pipeline.pid','guard.log']:
    if (control/name).exists():
        target=out/(Path(name).stem+'-002'+Path(name).suffix)
        if target.exists():raise RuntimeError('Control evidence already retrieved')
        shutil.copyfile(control/name,target)
paths=sorted(p for p in (run/'artifacts').rglob('*') if p.is_file())
paths += [p for p in (run/'runtime').iterdir() if p.is_file() and p.suffix in ('.json','.txt')]
rows=[{{'path':p.relative_to(run).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()}} for p in paths]
inventory={{'files':rows,'payload_bytes':sum(r['bytes'] for r in rows)}}
inventory_path=run/'transfer/inventory-002.json'
if inventory_path.exists():raise RuntimeError('Inventory already exists')
inventory_path.write_text(json.dumps(inventory,indent=2)+'\\n')
archive=Path('/workspace/run061-output-002.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    for p in paths+[inventory_path]:tar.add(p,arcname=p.relative_to(run).as_posix(),recursive=False)
print(json.dumps({{'bytes':archive.stat().st_size,'sha256':hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()}}))
PY"""
    receipt = json.loads(transport.execute(client, command, timeout=120))
    target = RUN / 'retrieval/output-002.tar.gz'
    with client.open_sftp() as sftp:
        sftp.get('/workspace/run061-output-002.tar.gz', str(target))
    if target.stat().st_size != receipt['bytes'] or sha(target) != receipt['sha256']:
        raise RuntimeError('Retry archive hash mismatch')
    with tarfile.open(target) as archive:
        for member in archive:
            path = (RUN / member.name).resolve()
            if not path.is_relative_to(RUN) or not member.isfile():
                raise RuntimeError('Unsafe member')
            with archive.extractfile(member) as source:
                payload = source.read()
            if path.exists():
                if path.read_bytes() != payload:
                    raise RuntimeError(f'Existing evidence changed: {member.name}')
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
    inventory = read(RUN / 'transfer/inventory-002.json')
    for row in inventory['files']:
        if record(RUN / row['path']) != row:
            raise RuntimeError('Retry member hash mismatch')
    write(RUN / 'transfer/receipt-002.json', {'archive': receipt,
          'verified_files': len(inventory['files']), 'payload_bytes': inventory['payload_bytes'], 'status': 'verified'})
    print(json.dumps({'verified_files': len(inventory['files']), 'payload_bytes': inventory['payload_bytes']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'status', 'retrieve'])
    args = parser.parse_args()
    client, lease = transport.connect()
    try:
        if args.action == 'start': start(client, lease)
        elif args.action == 'status': transport.status(client)
        else: retrieve(client)
    finally:
        client.close()
