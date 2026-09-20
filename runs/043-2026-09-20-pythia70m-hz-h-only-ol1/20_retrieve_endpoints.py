"""Retrieve verified final inputs first so latency overlaps the full archive copy."""
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import tarfile
import time

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('run043_remote',RUN/'10_remote.py')
remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)


def digest(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    client,_=remote.connect('training')
    root=remote.REMOTE+'/runs/'+RUN.name
    staging=RUN/'retrieval/endpoints-001';staging.mkdir(parents=True,exist_ok=True)
    prepare=f'''python3 - <<'PY'
import hashlib,json,tarfile
from pathlib import Path
root=Path({root!r})
assert Path('/workspace/run043-control/training-verified').exists()
cohort=json.loads((root/'artifacts/verification.json').read_text())
assert cohort['status']=='verified' and cohort['condition_count']==4
paths=[root/'artifacts/verification.json']
for row in cohort['conditions']:
 attempt=root/'artifacts/attempts'/row['attempt_id']
 paths += [attempt/'checkpoints/step_000712'/name for name in ['config.json','model.safetensors','checkpoint_metadata.json','generation_config.json']]
 paths += [attempt/name for name in ['manifest.json','config.yaml','diagnostics/logical_products.json']]
def record(p):
 with p.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
 return dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest)
folder=root/'transfer';folder.mkdir(exist_ok=True)
inventory=folder/'endpoints-inventory-001.json'
archive=folder/'endpoints-001.tar'
if not archive.exists():
 inventory.write_text(json.dumps(dict(files=[record(p) for p in paths]),indent=2)+'\\n')
 with tarfile.open(archive,'w') as bundle:
  for p in paths+[inventory]:bundle.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
print(json.dumps(record(archive)))
PY'''
    try:
        receipt=json.loads(remote.execute(client,prepare,timeout=120))
        archive=staging/'endpoints-001.tar'
        if not archive.exists():
            part=archive.with_suffix('.tar.part')
            offset=part.stat().st_size if part.exists() else 0
            assert offset<=receipt['bytes']
            transport=client.get_transport()
            transport.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            transport.sock.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,8*1024**2)
            channel=transport.open_session(window_size=64*1024**2,max_packet_size=1024**2)
            channel.settimeout(120)
            channel.exec_command(f'tail -c +{offset+1} {root}/{receipt["path"]}')
            start=time.monotonic();last=start;done=offset
            with part.open('ab') as handle:
                while chunk:=channel.recv(1024**2):
                    handle.write(chunk);done+=len(chunk)
                    if time.monotonic()-last>=30:
                        print(json.dumps({'bytes':done,'total':receipt['bytes']}),flush=True);last=time.monotonic()
            assert channel.recv_exit_status()==0;channel.close()
            assert part.stat().st_size==receipt['bytes'] and digest(part)==receipt['sha256']
            part.replace(archive)
        assert archive.stat().st_size==receipt['bytes'] and digest(archive)==receipt['sha256']
    finally:client.close()
    with tarfile.open(archive) as bundle:
        inventory=json.load(bundle.extractfile('transfer/endpoints-inventory-001.json'))
        rows={r['path']:r for r in inventory['files']}
        assert {m.name for m in bundle.getmembers()}==set(rows)|{'transfer/endpoints-inventory-001.json'}
        for member in bundle.getmembers():
            target=RUN/member.name
            assert member.isfile() and target.resolve().is_relative_to(RUN)
            content=bundle.extractfile(member).read()
            if member.name in rows:
                row=rows[member.name]
                assert len(content)==row['bytes'] and hashlib.sha256(content).hexdigest()==row['sha256']
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert target.read_bytes()==content
            else:target.write_bytes(content)
    report=dict(archive=receipt,files=len(rows),archive_and_members_verified=True,
                full_training_retrieval_still_required=True)
    (RUN/'prelaunch/retrieval-endpoints-001.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
