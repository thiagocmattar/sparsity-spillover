"""Retrieve and verify tag002 evidence after the owned pipeline exits."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

RUN=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("run063_transport",RUN/"infrastructure/retry-001/transport.py")
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
client,lease=transport.connect()
try:
    code="""from pathlib import Path
import hashlib,json,shutil,tarfile
run=Path('/workspace/run063');control=Path('/workspace/run063-control')
assert (control/'pipeline.exit').exists(), 'Pipeline still active'
out=run/'artifacts/control';out.mkdir(exist_ok=True)
for name in ['pipeline.log','pipeline.exit','pipeline.pid','guard.log']:
    if (control/name).exists():shutil.copyfile(control/name,out/name)
paths=sorted(p for p in (run/'artifacts').rglob('*') if p.is_file())
paths += [p for p in (run/'runtime').iterdir() if p.is_file() and p.suffix in ('.json','.txt')]
rows=[{'path':p.relative_to(run).as_posix(),'bytes':p.stat().st_size,
    'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p in paths]
inventory={'files':rows,'payload_bytes':sum(r['bytes'] for r in rows)}
(run/'transfer').mkdir(exist_ok=True)
(run/'transfer/inventory-002.json').write_text(json.dumps(inventory,indent=2)+'\\n')
archive=Path('/workspace/run063-output-002.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    for p in paths+[run/'transfer/inventory-002.json']:
        tar.add(p,arcname=p.relative_to(run).as_posix(),recursive=False)
print(json.dumps({'bytes':archive.stat().st_size,'sha256':hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()}))
"""
    receipt=json.loads(transport.original.execute(client,"python3 -c "+transport.original.shlex.quote(code),timeout=120))
    target=RUN/"retrieval/output-002.tar.gz";target.parent.mkdir(exist_ok=True)
    with client.open_sftp() as sftp:sftp.get("/workspace/run063-output-002.tar.gz",str(target))
    if target.stat().st_size!=receipt["bytes"] or transport.original.sha(target)!=receipt["sha256"]:
        raise RuntimeError("Returned archive hash mismatch")
    with tarfile.open(target) as archive:
        for member in archive:
            path=(RUN/member.name).resolve()
            if not path.is_relative_to(RUN) or not member.isfile():
                raise RuntimeError("Unsafe archive member")
            path.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as source,path.open("wb") as destination:
                shutil.copyfileobj(source,destination)
    inventory=transport.original.read(RUN/"transfer/inventory-002.json")
    for row in inventory["files"]:
        if transport.original.record(RUN/row["path"])!=row:
            raise RuntimeError("Returned member hash mismatch")
    transport.original.write(RUN/"transfer/receipt-002.json",{
        "archive":receipt,"verified_files":len(inventory["files"]),
        "payload_bytes":inventory["payload_bytes"],"status":"verified",
        "source_script":transport.original.record(__file__)})
    print(json.dumps({"verified_files":len(inventory["files"]),
        "payload_bytes":inventory["payload_bytes"],"archive_bytes":receipt["bytes"]}),flush=True)
finally:client.close()
