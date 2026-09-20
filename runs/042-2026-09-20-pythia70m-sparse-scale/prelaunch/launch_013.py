"""Apply hash-inventoried pre-execution overlay and launch persistent tests."""
import hashlib
import json
from pathlib import Path
import remote

RUN=Path(__file__).resolve().parent.parent

def main():
    assert json.loads((RUN/'prelaunch/upload-002.json').read_text())['verified']
    bundle=RUN/'bundles/overlay-013.tar.gz'
    digest=hashlib.sha256(bundle.read_bytes()).hexdigest()
    client=remote.connect()
    sftp=client.open_sftp()
    sftp.put(str(bundle),'/workspace/run042-overlay-013.tar.gz')
    command="echo '"+digest+"  /workspace/run042-overlay-013.tar.gz' | sha256sum -c - && "
    command+='tar -xzf /workspace/run042-overlay-013.tar.gz -C /workspace/run042'
    print(remote.execute(client,command),flush=True)
    check="""cd /workspace/run042 && python3 - <<'PY'
import json,hashlib
from pathlib import Path
for row in json.loads(Path('prelaunch/overlay-013-inventory.json').read_text())['files']:
 p=Path(row['path']); assert p.stat().st_size==row['bytes']; assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
print('Overlay hashes verified')
PY"""
    print(remote.execute(client,check),flush=True)
    launch="""cd /workspace/run042 && test ! -e runtime/pipeline-013.pid && nohup bash -c 'cd /workspace/run042; bash prelaunch/pipeline-013.sh >runtime/pipeline-013.log 2>&1; echo $? >runtime/pipeline-013.exit' >/workspace/run042/runtime/pipeline-launch-013.log 2>&1 </dev/null &"""
    # Separate small remote launcher retains its own PID before disconnection.
    launcher="#!/bin/bash\nset -e\ncd /workspace/run042\ntest ! -e runtime/pipeline-013.pid\nnohup bash -c 'bash prelaunch/pipeline-013.sh >runtime/pipeline-013.log 2>&1; echo $? >runtime/pipeline-013.exit' >runtime/pipeline-launch-013.log 2>&1 </dev/null &\necho $! >runtime/pipeline-013.pid\ncat runtime/pipeline-013.pid\n"
    with sftp.file('/workspace/run042/runtime/start-013.sh','w') as f:f.write(launcher)
    pid=remote.execute(client,'bash /workspace/run042/runtime/start-013.sh').strip()
    (RUN/'prelaunch/launch-013.json').write_text(json.dumps({'pid':int(pid),'overlay_sha256':digest,'deadline_epoch':1789926479},indent=2)+'\n')
    print({'pid':pid,'detached':True},flush=True)
    client.close()

if __name__=='__main__': main()
