"""Transfer frozen inputs while installing the pinned runtime under /workspace."""
import json
import time
import paramiko
from remote_io import HERE, connect, execute, progress


def main():
    c=connect(refresh=True)
    print(execute(c,'mkdir -p /workspace/run036/runtime /workspace/run036/provenance; nvidia-smi -L; df -h /workspace'),flush=True)
    s=paramiko.SFTPClient.from_transport(c.get_transport(),window_size=32*1024*1024,max_packet_size=128*1024)
    s.put(str(HERE/'provenance/pip-freeze.txt'),'/workspace/run036/provenance/pip-freeze.txt')
    setup=(HERE/'04_setup.sh').read_text()
    setup=setup.replace('"$RUN036/runtime/venv/bin/python" "$RUN036/01_prepare.py" verify',
        'while [ ! -f "$RUN036/runtime/inputs-ready" ]; do sleep 5; done\n"$RUN036/runtime/venv/bin/python" "$RUN036/01_prepare.py" verify')
    with s.open('/workspace/run036/environment-overlap.sh','w') as f:f.write(setup)
    print(execute(c,"cd /workspace/run036; nohup bash -c 'bash environment-overlap.sh; echo $? > runtime/setup.exit' > runtime/setup.log 2>&1 < /dev/null & echo $!"),flush=True)
    started=time.monotonic()
    s.put(str(HERE/'bundles/input-001.tar'),'/workspace/run036/input-001.tar',callback=progress())
    s.put(str(HERE/'bundles/input-001.receipt.json'),'/workspace/run036/input-receipt.json')
    script='''#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run036
python3 - <<'PY'
import json,hashlib
from pathlib import Path
p=Path('input-001.tar');r=json.loads(Path('input-receipt.json').read_text())
assert p.stat().st_size==r['bytes']
with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256']
print('INPUT_ARCHIVE_HASH_VERIFIED',flush=True)
PY
tar -xf input-001.tar
touch runtime/inputs-ready
'''
    with s.open('/workspace/run036/extract-input.sh','w') as f:f.write(script)
    print(execute(c,"cd /workspace/run036; nohup bash -c 'bash extract-input.sh; echo $? > runtime/extract.exit' > runtime/extract.log 2>&1 < /dev/null & echo $!"),flush=True)
    receipt={'uploaded_bytes':(HERE/'bundles/input-001.tar').stat().st_size,'seconds':time.monotonic()-started}
    (HERE/'prelaunch/upload-001.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    s.close();c.close()


if __name__=='__main__':main()
