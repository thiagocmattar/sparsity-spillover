"""Resume direct transfer with a detached receiver and explicit exit receipts."""
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

RUN = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RUN))
spec = importlib.util.spec_from_file_location('run068_head_transport', RUN / '09_remote.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
client, lease = transport.remote.connect('benchmark')
root = transport.remote.ROOT
archive = root / 'tmp/run068-head-upload.tar.gz'
started = time.monotonic()
sender = None
try:
    # Stop only a surviving receiver from the failed, source-identical transfer.
    transport.remote.execute(client,
        "for pid in $(pgrep -f '^/workspace/run068-control/runpodctl-v2.14.0 receive '); do kill \"$pid\"; done")
    with (root / 'tmp/run068-croc-upload-004-private.log').open('w', encoding='utf-8') as log:
        sender = subprocess.Popen([str(transport.remote.CLI), 'send', str(archive)],
                                  stdout=subprocess.PIPE, stderr=log, text=True, encoding='utf-8')
        code = sender.stdout.readline().strip()
        log.write(code + '\n'); log.flush()
        if not re.fullmatch('[A-Za-z0-9-]+', code):
            raise RuntimeError('Unrecognized private transfer handshake')
        command = ('cd /workspace/run068-control/incoming-003; '
                   'yes | /workspace/run068-control/runpodctl-v2.14.0 receive ' + shlex.quote(code) + '; '
                   'result=${PIPESTATUS[1]}; printf "%s\\n" "$result" > '
                   '/workspace/run068-control/receive-004.exit; exit "$result"')
        transport.remote.execute(client, 'nohup bash -c ' + shlex.quote(command) +
            ' > /workspace/run068-control/receive-004.private.log 2>&1 < /dev/null & '
            'echo $! > /workspace/run068-control/receive-004.pid')
        last = started
        while sender.poll() is None:
            time.sleep(10)
            now = time.monotonic()
            if now-last >= 30:
                p = root / 'tmp/run068-croc-upload-004-private.log'
                percentages = re.findall(r'(\d+)%', p.read_text(errors='replace'))
                print(json.dumps({'phase':'transfer','elapsed_seconds':now-started,
                                  'percent':percentages[-1] if percentages else None}),flush=True)
                last = now
        if sender.returncode != 0:
            raise RuntimeError('Local sender failed')
        log.write(sender.stdout.read())
    receiver = transport.remote.execute(client, 'cat /workspace/run068-control/receive-004.exit').strip()
    if receiver != '0':
        raise RuntimeError('Detached receiver failed')
    expected = json.loads((RUN / 'prelaunch/bundle.json').read_text())['archive']
    incoming = '/workspace/run068-control/incoming-003/' + archive.name
    if transport.remote.execute(client, 'sha256sum ' + incoming).split()[0] != expected['sha256']:
        raise RuntimeError('Direct transfer hash mismatch')
    transport.remote.execute(client,
        'if test -f /workspace/run068-head-upload.tar.gz; then '
        'mv /workspace/run068-head-upload.tar.gz /workspace/run068-control/upload-partial-before-004.tar.gz; fi; '
        'mv ' + incoming + ' /workspace/run068-head-upload.tar.gz')
    transport.remote.upload = lambda sftp, local, remote_path: None
    transport.prepare(client, lease, '35a4c2b7d470191f52e439df961ab77c9b5daec2')
    receipt = {'status':'transferred-and-launched','lane':'runpodctl2.14.0 encrypted send/receive',
               'archive':expected,'both_transfer_exits':0,'seconds':time.monotonic()-started,
               'scientific_inputs_changed':False,'receiver_detached':True}
    (Path(__file__).parent / 'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
finally:
    if sender is not None and sender.poll() is None:
        sender.terminate()
    client.close()
