"""Transfer the unchanged sealed archive through runpodctl's encrypted file lane."""
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
try:
    with (root / 'tmp/run068-croc-upload-private.log').open('w', encoding='utf-8') as log:
        sender = subprocess.Popen([str(transport.remote.CLI), 'send', str(archive)],
                                  stdout=subprocess.PIPE, stderr=log, text=True, encoding='utf-8')
        code = sender.stdout.readline().strip()
        if not re.fullmatch('[A-Za-z0-9-]+', code):
            sender.terminate()
            raise RuntimeError('Unrecognized private transfer handshake')
        # Only the ciphertext transfer code enters this transient remote command/log.
        command = ('cd /workspace/run068-control/incoming-003 && '
                   "printf 'y\\n' | /workspace/run068-control/runpodctl-v2.14.0 receive " + shlex.quote(code) +
                   ' > /workspace/run068-control/receive-003.private.log 2>&1; '
                   'result=${PIPESTATUS[1]}; exit "$result"')
        transport.remote.execute(client, command, timeout=300)
        rest = sender.stdout.read()
        log.write(rest)
        if sender.wait(timeout=30) != 0:
            raise RuntimeError('Local sender failed')
    expected = json.loads((RUN / 'prelaunch/bundle.json').read_text())['archive']
    incoming = '/workspace/run068-control/incoming-003/' + archive.name
    actual = transport.remote.execute(client, 'sha256sum ' + incoming).split()[0]
    if actual != expected['sha256']:
        raise RuntimeError('Direct transfer hash mismatch')
    transport.remote.execute(client,
        'if test -f /workspace/run068-head-upload.tar.gz; then '
        'mv /workspace/run068-head-upload.tar.gz /workspace/run068-control/upload-partial-002.tar.gz; fi; '
        'mv ' + incoming + ' /workspace/run068-head-upload.tar.gz')
    # The ordinary launcher re-verifies SHA256 and all81 members. Its runtime bytes stay sealed.
    transport.remote.upload = lambda sftp, local, remote_path: None
    transport.prepare(client, lease, '35a4c2b7d470191f52e439df961ab77c9b5daec2')
    receipt = {'status':'transferred-and-launched','lane':'runpodctl2.14.0 encrypted send/receive',
               'archive':expected,'both_transfer_exits':0,'seconds':time.monotonic()-started,
               'scientific_inputs_changed':False}
    (Path(__file__).parent / 'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
finally:
    client.close()
