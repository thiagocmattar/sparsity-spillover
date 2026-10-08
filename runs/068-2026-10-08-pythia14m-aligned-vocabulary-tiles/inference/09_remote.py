"""Transfer the sealed trained endpoints and launch the detached head comparison."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import sys

RUN = Path(__file__).resolve().parent
SOURCE = RUN.parent
sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location('run068_transport', SOURCE / '10_remote.py')
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)


def prepare(client, lease, source_commit):
    if len(source_commit) != 40 or any(c not in '0123456789abcdef' for c in source_commit):
        raise ValueError('Exact deployment commit required')
    control = '/workspace/run068-control'
    destination = '/workspace/run068-head'
    remote.execute(client, f'mkdir -p {control}; test ! -e {destination}')
    receipt = json.loads((RUN / 'prelaunch/bundle.json').read_text())
    inventory = json.loads((RUN / 'provenance/trained-upload-inventory.json').read_text())
    bundle = remote.ROOT / receipt['archive']['path']
    with client.open_sftp() as sftp:
        sftp.put(str(SOURCE / '12_deadline_guard.py'), control + '/12_deadline_guard.py')
        with sftp.open(control + '/guard-settings.json', 'w') as handle:
            handle.write(json.dumps({'pod_id': lease['pod']['id'], 'name': lease['pod']['name'],
                                    'deadline_epoch': lease['deadline_epoch'], 'control': control}))
        sftp.chmod(control + '/guard-settings.json', 0o600)
        remote.execute(client, f'nohup python3 -u {control}/12_deadline_guard.py '
                       f'{control}/guard-settings.json > {control}/guard.log 2>&1 < /dev/null &')
        remote.upload(sftp, bundle, '/workspace/run068-head-upload.tar.gz')
    if remote.execute(client, 'sha256sum /workspace/run068-head-upload.tar.gz').split()[0] != receipt['archive']['sha256']:
        raise RuntimeError('Endpoint bundle transfer hash mismatch')
    # Locally sealed membership is checked again after extraction, before setup.
    validation = """import hashlib,json
from pathlib import Path
root=Path('/workspace/run068-head')
run=root/'runs/068-2026-10-08-pythia14m-aligned-vocabulary-tiles/inference'
rows=json.loads((run/'provenance/trained-upload-inventory.json').read_text())
for row in rows:
 p=root/row['path']
 assert p.resolve().is_relative_to(root.resolve())
 with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
 assert p.stat().st_size==row['bytes'] and digest==row['sha256'],row['path']
print(json.dumps({'verified_upload_files':len(rows)}))
"""
    remote.execute(client, f'mkdir -p {destination} && tar -xzf /workspace/run068-head-upload.tar.gz -C {destination}')
    checked = remote.execute(client, "python3 - <<'PY'\n" + validation + 'PY\n')
    run = destination + '/runs/' + SOURCE.name + '/inference'
    wrapper = (f'cd {destination}; export RUN068_HEAD_SOURCE_COMMIT={source_commit}; '
               f'bash {run}/00_setup_remote.sh && bash {run}/03_execute.sh; '
               f'code=$?; printf "%s\\n" "$code" > {control}/pipeline.exit')
    remote.execute(client, f'nohup setsid bash -c {shlex.quote(wrapper)} > {control}/pipeline.log '
                   f'2>&1 < /dev/null & echo $! > {control}/pipeline.pid')
    result = {'status': 'uploaded-and-launched', 'source_commit': source_commit,
              'bundle': receipt, 'verified_upload_files': len(inventory),
              'verification_output': checked.strip(), 'pod_id': lease['pod']['id']}
    (RUN / 'prelaunch/upload-benchmark-001.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'command', 'get'])
    parser.add_argument('--source-commit')
    parser.add_argument('--command-file', type=Path)
    parser.add_argument('--remote')
    parser.add_argument('--local', type=Path)
    args = parser.parse_args()
    client, lease = remote.connect('benchmark')
    try:
        if args.action == 'prepare':
            prepare(client, lease, args.source_commit)
        elif args.action == 'command':
            print(remote.execute(client, args.command_file.read_text()), flush=True)
        else:
            args.local.parent.mkdir(parents=True, exist_ok=True)
            with client.open_sftp() as sftp:
                sftp.get(args.remote, str(args.local))
    finally:
        client.close()
