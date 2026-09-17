"""Run034 read-only monitoring and verified retrieval before owned-Pod deletion.

Run with --once for a status snapshot or without it for the detached five-minute
loop. Scientific failures are preserved and reported, never silently retrained.
The separate on-Pod and local deadline guards remain authoritative stop limits.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parents[1]
RECORDS = HERE / 'prelaunch'
spec = importlib.util.spec_from_file_location('cloud034', HERE / '07_cloud.py')
cloud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloud)


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def remote_status(label):
    conditions = json.loads((RECORDS / 'assignments.json').read_text())[label]
    script = '''import json,os,time,subprocess,statistics
from pathlib import Path
root=Path(ROOT)
control=Path(CONTROL)
rows=[]
for condition in CONDITIONS:
 p=control/condition
 row={'condition':condition,'verified':(p/'verified').exists(),
      'preflight_passed':(p/'preflight-passed').exists()}
 for key,name in [('exit_code','exit-code'),('pid','pid')]:
  if (p/name).exists():row[key]=(p/name).read_text().strip()
 if row.get('pid'):
  try:os.kill(int(row['pid']),0);row['process_alive']=True
  except ProcessLookupError:row['process_alive']=False
 for attempt in (root/'artifacts/attempts').glob('*'):
  manifest=attempt/'manifest.json'
  if not manifest.exists():continue
  m=json.loads(manifest.read_text())
  if m.get('condition',{}).get('id')!=condition:continue
  row['attempt']=attempt.name;row['attempt_status']=m.get('status')
  events=attempt/'events.jsonl'
  if events.exists():
   lines=events.read_text().splitlines();parsed=[]
   for line in lines:
    try:parsed.append(json.loads(line))
    except json.JSONDecodeError:pass
   trains=[e for e in parsed if e.get('event')=='train']
   if trains:
    row['last_train']=trains[-1]
    row['median_recent_step_seconds']=statistics.median([e['step_wall_seconds'] for e in trains[-10:]])
   row['seconds_since_event']=time.time()-events.stat().st_mtime
 pf=root/'prelaunch'/('remote-preflight-'+condition+'.json')
 if pf.exists():
  data=json.loads(pf.read_text());row['preflight_status']=data['status'];row['projection']=data['projection']
 if row.get('exit_code') not in (None,'0'):
  row['failure_logs']={f:(p/f).read_text(errors='replace')[-3500:] for f in ['preflight.log','train.log','verification.log'] if (p/f).exists()}
 rows.append(row)
print(json.dumps({'conditions':rows,'ready_markers':[p.name for p in control.glob('*ready')],
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True),
 'disk':subprocess.check_output(['df','-BG',str(root)],text=True)}))
'''.replace('ROOT', repr(cloud.REMOTE_RUN)).replace('CONTROL', repr(cloud.CONTROL)).replace('CONDITIONS', repr(conditions))
    with cloud.connect(label) as connection:
        output = cloud.command(connection, 'python3 -c ' + shlex.quote(script), timeout=45)
    result = json.loads(output)
    result.update(label=label, timestamp=stamp())
    for row in result['conditions']:
        event = row.get('last_train', {})
        if event:
            common = row.get('projection', {}).get('measured_common_seconds_conservative', 300)
            row['remaining_seconds_estimate'] = max(0, 712-event['step']) * row['median_recent_step_seconds'] + common
        row['warnings'] = []
        if row.get('exit_code') not in (None, '0'):
            row['warnings'].append('process exited unsuccessfully')
        if row.get('attempt_status') == 'failed':
            row['warnings'].append('scientific attempt failed; preserve evidence')
        if row.get('seconds_since_event', 0) > 600 and row.get('attempt_status') == 'running':
            row['warnings'].append('event stale beyond ten minutes')
        if event.get('optimizer_step_skipped') or event.get('gradient_overflow'):
            row['warnings'].append('optimizer overflow or skipped boundary')
    return result


def status_safe(label):
    receipt = RECORDS / 'retrieval' / label / 'receipt.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('pod_deleted'):
        return {'label': label, 'timestamp': stamp(), 'state': 'retrieved_verified_deleted'}
    try:
        return remote_status(label)
    except Exception as error:
        return {'label': label, 'timestamp': stamp(), 'error': str(error)}


def sync_ready_checkpoints(label, limit=1):
    """Copy only checkpoints whose final metadata marker has been published.

    Checkpoints are immutable after that marker. Partial downloads never appear
    at the final artifact path, and every copied file has its remote SHA checked.
    One checkpoint per Pod per normal tick keeps early recovery fair across Pods.
    Terminal retrieval calls this without a limit before scientific verification.
    """
    local = RECORDS / 'retrieval' / label
    local.mkdir(parents=True, exist_ok=True)
    ledger_path = local / 'checkpoint-files.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    assignments = json.loads((RECORDS / 'assignments.json').read_text())[label]
    script = '''import pathlib,json,hashlib
root=pathlib.Path(ROOT)
known=set(KNOWN)
groups=[]
for attempt in (root/'artifacts/attempts').glob('*'):
 manifest=attempt/'manifest.json'
 if not manifest.exists():continue
 m=json.loads(manifest.read_text())
 if m.get('condition',{}).get('id') not in CONDITIONS:continue
 for marker in (attempt/'checkpoints').glob('step_*/checkpoint_metadata.json'):
  try:state=json.loads(marker.read_text())
  except json.JSONDecodeError:continue
  missing=[p for p in marker.parent.iterdir() if p.is_file() and p.relative_to(root).as_posix() not in known]
  if missing:groups.append((state['step'],missing))
groups.sort(key=lambda item:item[0],reverse=True)
if LIMIT is not None:groups=groups[:LIMIT]
rows=[]
for _,files in groups:
 for p in files:
  with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
  rows.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha})
print(json.dumps(rows))
'''.replace('ROOT', repr(cloud.REMOTE_RUN)).replace('KNOWN', repr(list(ledger))).replace('CONDITIONS', repr(assignments)).replace('LIMIT', repr(limit))
    with cloud.connect(label) as connection:
        files = json.loads(cloud.command(connection, 'python3 -c ' + shlex.quote(script), timeout=180))
    info = json.loads((RECORDS / f'ssh-{label}.json').read_text())
    # Reuse byte-identical checkpoints already recovered from another condition.
    available = {}
    for other in (RECORDS / 'retrieval').glob('*/checkpoint-files.json'):
        for path, row in json.loads(other.read_text()).items():
            candidate = HERE / path
            if candidate.exists():
                available[row['sha256']] = candidate
    copied = 0
    for row in files:
        target = (HERE / row['path']).resolve()
        assert target.is_relative_to((HERE / 'artifacts/attempts').resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            duplicate = available.get(row['sha256'])
            if duplicate:
                assert digest(duplicate) == row['sha256']
                os.link(duplicate, target)
            else:
                temporary = target.with_name(target.name + '.partial')
                command = ['scp', '-B', '-X', 'nrequests=256', '-X', 'buffer=131072',
                           '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new',
                           '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=6',
                           '-i', info['ssh_key']['path'], '-P', str(info['port']),
                           'root@' + info['ip'] + ':' + cloud.REMOTE_RUN + '/' + row['path'], str(temporary)]
                with (local / 'checkpoint-download.log').open('ab') as log:
                    subprocess.run(command, check=True, stdout=log, stderr=log, timeout=1800)
                assert temporary.stat().st_size == row['bytes'] and digest(temporary) == row['sha256'], 'Checkpoint transfer hash mismatch'
                temporary.replace(target)
        assert target.stat().st_size == row['bytes'] and digest(target) == row['sha256']
        ledger[row['path']] = {**row, 'verified_at': stamp()}
        ledger_temporary = ledger_path.with_suffix('.json.tmp')
        cloud.write(ledger_temporary, ledger)
        ledger_temporary.replace(ledger_path)
        available[row['sha256']] = target
        copied += row['bytes']
    return {'label': label, 'verified_files_this_sync': len(files), 'verified_bytes_this_sync': copied,
            'total_verified_bytes': sum(row['bytes'] for row in ledger.values())}


def retrieve_and_delete(label):
    assignments = json.loads((RECORDS / 'assignments.json').read_text())[label]
    local = RECORDS / 'retrieval' / label
    local.mkdir(parents=True, exist_ok=True)
    receipt_path = local / 'receipt.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    lease = json.loads((RECORDS / f'lease-{label}.json').read_text())
    if not receipt.get('locally_verified'):
        snapshot = remote_status(label)
        assert all(r['verified'] and r.get('exit_code') == '0' for r in snapshot['conditions'])
        cloud.write(local / 'completion-snapshot.json', snapshot)
        sync_ready_checkpoints(label, limit=None)
        archive_remote = '/workspace/run034-results.tar'
        with cloud.connect(label) as connection:
            command = ' && '.join('test -f ' + shlex.quote(cloud.CONTROL + '/' + c + '/verified') for c in assignments)
            command += " && tar -b 2048 --exclude='artifacts/attempts/*/checkpoints' -cf " + archive_remote + ' -C ' + cloud.REMOTE_RUN
            command += ' artifacts ' + ' '.join('prelaunch/remote-preflight-' + c + '.json' for c in assignments)
            command += ' -C /workspace run034-control'
            command += ' && sha256sum ' + archive_remote
            remote_sha = cloud.command(connection, command, timeout=600).split()[0]
        assert len(remote_sha) == 64
        info = json.loads((RECORDS / f'ssh-{label}.json').read_text())
        archive = local / 'results.tar'
        if not archive.exists() or digest(archive) != remote_sha:
            command = ['scp', '-B', '-X', 'nrequests=256', '-X', 'buffer=131072', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new',
                       '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=6',
                       '-i', info['ssh_key']['path'], '-P', str(info['port']),
                       'root@' + info['ip'] + ':' + archive_remote, str(archive)]
            with (local / 'download.log').open('ab') as log:
                subprocess.run(command, check=True, stdout=log, stderr=log, timeout=3600)
        assert digest(archive) == remote_sha, 'Downloaded archive hash mismatch'
        extracted = local / 'extracted'
        extracted.mkdir(exist_ok=True)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                destination = (extracted / member.name).resolve()
                if not destination.is_relative_to(extracted.resolve()) or member.issym() or member.islnk():
                    raise ValueError('Unsafe archive entry')
            tar.extractall(extracted, filter='data')
        for subtree in ['artifacts', 'prelaunch']:
            for source in (extracted / subtree).rglob('*'):
                if not source.is_file():
                    continue
                target = HERE / source.relative_to(extracted)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    assert digest(target) == digest(source), 'Existing local artifact differs: ' + str(target)
                else:
                    shutil.copy2(source, target)
        verified = []
        for condition in assignments:
            process = subprocess.run([sys.executable, str(HERE / '03_verify.py'), '--condition', condition],
                                     capture_output=True, text=True, timeout=600)
            (local / (condition + '-verification.txt')).write_text(process.stdout + process.stderr, encoding='utf-8')
            if process.returncode:
                raise RuntimeError('Local scientific verification failed: ' + condition)
            verified.append(condition)
        receipt = {'timestamp': stamp(), 'label': label, 'pod_id': lease['pod']['id'],
                   'archive_sha256': remote_sha, 'archive_bytes': archive.stat().st_size,
                   'conditions': verified, 'locally_verified': True, 'pod_deleted': False}
        cloud.write(receipt_path, receipt)
    # Deletion is allowed only after the persisted local scientific-verification receipt.
    assert receipt['locally_verified'] and receipt['pod_id'] == lease['pod']['id']
    pods = cloud.cli('pod', 'list', '--all')
    present = [p for p in pods if p['id'] == lease['pod']['id']]
    if present:
        assert len(present) == 1 and present[0]['name'] == 'run034-' + label
        cloud.cli('pod', 'delete', lease['pod']['id'])
    assert not any(p['id'] == lease['pod']['id'] for p in cloud.cli('pod', 'list', '--all'))
    receipt.update(pod_deleted=True, deletion_confirmed_at=stamp())
    cloud.write(receipt_path, receipt)
    guard_pid = RECORDS / ('local-guard-' + label + '.pid')
    if guard_pid.exists():
        pid = int(guard_pid.read_text().strip())
        stop = (f'$g = Get-CimInstance Win32_Process -Filter "ProcessId = {pid}"; '
                f'if ($g.CommandLine -like "*09_local_stop_guard.ps1*" -and '
                f'$g.CommandLine -like "*{lease["pod"]["id"]}*") '
                f'{{ Stop-Process -Id {pid} -ErrorAction SilentlyContinue }}')
        subprocess.run(['powershell', '-NoProfile', '-Command', stop],
                       capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    labels = list(json.loads((RECORDS / 'assignments.json').read_text()))
    history = RECORDS / 'monitoring.jsonl'
    transfers = ThreadPoolExecutor(max_workers=2)
    pending = {}
    while True:
        with ThreadPoolExecutor(max_workers=len(labels)) as pool:
            rows = list(pool.map(status_safe, labels))
        snapshot = {'timestamp': stamp(), 'pods': rows}
        cloud.write(RECORDS / 'latest-status.json', snapshot)
        with history.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(snapshot) + '\n')
        compact = [{'label': r['label'], 'state': r.get('state'), 'error': r.get('error'),
                    'conditions': [{k:v for k,v in c.items() if k in ['condition','verified','preflight_passed','attempt_status','remaining_seconds_estimate','warnings']} |
                                   {'step': c.get('last_train', {}).get('step'), 'loss': c.get('last_train', {}).get('task_loss'),
                                    'tokens_per_second': c.get('last_train', {}).get('tokens_per_second')}
                                   for c in r.get('conditions', [])]} for r in rows]
        print(json.dumps({'timestamp': stamp(), 'pods': compact}), flush=True)
        if args.once:
            transfers.shutdown(wait=False)
            return
        for label, (kind, future) in list(pending.items()):
            if not future.done():
                continue
            try:
                print(json.dumps({kind: future.result()}), flush=True)
            except Exception as error:
                print(json.dumps({'retrieval_error': label, 'error': str(error)}), flush=True)
            del pending[label]
        for row in rows:
            if row['label'] not in pending and row.get('conditions') and all(c['verified'] and c.get('exit_code') == '0' for c in row['conditions']):
                pending[row['label']] = ('retrieval', transfers.submit(retrieve_and_delete, row['label']))
            elif row['label'] not in pending and any(c.get('attempt_status') in ['running','completed'] for c in row.get('conditions', [])):
                pending[row['label']] = ('checkpoint_sync', transfers.submit(sync_ready_checkpoints, row['label']))
        if all(r.get('state') == 'retrieved_verified_deleted' for r in rows):
            transfers.shutdown(wait=True)
            process = subprocess.run([sys.executable, str(HERE / '03_verify.py')], capture_output=True, text=True)
            (RECORDS / 'cohort-verification.txt').write_text(process.stdout + process.stderr, encoding='utf-8')
            cloud.write(RECORDS / 'final-resource-audit.json', {'timestamp': stamp(), 'pods': cloud.cli('pod', 'list', '--all')})
            if process.returncode:
                raise RuntimeError('Cohort verification failed')
            return
        # Avoid polling unchanged processes; check earlier only for projected completion.
        remaining = [c['remaining_seconds_estimate'] for r in rows for c in r.get('conditions', [])
                     if c.get('remaining_seconds_estimate', 0) > 0 and not c.get('verified')]
        time.sleep(max(30, min([300] + remaining)))


if __name__ == '__main__':
    main()
