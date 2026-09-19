"""Retrieve the closed output archive; verification is a separate local step."""
from pathlib import Path
import hashlib
import json
import remote


if __name__ == '__main__':
    destination = Path(__file__).resolve().parent.parent / 'retrieval'
    destination.mkdir(exist_ok=True)
    client = remote.connect()
    try:
        with client.open_sftp() as sftp:
            for name in ['receipt-001.json', 'output-001.tar.gz']:
                target = destination / name
                if target.exists():
                    raise FileExistsError(f'Preserve prior transfer: {target}')
                temporary = target.with_suffix(target.suffix + '.partial')
                if temporary.exists():
                    raise FileExistsError(f'Inspect incomplete transfer: {temporary}')
                sftp.get('/workspace/run038/transfer/' + name, str(temporary),
                         callback=remote.progress())
                temporary.rename(target)
                print(f'Retrieved {name}', flush=True)
            # The collector snapshots its own worker log before the final exit.
            # Preserve the final state separately, without replacing that snapshot.
            command = """python3 - <<'PY'
import hashlib,json
from pathlib import Path
r=Path('/workspace/run038')
paths={'execution-final.log':'runtime/execution-001.log',
       'execution-final.exit':'runtime/execution-001.exit',
       'deadline-guard-final.log':'runtime/deadline-guard.log'}
assert (r/paths['execution-final.exit']).is_file()
rows=[]
for name,path in paths.items():
 b=(r/path).read_bytes();rows.append({'name':name,'remote_path':path,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
print(json.dumps(rows))
PY"""
            rows = json.loads(remote.execute(client, command))
            closeout = destination.parent / 'artifacts/closeout'
            closeout.mkdir(parents=True, exist_ok=True)
            for row in rows:
                target = closeout / row['name']
                if target.exists():
                    raise FileExistsError(target)
                sftp.get('/workspace/run038/' + row['remote_path'], str(target))
                content = target.read_bytes()
                assert len(content) == row['bytes']
                assert hashlib.sha256(content).hexdigest() == row['sha256']
            (closeout/'final-files.json').write_text(json.dumps(rows, indent=2)+'\n',
                                                    encoding='utf-8', newline='\n')
            print('Verified final worker/guard records', flush=True)
    finally:
        client.close()
