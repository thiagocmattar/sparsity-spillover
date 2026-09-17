"""On-Pod, scoped recovery deadline; credentials are read once and unlinked."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import urllib.request

secret_path = Path(sys.argv[1])
settings = json.loads(secret_path.read_text())
secret_path.unlink()
pod_id = settings['pod_id']
expected_name = settings['name']
assert expected_name.startswith('run032-') and pod_id.isalnum()
base = 'https://api.runpod.io/v2/pods/' + pod_id
headers = {'Authorization': 'Bearer ' + settings['api_key'],
           'Content-Type': 'application/json', 'User-Agent': 'run032-recovery'}


def request(action=None):
    req = urllib.request.Request(base if action is None else base + '/action',
          data=None if action is None else json.dumps({'action': action}).encode(),
          headers=headers, method='GET' if action is None else 'POST')
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.load(response)


current = request()
if current.get('name') != expected_name:
    raise RuntimeError('Recovery target identity mismatch')
print(json.dumps({'event': 'armed', 'pod_id': pod_id,
                  'utc': datetime.now(timezone.utc).isoformat(),
                  'deadline_epoch': settings['deadline_epoch']}), flush=True)
while time.time() < settings['deadline_epoch']:
    time.sleep(min(30, max(0, settings['deadline_epoch'] - time.time())))
for attempt in range(4):
    try:
        request('stop')
        print('STOP_REQUESTED', flush=True)
        break
    except Exception as error:
        print(type(error).__name__, getattr(error, 'code', None), flush=True)
        time.sleep(10)
