"""Scoped on-Pod stop deadline; consume and unlink the temporary credential file."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import urllib.request


def main():
    secret_path=Path(sys.argv[1])
    settings=json.loads(secret_path.read_text())
    secret_path.unlink()
    pod_id=settings['pod_id'];expected_name=settings['name']
    if not (expected_name.startswith('run056-') and pod_id.isalnum()):
        raise ValueError('Unscoped target')
    base='https://api.runpod.io/v2/pods/'+pod_id
    headers={'Authorization':'Bearer '+settings['api_key'],'Content-Type':'application/json',
             'User-Agent':'run056-deadline'}

    def request(action=None):
        req=urllib.request.Request(base if action is None else base+'/action',
            data=None if action is None else json.dumps({'action':action}).encode(),
            headers=headers,method='GET' if action is None else 'POST')
        with urllib.request.urlopen(req,timeout=20) as response:
            body=response.read()
            return json.loads(body) if body else {}

    if request().get('name')!=expected_name:
        raise RuntimeError('Pod identity mismatch')
    print(json.dumps({'event':'armed','pod_id':pod_id,'deadline_epoch':settings['deadline_epoch'],
                      'utc':datetime.now(timezone.utc).isoformat()}),flush=True)
    while time.time()<settings['deadline_epoch']:
        time.sleep(min(30,max(0,settings['deadline_epoch']-time.time())))
    for attempt in range(12):
        try:
            request('stop');print('STOP_REQUESTED',flush=True);return
        except Exception as error:
            print(type(error).__name__,getattr(error,'code',None),flush=True);time.sleep(10)
    raise RuntimeError('Deadline stop could not reach the provider; independent local guard must intervene')


if __name__=='__main__':main()
