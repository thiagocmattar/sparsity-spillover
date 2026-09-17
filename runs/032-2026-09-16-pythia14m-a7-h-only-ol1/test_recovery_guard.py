"""Exercise the cost guard without credentials, network calls, or real waits."""
import ast
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

import pytest

SOURCE=Path(__file__).with_name('22_retry_kappa05.py')
TREE=ast.parse(SOURCE.read_text())
GUARD=next(ast.literal_eval(node.value) for node in TREE.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BOOT_GUARD' for t in node.targets))

def harness(monkeypatch,name='run032-k0p5',transient=False):
    clock=[100.0]
    calls=[]
    failures={'GET':int(transient),'POST':int(transient)}
    monkeypatch.setenv('RUNPOD_POD_ID','ownpodcopy')
    monkeypatch.setenv('RUN032_RECOVERY_STOP_KEY','test-only-credential')
    monkeypatch.setattr(time,'time',lambda:clock[0])
    monkeypatch.setattr(time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    def open_request(req,timeout):
        calls.append((clock[0],req.full_url,req.get_method(),req.data))
        method=req.get_method()
        if failures[method]:
            failures[method]-=1
            raise urllib.error.HTTPError(req.full_url,503,'temporary',{},None)
        return io.BytesIO(json.dumps({'id':'ownpodcopy','name':name}).encode())
    monkeypatch.setattr(urllib.request,'urlopen',open_request)
    return clock,calls

def test_boot_guard_stops_its_runtime_pod_after_one_hour(monkeypatch):
    clock,calls=harness(monkeypatch,name='run032-k0p5-migrated')
    exec(GUARD,{})
    posts=[c for c in calls if c[2]=='POST']
    assert len(posts)==1
    assert posts[0][0]>=3700
    assert posts[0][1]=='https://api.runpod.io/v2/pods/ownpodcopy/action'
    assert json.loads(posts[0][3])=={'action':'stop'}
    import os
    assert 'RUN032_RECOVERY_STOP_KEY' not in os.environ

def test_boot_guard_retries_transient_api_errors(monkeypatch):
    clock,calls=harness(monkeypatch,transient=True)
    exec(GUARD,{})
    assert [c[2] for c in calls]==['GET','GET','POST','POST']
    assert all(c[0]>=3700 for c in calls if c[2]=='POST')

def test_boot_guard_never_stops_an_unmatched_name(monkeypatch):
    clock,calls=harness(monkeypatch,name='unrelated-pod')
    class Halt(BaseException):pass
    monkeypatch.setattr(time,'sleep',lambda seconds:(_ for _ in ()).throw(Halt()))
    with pytest.raises(Halt):exec(GUARD,{})
    assert [c[2] for c in calls]==['GET']
