"""One bounded read-only progress summary after the agreed idle interval."""
import argparse,shlex
from transport import connect,execute
p=argparse.ArgumentParser();p.add_argument('attempt',default='final-001',nargs='?');a=p.parse_args()
assert a.attempt.replace('-','').isalnum()
code='''import pathlib,json,sys,datetime
r=pathlib.Path('/workspace/run051/artifacts');root=r/sys.argv[1]
def read(p):return json.loads(p.read_text()) if p.exists() else {}
s=read(root/'status.json');children=read(root/'children.json') or []
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':s.get('stage'),'completed':len(children),'total':s.get('total'),'current':s.get('child'),'remaining_minutes':round(s.get('remaining_seconds',0)/60,1) if s.get('remaining_seconds') else None}
if s.get('child'):
 x=read(r/s['child']/'status.json');d['current_stage']=x.get('stage');d['current_blocks']=x.get('blocks');d['blocks_per_second']=x.get('blocks_per_second');d['input_tokens_per_second']=x.get('input_tokens_per_second');d['current_implementation']=x.get('implementation')
if children:
 last=children[-1];x=read(r/last['attempt']/'result.json');d['last']=last['attempt'];d['last_seconds']=round(x.get('elapsed_seconds',0),1);d['last_exit']=last['exit_code'];d['all_last_qualified']=all(x.get('qualification',{}).values());d['last_loss']={k:round(v,7) for k,v in x.get('loss',{}).items() if k in ('native','sparse_c_graph')};d['last_ms']={k:round(v['geomean_host_ms'],6) for k,v in x.get('timing',{}).items()}
d['failed_qualifications']=[{'attempt':x['attempt'],'modes':[k for k,v in (x.get('qualified') or {}).items() if not v]} for x in children if x.get('qualified') and not all(x['qualified'].values())]
result=read(root/'result.json')
if result:d['terminal_status']=result['status']
print(json.dumps(d))
'''
c=connect()
try:print(execute(c,'python3 -c '+shlex.quote(code)+' '+shlex.quote(a.attempt),timeout=30).strip())
finally:c.close()
