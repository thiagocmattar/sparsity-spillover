"""Copy completed attempt artifacts and verify local hashes against remote bytes."""
import argparse,hashlib,json,stat
from pathlib import Path
from transport import connect,execute
p=argparse.ArgumentParser();p.add_argument('attempts',nargs='+');a=p.parse_args()
run=Path(__file__).resolve().parent.parent
client=connect();receipts=[]
try:
 s=client.open_sftp()
 for attempt in a.attempts:
  assert attempt.replace('-','').isalnum()
  root='/workspace/run050/artifacts/'+attempt
  def visit(remote,local):
   local.mkdir(parents=True,exist_ok=True)
   for item in s.listdir_attr(remote):
    src=remote+'/'+item.filename;dst=local/item.filename
    if stat.S_ISDIR(item.st_mode):visit(src,dst)
    elif stat.S_ISREG(item.st_mode):
     digest=execute(client,'sha256sum '+src).split()[0]
     if not dst.exists() or hashlib.sha256(dst.read_bytes()).hexdigest()!=digest:s.get(src,str(dst))
     assert hashlib.sha256(dst.read_bytes()).hexdigest()==digest
     receipts.append({'path':dst.relative_to(run).as_posix(),'bytes':dst.stat().st_size,'sha256':digest})
  visit(root,run/'artifacts'/attempt)
  for suffix in ('log','exit','pid','sh'):
   try:s.get(f'/workspace/run050-control/{attempt}.{suffix}',str(run/'prelaunch'/f'{attempt}.{suffix}'))
   except FileNotFoundError:pass
 s.close()
 dest=run/'retrieval';dest.mkdir(exist_ok=True)
 (dest/('receipt-'+a.attempts[-1]+'.json')).write_text(json.dumps({'files':receipts},indent=2)+'\n')
 print(json.dumps({'verified_files':len(receipts),'bytes':sum(r['bytes'] for r in receipts)}))
finally:client.close()
