python3 - <<'PY'
from pathlib import Path
import json,time,os
control=Path('/workspace/run062-latency-control')
log=control/'pipeline.log'
out={'log_age_seconds':time.time()-log.stat().st_mtime,'uv_processes':[]}
for path in Path('/proc').glob('[0-9]*'):
 try:
  args=(path/'cmdline').read_bytes().split(b'\0')
  if len(args)<3 or not args[0].endswith(b'/uv') or args[1:3]!=[b'pip',b'install']: continue
  io=(path/'io').read_text()
  files=[]
  for fd in (path/'fd').iterdir():
   try:
    target=os.readlink(fd)
    if '/run062-uv-cache/' not in target: continue
    stat=fd.stat();files.append({'target':target,'bytes':stat.st_size,'age_seconds':time.time()-stat.st_mtime})
   except OSError: pass
  out['uv_processes'].append({'pid':int(path.name),'state':(path/'wchan').read_text(),'io':io,'open_cache_files':files})
 except OSError: pass
print(json.dumps(out,indent=2))
PY
