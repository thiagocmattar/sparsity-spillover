import concurrent.futures,subprocess,pathlib,math,json,time
source=pathlib.Path('/opt/run034-inputs.tar')
offset=3889984512
size=source.stat().st_size
piece=math.ceil((size-offset)/4)
base=['ssh','-i','/opt/run034-transfer-key','-o','BatchMode=yes','-o','ServerAliveInterval=15','-p','10529','root@213.181.104.55']
def copy(index):
 start=offset+index*piece
 count=min(piece,size-start)
 if count<=0:return
 process=subprocess.Popen(base+['cat > /opt/run034-tail-'+str(index)],stdin=subprocess.PIPE)
 with source.open('rb') as handle:
  handle.seek(start);remaining=count
  while remaining:
   chunk=handle.read(min(1024*1024,remaining));assert chunk
   process.stdin.write(chunk);remaining-=len(chunk)
 process.stdin.close()
 assert process.wait()==0
 print(json.dumps({'stripe':index,'bytes':count,'time':time.time()}),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(copy,range(4)))
command='cat /opt/run034-tail-0 /opt/run034-tail-1 /opt/run034-tail-2 /opt/run034-tail-3 >> /opt/run034-inputs.tar && echo "7824126df3516a84dd7440ad453dfabce7d4571d988622340680e81b3ba3c8ee  /opt/run034-inputs.tar" | sha256sum -c - && tar -xf /opt/run034-inputs.tar -C /workspace/sparsity-spillover && touch /workspace/run034-control/initialization-ready /workspace/run034-control/cache-ready'
subprocess.run(base+[command],check=True)
