import pathlib,time,json,hashlib,numpy as np,shutil
source=pathlib.Path('/workspace/sparsity-spillover/data/tokenized/minipile-pythia-14m-full/train/tokens.int32.bin')
local=pathlib.Path('/tmp/run054-data/train.tokens.int32.bin');local.parent.mkdir(exist_ok=True)
t=time.perf_counter();shutil.copyfile(source,local);print(json.dumps({'copy_seconds':time.perf_counter()-t}),flush=True)
with local.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()=='da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c'
for p in [source,local]:
 data=np.memmap(p,dtype=np.int32,mode='r');times=[]
 for repeat in range(3):
  starts=np.random.default_rng(1234+repeat).integers(0,len(data)//2048,size=1024)*2048
  t=time.perf_counter()
  for batch in starts.reshape(-1,4):tmp=np.stack([data[int(s):int(s)+2048] for s in batch])
  times.append(time.perf_counter()-t)
 print(json.dumps({'path':str(p),'one_update_data_assembly_seconds':times}),flush=True)
