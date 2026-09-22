"""Training qualification or frozen full validation in one fresh GPU process."""
import argparse,time,math,traceback
from support import RUN,BASE,read,write,event,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--condition',required=True,choices=('c00','c24','c25'))
 p.add_argument('--phase',choices=('smoke','training','final'),required=True);p.add_argument('--replicate',type=int,default=1)
 p.add_argument('--selection',required=True);a=p.parse_args()
 dest=RUN/'artifacts'/a.attempt;dest.mkdir(parents=True,exist_ok=False)
 started=time.monotonic();result={'status':'running','arguments':vars(a)}
 def emit(stage,**fields):event(dest,stage,condition=a.condition,elapsed_seconds=time.monotonic()-started,**fields)
 try:
  import bootstrap
  bootstrap.setup()
  import numpy as np,torch
  import adapter
  cfg=read(RUN/'config.json');selection_path=RUN/'provenance'/a.selection;selection=read(selection_path)
  if a.phase=='final':
   for name,digest in selection['kernel_source_hashes'].items():assert sha(RUN/name)==digest,('Frozen source changed',name)
  result['selection']={'path':selection_path.relative_to(RUN).as_posix(),'sha256':sha(selection_path)}
  result['source_hashes']={p.name:sha(p) for p in list(RUN.glob('*.py'))+list(RUN.glob('*.cu'))}
  result['checkpoint']=bootstrap.checkpoint(a.condition)
  if a.phase=='final':
   manifest=read(BASE/'provenance/inputs.json')['validation'];path=bootstrap.base_io.verify(manifest)
   data=np.memmap(path,dtype=np.int32,mode='r');assert divmod(len(data),2048)==(338,1444)
   blocks=338;indices=np.random.default_rng(cfg['timing_seed']).choice(338,64,replace=False);passes=7
  else:
   manifest=read(RUN/'provenance/inputs.json')['training'];path=RUN/manifest['path'];assert sha(path)==manifest['sha256']
   data=np.memmap(path,dtype=np.int32,mode='r');blocks=4 if a.phase=='smoke' else 128
   indices=np.arange(blocks);passes=2 if a.phase=='smoke' else 5
  result['data_identity']=manifest
  policy_modes={'dense_fused':{k:'dense_fused' for k in selection['dense']},'dense_policy':selection['dense'],**selection['policies'],**selection.get('ablations',{})}
  # Retain explicit one-site ablations only when they differ from a policy and
  # from its dense control; avoid reporting duplicate execution as new evidence.
  aliases={}
  for name,policy in list(selection['policies'].items()):
   for site in ('h','z'):
    ablation={k:policy[k] if k.startswith(site+'.') else selection['dense'][k] for k in policy}
    same=next((n for n,v in policy_modes.items() if v==ablation),None)
    if same:aliases[name+'_'+site+'_only']=same
    else:policy_modes[name+'_'+site+'_only']=ablation
  # Final selection can exclude policies rejected on the fixed training set.
  if 'qualified_modes' in selection and a.phase=='final':
   policy_modes={k:v for k,v in policy_modes.items() if k in selection['qualified_modes']}
  result.update(policies=policy_modes,ablation_aliases=aliases)
  modes=['native','native_hz','opt073',*policy_modes]
  inputs=[torch.tensor(data[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in indices]
  models={};runners={}
  with torch.inference_mode():
   for name in modes:
    emit('loading',implementation=name)
    model=bootstrap.model(a.condition,custom=name=='native_hz')
    if name=='opt073':bootstrap.base_install.install(model,'opt073')
    elif name in policy_modes:adapter.install(model,policy_modes[name])
    models[name]=model
    if name=='native':
     runner=bootstrap.replay.dense.DenseRunner(lambda x,m=model:m(input_ids=x,use_cache=False).logits,inputs[0].clone(),'native');runner.prepare();runners['native']=runner
    runner=bootstrap.replay.dense.DenseRunner(lambda x,m=model:bootstrap.replay.scaffold.forward(m,x),inputs[0].clone(),'graph')
    runner.prepare();runners[name+'_graph']=runner
   free,total=torch.cuda.mem_get_info();assert free>=8*1024**3,'8GiB minimum headroom'
   result['runtime']={'gpu':torch.cuda.get_device_name(),'uuid':str(torch.cuda.get_device_properties(0).uuid),'free_bytes':free,'total_bytes':total,'torch':torch.__version__,'cuda':torch.version.cuda}
   for runner in runners.values():
    for ids in inputs:runner.stage(ids);runner()
   torch.cuda.synchronize();emit('timing',inputs=len(inputs),passes=passes)
   samples=bootstrap.replay.dense.paired_probe({k:r for k,r in runners.items() if k.endswith('_graph')},inputs,passes=passes,seed=cfg['timing_seed']+a.replicate-1)
   timing=bootstrap.replay.dense.timing_summary(samples,reference='native_graph')
   for name,row in timing.items():
    vals=[r['host_ms'] for r in samples if r['mode']==name]
    row['geomean_host_ms']=math.exp(math.fsum(map(math.log,vals))/len(vals))
   write(dest/'timing.json',{'indices':indices.tolist(),'samples':samples,'summary':timing});result['timing']=timing
   valid_start=time.monotonic();emit('validation' if a.phase=='final' else 'training_numerics',target_blocks=blocks)
   def progress(**fields):
    completed=fields['blocks'];elapsed=time.monotonic()-valid_start
    emit('validation' if a.phase=='final' else 'training_numerics',**fields,blocks_per_second=completed/elapsed,remaining_seconds=(blocks-completed)*elapsed/completed)
   stream=(torch.tensor(data[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in range(blocks))
   quality=bootstrap.replay.dense.compare_inputs(runners,stream,cfg['numerical_bounds'],progress=progress)
   quality.update(blocks=blocks,documents=500 if a.phase=='final' else None,input_tokens=blocks*2048,excluded_tail_tokens=1444 if a.phase=='final' else 0)
   if a.phase=='training':
    quality['split_gate_pass']={split:{name:all(row['pass'] for row in rows[lo:hi]) for name,rows in quality['gates'].items()} for split,lo,hi in [('development',0,64),('confirmation',64,128)]}
   write(dest/'quality.json',quality)
   result.update(status='complete',qualification=quality['pass'],loss=quality['loss'],loss_delta=quality['loss_delta'],validation_blocks=blocks,peak_allocated_bytes=torch.cuda.max_memory_allocated())
   emit('complete',loss=quality['loss'],qualified=quality['pass'],timing_ms={k:v['geomean_host_ms'] for k,v in timing.items()})
 except Exception as exc:
  result.update(status='failed',error=str(exc),traceback=traceback.format_exc());emit('failed',error=str(exc))
 finally:
  result['elapsed_seconds']=time.monotonic()-started;write(dest/'result.json',result)
 return 0 if result['status']=='complete' else 1

if __name__=='__main__':raise SystemExit(main())
