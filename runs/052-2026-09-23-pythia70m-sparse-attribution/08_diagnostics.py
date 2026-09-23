"""Retain full-validation diagnostics and separate profiling after policy freeze."""
import argparse,time,traceback,gc
from support import RUN,BASE,read,write,event,sha,source_hashes

def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--selection',required=True);p.add_argument('--condition',required=True,choices=('c00','c24','c25'));p.add_argument('--phase',choices=('smoke','final'),default='final');a=p.parse_args()
 dest=RUN/'artifacts'/a.attempt;dest.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 import bootstrap
 bootstrap.setup()
 import torch,numpy as np,adapter
 from structure import Structure,gated
 from work_counters import Work
 # Archived imports also expose modules called diagnostics/profiling. Bind the
 # declared Run049 helpers explicitly rather than trusting sys.path precedence.
 for helper in ('frozen_diagnostics','tile_oracle','parallel_work_oracle'):
  bootstrap.bind(helper,BASE/(helper+'.py'))
 collect=bootstrap.bind('run050_retained_diagnostics',BASE/'diagnostics.py').collect
 profile=bootstrap.bind('run050_retained_profiling',BASE/'profiling.py').collect
 selection=read(RUN/'provenance'/a.selection)
 if a.phase=='final':
  for name,digest in selection['kernel_source_hashes'].items():assert sha(RUN/name)==digest
  manifest=read(BASE/'provenance/inputs.json')['validation'];path=bootstrap.base_io.verify(manifest);blocks=338
 else:
  manifest=read(RUN/'provenance/inputs.json')['training'];path=RUN/manifest['path'];assert sha(path)==manifest['sha256'];blocks=4
 data=np.memmap(path,dtype=np.int32,mode='r')
 if a.phase=='final':assert divmod(len(data),2048)==(338,1444)
 write(dest/'source-freeze.json',{'selection':sha(RUN/'provenance'/a.selection),'sources':source_hashes(),'validation':manifest})
 native=bootstrap.model(a.condition,custom=False)
 policies={'dense_policy':selection['dense'],**selection['policies'],**selection.get('ablations',{})}
 if 'qualified_modes' in selection:policies={k:v for k,v in policies.items() if k in selection['qualified_modes']}
 if a.condition=='c00':policies={'dense_policy':selection['dense']}
 finished=[]
 for name,policy in policies.items():
  def emit(stage,**fields):event(dest,stage,condition=a.condition,implementation=name,elapsed_seconds=time.monotonic()-start,**fields)
  emit('loading')
  with torch.inference_mode():
   model=bootstrap.model(a.condition,custom=False);adapter.install(model,policy)
   structure=Structure();work=Work();originals=[]
   class Observe:
    def __init__(self,op,index):self.op,self.index=op,index
    def __getattr__(self,key):return getattr(self.op,key)
    def __call__(self,h,z,residual):
     output=self.op(h,z,residual)
     for site,x,threshold in [('h',h,self.op.th if self.op.gh else None),('z',z,self.op.tz if self.op.gz else None)]:
      x=gated(x.reshape(2048,-1),threshold);key=f'{site}.{self.index}'
      structure.add(key,x);work.add(key,getattr(self.op,site,None),x)
     return output
   for i,layer in enumerate(model.gpt_neox.layers):
    op=layer._run026_joint;originals.append((layer,op));layer._run026_joint=Observe(op,i)
   try:collect(model,native,data,dest/f'diagnostics-{name}.json',emit,bootstrap.checkpoint(a.condition)['canonical_logical_products']['architecture_maximum'],blocks=blocks)
   finally:
    for layer,op in originals:layer._run026_joint=op
   compiler={};compiler_dir=dest/('compiler-'+name);compiler_dir.mkdir(exist_ok=True)
   for i,layer in enumerate(model.gpt_neox.layers):
    for site in ('h','z'):
     op=getattr(layer._run026_joint,site,None)
     if op is None:continue
     if op.spec['family'] in ('e','f'):
      if not (compiler_dir/'manifest.json').exists():
       from compiler_evidence import retain
       retain(op.ext,compiler_dir)
      compiler[f'{site}-{i}-cuda']={'backend':op.spec,'extension_evidence':read(compiler_dir/'manifest.json'),
                                  'registers_and_spills':'see CUDA resource report; not inferred from logical counters'}
     for j,kernel in enumerate(op.compiled):
      stem=f'{site}-{i}-{j}';metadata=getattr(kernel,'metadata',None)
      compiler[stem]={'backend':op.spec,'registers':getattr(kernel,'n_regs',None),'spills':getattr(kernel,'n_spills',None),'metadata':str(metadata)}
      for kind in ('ptx','cubin'):
       if kind in kernel.asm:
        code=kernel.asm[kind];path=compiler_dir/(stem+'.'+kind)
        if isinstance(code,bytes):path.write_bytes(code)
        else:path.write_text(code)
   write(dest/f'compiler-{name}.json',compiler)
   write(dest/f'structure-{name}.json',{'coverage':{'blocks':blocks,'documents':500 if blocks==338 else None,'input_tokens':blocks*2048,'excluded_tail_tokens':1444 if blocks==338 else 0},'per_site_layer':structure.result()})
   write(dest/f'work-{name}.json',work.result())
   inputs=[torch.tensor(data[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in range(4)]
   runner=bootstrap.replay.dense.DenseRunner(lambda x,m=model:bootstrap.replay.scaffold.forward(m,x),inputs[0].clone(),'graph');runner.prepare()
   emit('profiling');profile({name:model},{name+'_graph':runner},inputs,dest)
   # profile() uses a fixed summary filename; preserve every mode's summary.
   (dest/'profile-summary.json').rename(dest/f'profile-summary-{name}.json')
   finished.append(name);write(dest/'completed.json',finished)
   del runner,model,structure,work,originals,inputs;gc.collect();torch.cuda.empty_cache()
 write(dest/'result.json',{'status':'complete','condition':a.condition,'implementations':finished,'elapsed_seconds':time.monotonic()-start})
 event(dest,'complete',elapsed_seconds=time.monotonic()-start)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  import sys
  attempt=sys.argv[sys.argv.index('--attempt')+1]
  write(RUN/'artifacts'/attempt/'result.json',{'status':'failed','error':str(exc),'traceback':traceback.format_exc()})
  raise
