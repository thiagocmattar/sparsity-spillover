"""Matched component screen over all64+64 training blocks at both kappas."""
import argparse,json,time,math,traceback,gc
from collections import defaultdict
from support import RUN,BASE,read,write,event,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--operators',default='operators-002');p.add_argument('--families',default='dense,a,b,c,d');a=p.parse_args()
 dest=RUN/'artifacts'/a.attempt;dest.mkdir(parents=True,exist_ok=False)
 import bootstrap
 bootstrap.setup()
 import torch,numpy as np
 import primitives as ops
 from structure import Structure,gated
 cfg=read(RUN/'config.json');manifest=read(RUN/'provenance/inputs.json')
 path=RUN/manifest['training']['path'];assert sha(path)==manifest['training']['sha256']
 data=np.memmap(path,dtype=np.int32,mode='r').reshape(128,2048)
 eligible=read(RUN/'artifacts'/a.operators/'result.json')['eligible']
 specs=[s for s in ops.candidates() if s['id'] in eligible and s['family'] in a.families.split(',')]
 assert {'dense_native','dense_fused'}<=set(eligible)
 write(dest/'source-freeze.json',{'files':{p.name:sha(p) for p in list(RUN.glob('*.py'))+list(RUN.glob('*.cu'))},'candidates':specs,'inputs':manifest})
 aggregate=defaultdict(lambda:defaultdict(lambda:{'logs':[],'cuda_logs':[],'qualified':True,'max_abs':0.,'max_relative_l2':0.,'violating_elements':0}))
 start=time.monotonic();capture={};sample_path=dest/'timings.jsonl'
 for cid in ('c24','c25'):
  with torch.inference_mode():
   model=bootstrap.model(cid)
   class Observe:
    def __init__(self,op,i):self.op,self.i=op,i
    def __call__(self,h,z,r):
     capture[f'h.{self.i}']=h;capture[f'z.{self.i}']=z
     return self.op(h,z,r)
   old={}
   for i,layer in enumerate(model.gpt_neox.layers):
    old[i]=layer._run026_joint;layer._run026_joint=Observe(old[i],i)
   stats={'development':Structure(),'confirmation':Structure()};runners={};operators={}
   losses=0.;tokens=0
   for index in range(128):
    ids=torch.tensor(data[index].copy(),device='cuda',dtype=torch.long)[None]
    logits=bootstrap.replay.scaffold.forward(model,ids)
    losses+=float(torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,50304),ids[:,1:].reshape(-1),reduction='sum'))
    tokens+=2047;del logits
    split='development' if index<64 else 'confirmation'
    for key,raw in capture.items():
     site,layer=key.split('.');original=old[int(layer)];threshold=original.th if site=='h' else original.tz
     linear=original.w2 if site=='h' else original.wo
     x=raw.reshape(2048,-1);stats[split].add(key,gated(x,threshold))
     if key not in runners:
      group={};ops_group={}
      for spec in specs:
       name=spec['id']
       try:
        op=ops.Linear(linear,threshold,spec)
        runner=bootstrap.replay.dense.DenseRunner(op,x.clone(),'graph');runner.prepare()
        group[name]=runner;ops_group[name]=op
       except Exception as exc:
        with (dest/'capture-failures.jsonl').open('a') as f:f.write(json.dumps({'condition':cid,'site':key,'candidate':name,'error':str(exc),'traceback':traceback.format_exc()})+'\n')
      assert {'dense_native','dense_fused'}<=set(group)
      runners[key]=group;operators[key]=ops_group
     group=runners[key];reference=torch.nn.functional.linear(gated(x,threshold),linear.weight,linear.bias)
     for name,runner in group.items():
      runner.stage(x);actual=runner();err=(actual.float()-reference.float()).abs()
      rel=float(torch.linalg.vector_norm(err)/torch.linalg.vector_norm(reference.float()).clamp_min(1e-20))
      violation=int((err>.25+.02*reference.float().abs()).sum());finite=bool(torch.isfinite(actual).all())
      target=aggregate[f'{cid}:{split}:{key}'][name]
      target['qualified']&=finite and violation==0 and rel<=.02
      target['max_abs']=max(target['max_abs'],float(err.max()));target['max_relative_l2']=max(target['max_relative_l2'],rel);target['violating_elements']+=violation
     torch.cuda.synchronize()
     samples=bootstrap.replay.dense.paired_probe(group,[x],passes=cfg['development_passes'],seed=2504+index)
     with sample_path.open('a') as f:
      for sample in samples:
       sample.update(condition=cid,split=split,block=index,site=key)
       target=aggregate[f'{cid}:{split}:{key}'][sample['mode']]
       target['logs'].append(math.log(sample['host_ms']));target['cuda_logs'].append(math.log(sample['cuda_ms']))
       f.write(json.dumps(sample)+'\n')
    event(dest,'screen',condition=cid,blocks=index+1,target_blocks=128,loss=losses/tokens,
          blocks_per_second=(index+1)/max(time.monotonic()-start,.01),elapsed_seconds=time.monotonic()-start)
    if (index+1)%16==0:write(dest/'aggregate-progress.json',summarize(aggregate))
   write(dest/f'structure-{cid}.json',{k:v.result() for k,v in stats.items()})
   del model,runners,operators,old,stats,capture;capture={};gc.collect();torch.cuda.empty_cache()
 summary=summarize(aggregate)
 write(dest/'summary.json',summary)
 write(dest/'result.json',{'status':'complete','elapsed_seconds':time.monotonic()-start,'conditions':['c24','c25'],'training_blocks':128,'candidates':specs})
 event(dest,'complete',elapsed_seconds=time.monotonic()-start)

def summarize(aggregate):
 return {key:{name:{**{k:v for k,v in d.items() if k not in ('logs','cuda_logs')},
                    'observations':len(d['logs']),'host_ms':math.exp(math.fsum(d['logs'])/len(d['logs'])) if d['logs'] else None,
                    'cuda_ms':math.exp(math.fsum(d['cuda_logs'])/len(d['cuda_logs'])) if d['cuda_logs'] else None}
              for name,d in group.items()} for key,group in aggregate.items()}

if __name__=='__main__':main()
