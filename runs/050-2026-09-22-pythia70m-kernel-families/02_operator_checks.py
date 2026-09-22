"""GPU tests of all bounded families, retaining unsupported and failed trials."""
import argparse,time,traceback
from support import RUN,write,event,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--families',default='dense,a,b,c,d');p.add_argument('--stage',choices=('initial','refinement','ablation','all'),default='initial');a=p.parse_args()
 dest=RUN/'artifacts'/a.attempt;dest.mkdir(parents=True,exist_ok=False)
 write(dest/'source-freeze.json',{'files':{p.name:sha(p) for p in list(RUN.glob('*.py'))+list(RUN.glob('*.cu'))}})
 import bootstrap
 cfg=bootstrap.setup();bootstrap.verify_baseline()
 import torch
 import primitives as ops
 results=[];start=time.monotonic()
 with torch.inference_mode():
  specs=[s for s in ops.candidates(include_ablations=True) if s['family'] in a.families.split(',') and (a.stage=='all' or s['family']=='dense' or s.get('stage','initial')==a.stage)]
  for s in specs:
   for k in (512,2048):
    event(dest,'operator',candidate=s['id'],k=k,elapsed_seconds=time.monotonic()-start)
    row={'candidate':s,'k':k,'cases':[]}
    try:
     linear=torch.nn.Linear(k,512,device='cuda',dtype=torch.bfloat16)
     op=ops.Linear(linear,.1,s,m=256)
     for case in ('zero','dense','sparse','boundary','overflow'):
      x=torch.randn((256,k),device='cuda',dtype=torch.bfloat16)*.2
      if case=='zero':x.zero_()
      if case=='sparse':x.masked_fill_(torch.rand_like(x.float())<.98,0)
      if case=='boundary':x.fill_(float(torch.tensor(.1,dtype=torch.bfloat16)))
      if case=='overflow':x.fill_(.5);x[:,1::4]=-.2
      expected=torch.nn.functional.linear(x.masked_fill(x<.1,0),linear.weight,linear.bias)
      actual=op(x).clone();torch.cuda.synchronize()
      error=(actual.float()-expected.float()).abs()
      record={'case':case,'max_abs':float(error.max()),'bound_violations':int((error>.25+.02*expected.float().abs()).sum()),
              'relative_l2':float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(expected.float()).clamp_min(1e-20))}
      record['pass']=bool(torch.isfinite(actual).all()) and record['bound_violations']==0 and record['relative_l2']<=.02
      if s['family']=='a':
       gated=x.masked_fill(x<.1,0);count=(gated!=0).reshape(256,k//256,256).sum(-1)
       assert torch.equal(op.count,count.int())
       decoded=torch.zeros_like(gated)
       packed=op.p.reshape(256,k//256,256).long()
       valid=torch.arange(256,device='cuda')[None,None,:]<op.count[:,:,None]
       rr=torch.arange(256,device='cuda')[:,None,None].expand_as(packed)[valid]
       idx=(packed&65535)[valid]
       val=(packed>>16).to(torch.int16).view(torch.bfloat16)[valid]
       decoded[rr,idx]=val
       assert torch.equal(decoded,gated),'Lossy packing'
      if s['family']=='d':
       gated=x.masked_fill(x<.1,0)
       first=gated-op.e
       assert int((first.reshape(256,-1,4)!=0).sum(-1).max())<=2
       assert torch.equal(first+op.e,gated)
       sparse_expected=first.float()@linear.weight.float().t()
       assert torch.allclose(op.y,sparse_expected,atol=.002,rtol=.002),'Sparse metadata or FP32 output incorrect'
      row['cases'].append(record)
     row['pass']=all(c['pass'] for c in row['cases'])
     # Dynamic inputs inside capture; conversion is not precomputed outside it.
     sample=torch.randn((256,k),device='cuda',dtype=torch.bfloat16)
     op(sample);torch.cuda.synchronize()
     graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph): captured=op(sample)
     sample.zero_();graph.replay();torch.cuda.synchronize()
     assert torch.equal(captured,linear.bias.expand_as(captured)),'Graph input change not observed'
     row['graph_changed_input_pass']=True
     row['compiler']=[];compiler=dest/'compiler';compiler.mkdir(exist_ok=True)
     for i,kernel in enumerate(op.compiled):
      stem=f"{s['id']}-K{k}-{i}"
      row['compiler'].append({'stem':stem,'registers':getattr(kernel,'n_regs',None),'spills':getattr(kernel,'n_spills',None),'metadata':str(getattr(kernel,'metadata',None))})
      for kind in ('ptx','cubin'):
       if kind in kernel.asm:
        code=kernel.asm[kind];path=compiler/(stem+'.'+kind)
        if isinstance(code,bytes):path.write_bytes(code)
        else:path.write_text(code)
    except Exception as exc:
     row.update(pass_=False,error=str(exc),traceback=traceback.format_exc());row['pass']=False
    results.append(row);write(dest/'operators.json',results)
  eligible=[s['id'] for s in specs if all(r['pass'] for r in results if r['candidate']['id']==s['id'])]
  write(dest/'result.json',{'status':'complete','eligible':eligible,'results':results,'elapsed_seconds':time.monotonic()-start,
                          'source_hashes':{p.name:sha(p) for p in list(RUN.glob('*.py'))+list(RUN.glob('*.cu'))}})
  event(dest,'complete',eligible=eligible,elapsed_seconds=time.monotonic()-start)

if __name__=='__main__':main()
