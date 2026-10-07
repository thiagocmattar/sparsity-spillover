"""Qualification of corner-specialized projection, counters and input replay."""
import sys
import time
from io_utils import RUN,ARCHIVE,write,record
sys.path.insert(0,str(ARCHIVE/'src'))
import torch
from kernel import extension,gated_value

def counts(x,skip):
    m,k=x.shape;potential=m//8*(k//16)*16
    live=(x!=0).reshape(m//8,8,k//16,16).any(1).any(-1)
    issued=int(live.sum())*16 if skip else potential
    return [issued,potential-issued,0]

def main():
    started=time.monotonic();torch.manual_seed(2801);checks=[]
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    generic=extension(False);special=extension(True)
    with torch.inference_mode():
        for kappa in [0.,.1]:
            threshold=float(torch.tensor(kappa,dtype=torch.bfloat16))
            for case in ['random','dead','live_signed','two_values','equality']:
                h=torch.randn((32,512),device='cuda',dtype=torch.bfloat16)
                z=torch.randn((32,128),device='cuda',dtype=torch.bfloat16)
                for x in [h,z]:
                    if case=='dead':x[::8,::16]=-1
                    if case=='live_signed':x[::8,::16]=1;x[:,1::16]=-3
                    if case=='two_values':
                        x.zero_();x[::8,0]=1;x[::8,1]=-3
                    if case=='equality':
                        x[::8,::16]=threshold
                        x[::8,0]=torch.nextafter(torch.tensor(threshold,device='cuda',dtype=torch.bfloat16),torch.tensor(float('inf'),device='cuda',dtype=torch.bfloat16))
                wh=torch.randn((128,512),device='cuda',dtype=torch.bfloat16)*.01
                wz=torch.randn((128,128),device='cuda',dtype=torch.bfloat16)*.01
                bh=torch.randn(128,device='cuda',dtype=torch.bfloat16)*.01
                bz=torch.randn(128,device='cuda',dtype=torch.bfloat16)*.01
                residual=torch.randn((32,128),device='cuda',dtype=torch.bfloat16)*.1
                hg=gated_value(h.reshape(2,16,512),2,threshold).reshape_as(h)
                zg=gated_value(z.reshape(2,16,128),2,threshold).reshape_as(z)
                expected=(torch.nn.functional.linear(hg,wh,bh)+torch.nn.functional.linear(zg,wz,bz))+residual
                for skip in [False,True]:
                    out=torch.empty_like(residual);reference=torch.empty_like(residual)
                    stats=torch.empty((4,8,6),device='cuda',dtype=torch.int64)
                    args=(h,z,wh,wz,wh.t().contiguous(),wz.t().contiguous(),bh,bz,residual)
                    def call(ext,target,count=False):ext.forward(*args,target,stats,threshold,threshold,2,2,skip,True,count)
                    call(special,out,True);call(generic,reference)
                    if not torch.allclose(out.float(),expected.float(),atol=.01,rtol=.02):raise ValueError('Independent branch/residual mismatch')
                    if not torch.allclose(out.float(),reference.float(),atol=.01,rtol=.02):raise ValueError('Generic/specialized mismatch')
                    actual=stats.sum((0,1)).cpu().tolist()
                    if actual[:2]+[actual[4]]!=counts(hg,skip) or actual[2:4]+[actual[5]]!=counts(zg,skip):raise ValueError('Independent integer count mismatch')
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):call(special,out)
                    h.neg_();z.neg_();graph.replay();call(generic,reference)
                    if not torch.allclose(out.float(),reference.float(),atol=.01,rtol=.02):raise ValueError('Changed-input mask/graph mismatch')
                    # Restore inputs before the next paired control.
                    h.neg_();z.neg_()
                    checks.append({'kappa':kappa,'case':case,'skip':skip,'counts':actual,'changed_input_passed':True})
    import inspection
    ext=inspection.extension()
    for k in [128,512]:
        x=torch.randn(2048,k,device='cuda',dtype=torch.bfloat16)
        a=torch.empty(2048,device='cuda',dtype=torch.int32);b=torch.empty_like(a)
        ext.inspect_only(x,a,0.,True);ext.inspect_only(x,b,0.,False)
        pooled=a.reshape(-1,8).to(torch.int64)[:,0].clone()
        for j in range(1,8):pooled.bitwise_or_(a.reshape(-1,8)[:,j].to(torch.int64))
        if not torch.equal(b.reshape(-1,8).to(torch.int64),pooled[:,None].expand(-1,8)):raise ValueError('Inspection OR mismatch')
        checks.append({'kind':'standalone_inspection','width':k,'passed':True})
    torch.cuda.synchronize()
    write(RUN/'prelaunch/component-cuda.json',{'status':'passed','checks':checks,'gpu':torch.cuda.get_device_name(),
        'torch':torch.__version__,'cuda':torch.version.cuda,'seconds':time.monotonic()-started,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'sources':[record(RUN/n) for n in ['joint.cu','generic.cu','inspection.cu']],
        'scope':'synthetic development qualification; not full validation or accepted benchmark latency'})
    print('Passed',len(checks),'CUDA checks')

if __name__=='__main__':main()
