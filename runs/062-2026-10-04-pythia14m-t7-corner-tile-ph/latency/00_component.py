"""Bounded CUDA qualification of gates, residual rounding and input replay."""
import sys
import time
from io_utils import RUN, ARCHIVE, write, record
sys.path.insert(0,str(ARCHIVE/'src'))
import torch
from kernel import extension,gated_value


def hybrid_expected(value,skip):
    x=value.reshape(-1,value.shape[-1]);m,k=x.shape
    potential=(m//8)*(k//16)*16
    if not skip:
        return [potential,0,0]
    live=x!=0;nnz=live.sum(-1)
    simple=(nnz<=2)&((x==0)|((x.abs()>=2.**-50)&(x.abs()<=2.**50))).all(-1)
    active=(live&~simple[:,None]).reshape(m//8,8,k//16,16).any(1).any(-1)
    issued=int(active.sum())*16
    return [issued,potential-issued,int(nnz[simple].sum())*128]


def main():
    started=time.monotonic();checks=[]
    torch.manual_seed(2801)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    ext=extension()
    with torch.inference_mode():
        from port_gates import extension as port_extension
        from tile_gate import CornerTileThreshold
        for rank in [3,4]:
            for kappa in [0.,.1]:
                threshold=float(torch.tensor(kappa,dtype=torch.bfloat16))
                shape=(2,16,32) if rank==3 else (2,4,16,32)
                values=[torch.randn(shape,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
                for x in values:
                    x[...,::8,::16]=threshold
                    x[...,0,0]=-1
                    x[...,0,1]=-3
                    x[...,8,16]=-threshold
                for modes in [[1,1,1],[3,3,3],[2,2,2],[4,4,4]]:
                    outputs=[torch.empty_like(x) for x in values]
                    port_extension().forward(values,outputs,modes,[threshold]*3)
                    def expected(x,mode):
                        if mode==1:return x.masked_fill(x<threshold,0)
                        if mode==3:return x.masked_fill(x.abs()<threshold,0)
                        return CornerTileThreshold(threshold,comparison='magnitude' if mode==4 else 'signed')(x)
                    assert all(torch.equal(y,expected(x,m)) for x,y,m in zip(values,outputs,modes))
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        port_extension().forward(values,outputs,modes,[threshold]*3)
                    for x in values:x.neg_()
                    graph.replay()
                    assert all(torch.equal(y,expected(x,m)) for x,y,m in zip(values,outputs,modes))
                    checks.append({'kind':'exact_port_gate_and_changed_input_graph','rank':rank,'mode':modes[0],'kappa':kappa,'exact':True})
        for k in [128,512]:
            for mode in [1,2]:
                for kappa in [0.,.1]:
                    threshold=float(torch.tensor(kappa,dtype=torch.bfloat16))
                    x=torch.randn((2,16,k),device='cuda',dtype=torch.bfloat16)
                    x[:,::8,::16]=threshold
                    x[:,0,0]=threshold+1
                    x[:,0,1]=-3
                    x[:,0,16]=torch.nextafter(torch.tensor(threshold,device='cuda',dtype=torch.bfloat16),torch.tensor(float('inf'),device='cuda',dtype=torch.bfloat16))
                    output=torch.empty_like(x)
                    ext.gate_values(x.reshape(-1,k),output.reshape(-1,k),threshold,mode)
                    expected=gated_value(x,mode,threshold)
                    if not torch.equal(output,expected):
                        raise ValueError('Gate differs at equality, signed interior or batch boundary')
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        ext.gate_values(x.reshape(-1,k),output.reshape(-1,k),threshold,mode)
                    x.neg_();graph.replay()
                    if not torch.equal(output,gated_value(x,mode,threshold)):
                        raise ValueError('Changed-input graph reused an old mask')
                    checks.append({'kind':'gate_and_changed_input_graph','width':k,'mode':mode,'kappa':kappa,'exact':True})
        for mode in [1,2]:
            for kappa in [0.,.1]:
                threshold=float(torch.tensor(kappa,dtype=torch.bfloat16))
                h=torch.randn((2,16,512),device='cuda',dtype=torch.bfloat16)
                z=torch.randn((2,16,128),device='cuda',dtype=torch.bfloat16)
                # One all-dead temporal block per batch; a surviving tile with negative interior.
                h[:,0,::16]=-1;z[:,0,::16]=-1
                h[:,8,0]=1;h[:,8,1]=-3;z[:,8,0]=1;z[:,8,1]=-3
                h[:,8,16:]=0;z[:,8,16:]=0
                wh=torch.randn((128,512),device='cuda',dtype=torch.bfloat16)*.01
                wz=torch.randn((128,128),device='cuda',dtype=torch.bfloat16)*.01
                bh=torch.randn(128,device='cuda',dtype=torch.bfloat16)*.01
                bz=torch.randn(128,device='cuda',dtype=torch.bfloat16)*.01
                residual=torch.randn((2,16,128),device='cuda',dtype=torch.bfloat16)*.1
                hg=gated_value(h,mode,threshold);zg=gated_value(z,mode,threshold)
                expected=(torch.nn.functional.linear(hg,wh,bh)+torch.nn.functional.linear(zg,wz,bz))+residual
                for skip in [False,True]:
                    output=torch.empty_like(residual);stats=torch.empty((4,8,6),device='cuda',dtype=torch.int64)
                    ext.forward(h.reshape(-1,512),z.reshape(-1,128),wh,wz,wh.t().contiguous(),wz.t().contiguous(),
                                bh,bz,residual.reshape(-1,128),output.reshape(-1,128),stats,
                                threshold,threshold,mode,mode,skip,True,True)
                    error=float((output.float()-expected.float()).abs().max())
                    if not torch.allclose(output.float(),expected.float(),atol=.01,rtol=.02):
                        raise ValueError(f'Projection/residual mismatch: {error}')
                    counts=stats.sum((0,1)).cpu().tolist()
                    if counts[:2]+[counts[4]] != hybrid_expected(hg,skip) or counts[2:4]+[counts[5]] != hybrid_expected(zg,skip):
                        raise ValueError('Instrumented counts differ from independent operands')
                    checks.append({'kind':'joint_projection_and_counters','mode':mode,'kappa':kappa,'skip':skip,'max_abs':error,'counts':counts})
    torch.cuda.synchronize()
    write(RUN/'prelaunch/component-cuda.json',{'status':'passed','checks':checks,'gpu':torch.cuda.get_device_name(),
        'torch':torch.__version__,'cuda':torch.version.cuda,'seconds':time.monotonic()-started,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'source':record(RUN/'joint.cu'),
        'scope':'synthetic component qualification only; not full-model correctness or accepted benchmark latency'})
    print(f'Passed {len(checks)} CUDA checks in {time.monotonic()-started:.2f}s')


if __name__=='__main__':
    main()
