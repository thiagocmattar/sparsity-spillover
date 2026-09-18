"""CUDA correctness and count checks before checkpoint benchmarking, not tuning."""
import json
from types import SimpleNamespace
import torch
import replay
from io_utils import RUN,module,write
from sparsity_research.sites import FixedOneSidedThreshold

def main():
    torch.manual_seed(2801)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    candidate=module('run035_operator_candidate',RUN/'kernel/candidate.py')
    results=[]
    def check(name,actual,expected,atol=0,rtol=0):
        delta=(actual.float()-expected.float()).abs()
        row={'name':name,'maximum_absolute_error':float(delta.max()),'atol':atol,'rtol':rtol}
        results.append(row);print(json.dumps(row),flush=True)
        torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
    with torch.inference_mode():
        for kappa in [0,.01,.05,.1,.5]:
            norms=[torch.nn.LayerNorm(512,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
            for n in norms:n.weight.copy_(torch.randn_like(n.weight));n.bias.copy_(torch.randn_like(n.bias))
            x=torch.randn(32,512,device='cuda',dtype=torch.bfloat16);x[0]=0;x[1]=2
            gates=[FixedOneSidedThreshold(kappa) for _ in range(2)]
            pair=candidate.norm.NormPair(*norms,*gates)
            check(f'norm_a_{kappa}',pair.start(x),gates[0](norms[0](x)))
            check(f'norm_m_{kappa}',pair.finish(x),gates[1](norms[1](x)))
        for n in [1536,2048]:
            linear=torch.nn.Linear(512,n,device='cuda',dtype=torch.bfloat16)
            x=torch.randn(32,512,device='cuda',dtype=torch.bfloat16);x[:16,:256]=0
            sparse=candidate.projection.Projection(linear,True)(x).clone()
            dense=candidate.projection.Projection(linear,False)(x).clone()
            check(f'projection_skip_{n}',sparse,dense)
            check(f'projection_native_{n}',sparse,linear(x),atol=.015625,rtol=.02)
        w2=torch.nn.Linear(2048,512,device='cuda',dtype=torch.bfloat16)
        wo=torch.nn.Linear(512,512,device='cuda',dtype=torch.bfloat16)
        for case in ['zero','short_high_index','complex_sparse','dense']:
            h=torch.zeros(32,2048,device='cuda',dtype=torch.bfloat16);z=torch.zeros(32,512,device='cuda',dtype=torch.bfloat16)
            if case=='short_high_index':h[:,[1025,2047]]=1;z[:,[400,511]]=1
            if case=='complex_sparse':h[:,[0,1025,2047]]=1;z[:,[0,400,511]]=1
            if case=='dense':h.normal_();z.normal_()
            r=torch.randn(32,512,device='cuda',dtype=torch.bfloat16)
            previous=SimpleNamespace(w2=w2,wo=wo,gh=False,gz=False,th=0.,tz=0.,skip=True)
            op=candidate.joint.Joint(previous);op.count=True
            sparse=op(h,z,r).clone();stat=op.stats.sum((0,1,2)).tolist()
            assert stat[0]+stat[1]==32768 and stat[2]+stat[3]==8192,stat
            op.skip=False;dense=op(h,z,r).clone()
            check(f'joint_skip_{case}',sparse,dense,atol=.015625,rtol=.02)
            check(f'joint_native_{case}',sparse,(w2(h)+wo(z))+r,atol=.03125,rtol=.02)
        for case in ['zero_q','dense','sparse']:
            q,k,v=[torch.randn(1,8,2048,64,device='cuda',dtype=torch.bfloat16)*.1 for _ in range(3)]
            if case=='zero_q':q.zero_()
            if case=='sparse':q[:,:,:1024]=0;k[:,:,:512]=0;v[:,:,:1024]=0
            op=candidate.attention.Attention(skip=True,shortcut=False)
            actual=op(q,k,v,.125,count=True).clone()
            stat=op.stats.sum((0,1,2,3)).tolist()
            assert stat[0]+stat[1]==589824 and stat[2]+stat[3]==589824,stat
            op.skip=False;dense=op(q,k,v,.125).clone()
            check(f'attention_skip_{case}',actual,dense)
            native=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,scale=.125)
            check(f'attention_native_{case}',actual,native,atol=.001,rtol=.02)
    write(RUN/'artifacts/operator-checks.json',{'status':'passed','checks':results,
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__})

if __name__=='__main__':main()
