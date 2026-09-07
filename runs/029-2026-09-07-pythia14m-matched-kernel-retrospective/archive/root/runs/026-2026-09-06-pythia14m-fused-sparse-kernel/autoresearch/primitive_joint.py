"""K018 GPU tests include two BF16 additions and rotating graph inputs."""
import torch
from common import HERE,RUN,module,write_json,numerical_gate


def main():
    candidate=module('primitive_k018',HERE/'candidates/k018/candidate.py')
    torch.manual_seed(2618)
    rows=[]
    with torch.inference_mode():
        for kh,kz,n in [(512,128,128),(33,17,129)]:
            for m in [3,2048]:
                h=torch.randn(m,kh,device='cuda',dtype=torch.bfloat16)
                z=torch.randn(m,kz,device='cuda',dtype=torch.bfloat16)
                wh=(torch.randn(kh,n,device='cuda')*.1).bfloat16()
                wz=(torch.randn(kz,n,device='cuda')*.1).bfloat16()
                bh=torch.randn(n,device='cuda',dtype=torch.bfloat16)
                bz=torch.randn(n,device='cuda',dtype=torch.bfloat16)
                r=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)
                out=torch.empty_like(r)
                fn=lambda: candidate.extension().forward(h,z,wh,wz,bh,bz,r,out,.5,.5)
                fn()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): fn()
                for pattern in ['random','zero','equal','changed']:
                    if pattern=='zero': h.zero_(); z.zero_()
                    if pattern=='equal': h.fill_(.5); z.fill_(.5)
                    if pattern=='changed': h.normal_(); z.normal_(); r.normal_()
                    graph.replay()
                    a=(h.masked_fill(h<.5,0).float()@wh.float()+bh.float()).bfloat16()
                    b=(z.masked_fill(z<.5,0).float()@wz.float()+bz.float()).bfloat16()
                    expected=(a+b)+r
                    gate=numerical_gate(expected,out,relative_l2=.02,atol=.125,rtol=.02)
                    rows.append(dict(kh=kh,kz=kz,n=n,m=m,pattern=pattern,**gate))
    write_json(RUN/'runtime/primitive-joint.json',{'cases':rows,'pass':all(r['pass'] for r in rows)})
    print(f"K018 {sum(r['pass'] for r in rows)}/{len(rows)} passed",flush=True)
    if not all(r['pass'] for r in rows): raise RuntimeError('Primitive gate failed')


if __name__=='__main__': main()
