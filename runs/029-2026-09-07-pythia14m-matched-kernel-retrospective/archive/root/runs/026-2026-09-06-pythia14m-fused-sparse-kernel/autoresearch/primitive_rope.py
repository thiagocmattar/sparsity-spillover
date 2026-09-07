"""Exact bitwise fused RoPE/gate checks, including repeated graph buffers."""
import torch
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
from common import HERE,RUN,module,write_json


def main():
    candidate=module('primitive_k019',HERE/'candidates/k019/candidate.py')
    torch.manual_seed(2619)
    rows=[]
    with torch.inference_mode():
        for b,t,h,d,r in [(1,2048,4,32,8),(2,7,3,8,4)]:
            x=torch.randn(b,t,h*3*d,device='cuda',dtype=torch.bfloat16)
            c=torch.randn(b,t,r,device='cuda',dtype=torch.bfloat16)
            s=torch.randn_like(c)
            outputs=[torch.empty(b,h,t,d,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
            fn=lambda:candidate.extension().forward(x,c,s,*outputs,h,d,.5,.5,.5)
            fn()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): fn()
            for pattern in ['random','zero','equal','negative','changed']:
                if pattern=='zero': x.zero_()
                if pattern=='equal': x.fill_(.5); c.fill_(1); s.zero_()
                if pattern=='negative': x.fill_(-.5)
                if pattern=='changed': x.normal_(); c.normal_(); s.normal_()
                graph.replay()
                q,k,v=x.view(b,t,h,3*d).transpose(1,2).chunk(3,dim=-1)
                q,k=apply_rotary_pos_emb(q,k,c,s)
                expected=[value.masked_fill(value.abs()<.5,0) for value in [q,k,v]]
                equal=[torch.equal(a,e) for a,e in zip(outputs,expected)]
                rows.append(dict(shape=[b,t,h,d,r],pattern=pattern,exact_qkv=equal,passed=all(equal)))
    write_json(RUN/'runtime/primitive-rope.json',{'cases':rows,'pass':all(r['passed'] for r in rows)})
    print(f"K019 exact {sum(r['passed'] for r in rows)}/{len(rows)} passed",flush=True)
    if not all(r['passed'] for r in rows): raise RuntimeError('Exact RoPE gate failed')


if __name__=='__main__':main()
