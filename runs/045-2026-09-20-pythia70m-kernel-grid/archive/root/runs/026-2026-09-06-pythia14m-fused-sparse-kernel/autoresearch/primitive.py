"""GPU numerical and replay tests of all six K017 launch configurations."""
import torch
from common import HERE, RUN, module, write_json, numerical_gate


def main():
    candidate = module('primitive_k017', HERE/'candidates/k017/candidate.py')
    torch.manual_seed(2601)
    results = []
    with torch.inference_mode():
        for elements in (2, 4, 8):
            for warps in (4, 8):
                for k, n in ((128,128),(128,384),(128,512),(512,128),(33,129)):
                    layer = torch.nn.Linear(k,n).to('cuda',dtype=torch.bfloat16).eval()
                    fused = candidate.GatedLinear(layer,.5,elements,warps).eval()
                    x = torch.randn(7,k,device='cuda',dtype=torch.bfloat16)
                    x[:,0] = .5
                    fused(x)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = fused(x)
                    for pattern in ('random','zero','equal','signed'):
                        if pattern == 'zero': x.zero_()
                        if pattern == 'equal': x.fill_(.5)
                        if pattern == 'signed': x.normal_()
                        graph.replay()
                        expected = (x.masked_fill(x < .5,0).float() @ layer.weight.float().T + layer.bias.float()).bfloat16()
                        gate = numerical_gate(expected,actual,relative_l2=.02,atol=.125,rtol=.02)
                        results.append(dict(elements=elements,warps=warps,k=k,n=n,pattern=pattern,**gate))
    write_json(RUN/'runtime/primitive.json', {'cases':results,'pass':all(r['pass'] for r in results)})
    print(f"{sum(r['pass'] for r in results)}/{len(results)} CUDA gate/replay cases passed",flush=True)
    if not all(r['pass'] for r in results): raise RuntimeError('Primitive failure')


if __name__ == '__main__': main()
