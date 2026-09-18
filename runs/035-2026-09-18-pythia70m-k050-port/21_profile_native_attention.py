"""Record actual pinned native SDPA kernel dispatch for the 70M attention shape."""
import json
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent

def main():
    torch.manual_seed(2801)
    with torch.inference_mode():
        q,k,v=[torch.randn(1,8,2048,64,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        for _ in range(5):torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
            torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
            torch.cuda.synchronize()
        path=HERE/'artifacts/native-attention-trace.json';p.export_chrome_trace(str(path))
    events=json.loads(path.read_text())['traceEvents']
    kernels=[{'name':e['name'],'args':e.get('args')} for e in events if e.get('cat')=='kernel']
    (HERE/'artifacts/native-attention-dispatch.json').write_text(json.dumps(kernels,indent=2)+'\n')
    print(json.dumps(kernels,indent=2))

if __name__=='__main__':main()
