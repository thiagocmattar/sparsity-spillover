import json
from pathlib import Path
import sys
import numpy as np
import torch

root = Path('/workspace/run031')
run = root/'runs/031-2026-09-08-signed-activation-density'
sys.path[:0] = [str(run),str(root/'src')]
from density import SignedHistogram, edges, write_json
from sparsity_research.metrics import ActivationAccumulator

checks = []
for dtype in (torch.float16,torch.float32):
    x = torch.tensor([-9,-8,-.5,-.05,-.001,-0.,0.,.001,.05,.5,8,9],dtype=dtype,device='cuda')
    values = {'h.layer_0':x}
    moments = ActivationAccumulator()
    hist = SignedHistogram(torch=torch,device='cuda')
    moments.update(values,torch=torch)
    hist.update(values)
    row = hist.rows(moments.rows())[0]
    cpu = x.float().cpu().numpy()
    inside = cpu[(cpu!=0)&(cpu>=-8)&(cpu<=8)]
    expected,_ = np.histogram(inside,bins=edges())
    assert row['histogram']==expected.tolist()
    assert row['exact_zero_count']==2 and row['underflow']==row['overflow']==1
    checks.append({'dtype':str(dtype),'passed':True})
result = {'checks':checks,'torch':torch.__version__,'cuda':torch.version.cuda,
          'gpu':torch.cuda.get_device_name(),'total_gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory}
write_json(run/'artifacts/cuda-boundary-checks.json',result)
print(json.dumps(result))
