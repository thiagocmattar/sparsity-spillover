"""CUDA checks for the new D256/K1024/K256 shapes and mask-word tail."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from run_config import write_json
import port
from tile_oracle import counts
from parallel_work_oracle import extra_scalar_products


def main():
    torch.manual_seed(2801)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    joint = port.module("run054_component_joint",port.HERE/"kernel/joint/joint.py")
    rows = []
    with torch.inference_mode():
        w2 = torch.nn.Linear(1024,256,device="cuda",dtype=torch.bfloat16)
        wo = torch.nn.Linear(256,256,device="cuda",dtype=torch.bfloat16)
        residual = torch.randn(16,256,device="cuda",dtype=torch.bfloat16)
        for kind in ("zero", "short", "mixed", "dense", "last_z_tile"):
            h = torch.zeros(16,1024,device="cuda",dtype=torch.bfloat16)
            z = torch.zeros(16,256,device="cuda",dtype=torch.bfloat16)
            if kind in ("short", "mixed"):
                h[:,:3] = .5
                z[:,253:] = .5
            if kind == "mixed":
                h[0] = torch.randn_like(h[0])
            if kind == "dense":
                h.normal_()
                z.normal_()
            if kind == "last_z_tile":
                z[:,240:] = .5  # z is only half a 32-bit K16 mask word.
            for skip in (False,True):
                op = joint.Joint(SimpleNamespace(w2=w2,wo=wo,gh=True,gz=True,th=.1,tz=.1,skip=skip))
                op.count = True
                actual = op(h,z,residual)
                ah, az = h.masked_fill(h<.1,0),z.masked_fill(z<.1,0)
                expected = (torch.nn.functional.linear(ah,w2.weight,w2.bias)+torch.nn.functional.linear(az,wo.weight,wo.bias))+residual
                difference = (actual.float()-expected.float()).abs()
                if not torch.all(difference <= .02+.02*expected.float().abs()):
                    raise RuntimeError(f"Joint arithmetic failed {kind}/{skip}")
                stat = op.stats.sum((0,1,2)).cpu().tolist()
                extra = extra_scalar_products(ah,az,skip=skip,fast_weights=op.fast_weights)
                for i,value in enumerate((ah,az)):
                    oracle = counts(value,8,op.fast_weights,skip,8)
                    oracle[2] += extra[i]
                    if oracle != [stat[2*i],stat[2*i+1],stat[4+i]]:
                        raise RuntimeError(f"Counters failed {kind}/{skip}: {oracle}, {stat}")
                rows.append(dict(kind=kind,skip=skip,max_abs_error=float(difference.max()),counts=stat))
    result = dict(status="passed",gpu=torch.cuda.get_device_name(),cases=rows)
    write_json(port.HERE/"component-check.json",result)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
