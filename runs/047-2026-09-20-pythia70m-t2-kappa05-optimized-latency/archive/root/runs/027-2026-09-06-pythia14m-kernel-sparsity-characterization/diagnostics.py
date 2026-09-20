"""Untimed statistics on actual native operands, without replacing attention."""
import time
import torch
from sparsity_research.capture import ActivationCapture
from sparsity_research.metrics import ActivationAccumulator, weight_statistics
from run027_common import write_json


def collect(model,validation,dest,emit):
    accumulator=ActivationAccumulator((0.,.001,.01))
    histograms={}
    handles=[]
    weights=weight_statistics(model)
    started=time.monotonic()
    with ActivationCapture(model,['a','m','h'],torch=torch) as capture, torch.inference_mode():
        def pre_hook(name):
            def hook(module,args): capture.activations[name]=args[0]
            return hook
        for i,layer in enumerate(model.gpt_neox.layers):
            handles.append(layer.attention.dense.register_forward_pre_hook(pre_hook(f'z.layer_{i}')))
        try:
            for i in range(338):
                ids=torch.tensor(validation[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None]
                model(input_ids=ids,use_cache=False)
                accumulator.update(capture.activations,torch=torch)
                for name,value in capture.activations.items():
                    if name.split('.')[0] not in {'h','z'}: continue
                    width=value.shape[-1]
                    hist=torch.bincount(torch.count_nonzero(value.reshape(-1,width),dim=-1),minlength=width+1)
                    histograms.setdefault(name,torch.zeros_like(hist)).add_(hist)
                capture.clear()
                if (i+1)%64==0: emit('diagnostics',blocks=i+1)
        finally:
            for handle in handles: handle.remove()
    per_site=accumulator.rows()
    eligible_zero=sum(r['exact_zero_count']*128 for r in per_site if r['name'].split('.')[0] in {'h','z'})
    eligible_total=sum(r['total']*128 for r in per_site if r['name'].split('.')[0] in {'h','z'})
    write_json(dest,{'coverage':{'documents':500,'blocks':338,'input_tokens':692224,'excluded_tail':1444},
        'scope':'native BF16 SDPA actual a/m/h/z operands; hooks only, no attention wrapper substitution',
        'per_site_layer':per_site,'pooled_by_site':accumulator.pooled_by_site(),'weights':weights,
        'active_features_per_row':{k:v.cpu().tolist() for k,v in histograms.items()},
        'eligible_zero_products':eligible_zero,'eligible_products':eligible_total,
        'eligible_zero_fraction':eligible_zero/eligible_total,'elapsed_seconds':time.monotonic()-started})
