"""Pool actual optimized joint operands over the complete validation workload."""
import math
import torch
from runio import write

def collect(model,forward,validation,path,emit):
    rows={};saved=[]
    for index,layer in enumerate(model.gpt_neox.layers):
        joint=layer._run026_joint;saved.append(joint)
        def wrapped(h,z,residual,j=joint,i=index):
            for site,x,threshold in [('h',h,j.th),('z',z,j.tz)]:
                t=float(torch.tensor(threshold,dtype=torch.bfloat16));x=torch.where(x>=t,x,0).float().reshape(2048,-1)
                key=f'{site}.{i}';r=rows.setdefault(key,{'count':0,'zeros':0,'near_0p001':0,'near_0p01':0,'sum_squares':0.,'row_support_histogram':[0]*(x.shape[1]+1),'tiles8x16':0,'empty_tiles8x16':0})
                r['count']+=x.numel();r['zeros']+=int((x==0).sum());r['near_0p001']+=int((x.abs()<.001).sum());r['near_0p01']+=int((x.abs()<.01).sum());r['sum_squares']+=float(x.double().square().sum())
                histogram=torch.bincount((x!=0).sum(1),minlength=x.shape[1]+1).cpu().tolist()
                r['row_support_histogram']=[a+b for a,b in zip(r['row_support_histogram'],histogram)]
                tiles=x.reshape(256,8,x.shape[1]//16,16);empty=(tiles==0).all(3).all(1)
                r['tiles8x16']+=empty.numel();r['empty_tiles8x16']+=int(empty.sum())
            return j(h,z,residual)
        layer._run026_joint=wrapped
    try:
        for block in range(338):
            ids=torch.tensor(validation[block*2048:(block+1)*2048].copy(),device='cuda',dtype=torch.long)[None]
            forward(model,ids)
            if (block+1)%32==0 or block==337:emit('diagnostics',blocks=block+1,target_blocks=338)
    finally:
        for layer,joint in zip(model.gpt_neox.layers,saved):layer._run026_joint=joint
    for r in rows.values():r.update(rms=math.sqrt(r['sum_squares']/r['count']),l2=math.sqrt(r['sum_squares']),zero_fraction=r['zeros']/r['count'])
    weights={name:{'elements':p.numel(),'l2':float(p.double().norm()),'rms':float(p.double().square().mean().sqrt())} for name,p in model.named_parameters()}
    write(path,{'status':'complete','coverage':{'blocks':338,'documents':500,'input_tokens':338*2048,'excluded_tail_tokens':1444},'operands':'Actual sparse implementation joint inputs, scalar BF16 gate applied before pooled counts','activations':rows,'weights':weights})
