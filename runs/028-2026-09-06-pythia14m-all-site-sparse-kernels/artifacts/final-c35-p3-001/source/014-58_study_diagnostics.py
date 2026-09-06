"""Untimed actual-operand diagnostics for the fixed K036/no-prefix study.

Projection MMA counts follow the executed16x8x16 operator's A-fragment test.
Attention MMA counts come from its instrumented CUDA path. Neither count is
the canonical FP16 R_model. The BF16 scalar opportunity below is a lower
bound: it counts activation zeros and Q/K unions, but not weight zeros or
softmax-probability underflow. V-only PV counts must not be called full R_model.
"""
import time
import torch
from sparsity_research.capture import ActivationCapture
from sparsity_research.metrics import ActivationAccumulator,weight_statistics
from common import write_json


def projection_counts(x,n):
    k=x.shape[-1];x=x.reshape(-1,k);m=x.shape[0]
    if m%16 or k%16 or n%8:raise ValueError('Unpadded MMA shape required')
    zeros=x==0
    tiles=int(zeros.reshape(m//16,16,k//16,16).all(1).all(-1).sum())
    total_mmas=(m//16)*(k//16)*(n//8);skipped=tiles*(n//8)
    return {'product_count':m*k*n,'zero_product_count':int(zeros.sum())*n,
            'issued_mmas':total_mmas-skipped,'skipped_mmas':skipped}


def attention_opportunities(q,k,v):
    t=q.shape[-2]
    if q.shape!=k.shape or q.shape!=v.shape:raise ValueError('Fixed equal QKV shapes')
    total=q.numel()*(t+1)//2
    live_qk=int(((q!=0)*(k!=0).to(torch.int64).cumsum(-2)).sum())
    multiplicity=torch.arange(t,0,-1,device=v.device,dtype=torch.int64)[None,None,:,None]
    live_pv=int(((v!=0)*multiplicity).sum())
    return {'qk_scores':{'product_count':total,'zero_product_count':total-live_qk},
            'probability_value':{'product_count':total,'zero_product_count':total-live_pv}}


def collect(model,native,validation,dest,emit,architecture,*,blocks=338):
    started=time.monotonic();accumulator=ActivationAccumulator((0.,.001,.01))
    weights=weight_statistics(native);histograms={};counts={};attention_counts={}
    originals=[]
    def add(operation,row):
        target=counts.setdefault(operation,{k:0 for k in row})
        for key,value in row.items():target[key]+=int(value)
    with ActivationCapture(model,['a','m'],torch=torch) as capture,torch.inference_mode():
        class Joint:
            def __init__(self,original,index):self.original,self.index=original,index
            def __call__(self,h,z,residual):
                original=self.original
                for site,value,gate,threshold in [('h',h,original.gh,original.th),('z',z,original.gz,original.tz)]:
                    capture.activations[f'{site}.layer_{self.index}']=value.masked_fill(value<threshold,0) if gate else value
                return original(h,z,residual)
        class Attention:
            def __init__(self,original,index):self.original,self.index=original,index
            def __call__(self,q,k,v,scale):
                output=self.original(q,k,v,scale,count=True)
                for site,value in [('q_post',q),('k_post',k),('v',v)]:capture.activations[f'{site}.layer_{self.index}']=value
                stat=self.original.stats.sum((0,1,2,3)).cpu().tolist()
                prefix=self.original.prefix_stats.sum().item()
                if prefix!=0 or stat[0]+stat[1]!=147456 or stat[2]+stat[3]!=147456:
                    raise ValueError('No-prefix attention counter conservation failed')
                target=attention_counts.setdefault(str(self.index),[0,0,0,0])
                for j,value in enumerate(stat):target[j]+=int(value)
                for op,row in attention_opportunities(q,k,v).items():add(op,row)
                return output
        for index,layer in enumerate(model.gpt_neox.layers):
            originals.append((layer,layer._run026_joint,layer.attention._run028_attention))
            layer._run026_joint=Joint(layer._run026_joint,index)
            layer.attention._run028_attention=Attention(layer.attention._run028_attention,index)
        try:
            for index in range(blocks):
                ids=torch.tensor(validation[index*2048:(index+1)*2048].copy(),device='cuda',dtype=torch.long)[None]
                model(input_ids=ids,use_cache=False)
                accumulator.update(capture.activations,torch=torch)
                for name,value in capture.activations.items():
                    site=name.split('.')[0];width=value.shape[-1]
                    hist=torch.bincount(torch.count_nonzero(value.reshape(-1,width),dim=-1),minlength=width+1)
                    histograms.setdefault(name,torch.zeros_like(hist)).add_(hist)
                    if site in {'a','m','h','z'}:
                        operation,n={'a':('qkv_projection',384),'m':('mlp_w1',512),
                                     'h':('mlp_w2',128),'z':('attention_output_projection',128)}[site]
                        add(operation,projection_counts(value,n))
                capture.clear()
                if (index+1)%32==0:
                    elapsed=time.monotonic()-started
                    emit('diagnostics',blocks=index+1,input_tokens_per_second=(index+1)*2048/elapsed,
                         remaining_seconds=(blocks-index-1)*elapsed/(index+1))
        finally:
            for layer,joint,attention in originals:
                layer._run026_joint=joint;layer.attention._run028_attention=attention
    for operation,expected in architecture['per_block_operation_products'].items():
        if counts[operation]['product_count']!=expected*architecture['layers']*blocks:
            raise ValueError(f'Logical denominator changed: {operation}')
    numerator=sum(row['zero_product_count'] for row in counts.values())
    denominator=architecture['model_product_count']*blocks
    write_json(dest,{'status':'complete','coverage':{'blocks':blocks,'documents':500 if blocks==338 else None,
        'input_tokens':blocks*2048,'excluded_tail_tokens':1444 if blocks==338 else None},
        'precision':'actual BF16 candidate operands; untimed counting pass',
        'bf16_scalar_opportunity_lower_bound':{'zero_products':numerator,'model_products':denominator,
            'fraction':numerator/denominator,'per_operation':counts,
            'exclusions':'weight zeros and probability underflow; not canonical FP16 R_model'},
        'projection_counter_method':'operand-derived executed16x8x16 A-fragment tests; weight loads remain',
        'attention_counter_method':'CUDA instrumented issued/skipped16x8x16 MMA atoms, including causal padding',
        'attention_counts_by_layer':attention_counts,
        'attention_counter_fields':['qk_issued','qk_skipped','pv_issued','pv_skipped'],
        'mma_fma_equivalent_terms_per_atom':2048,
        'per_site_layer':accumulator.rows(),'pooled_by_site':accumulator.pooled_by_site(),
        'active_features_per_row':{name:value.cpu().tolist() for name,value in histograms.items()},
        'native_weight_statistics':weights,'elapsed_seconds':time.monotonic()-started})
