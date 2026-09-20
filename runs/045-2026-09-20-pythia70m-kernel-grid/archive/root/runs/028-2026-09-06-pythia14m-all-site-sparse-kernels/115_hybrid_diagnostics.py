"""Untimed actual-operand diagnostics for the K049/K050 hybrid/no-prefix study.

Input projection MMA counts follow the executed16x8x16 A-fragment test.
Output projection counts come from the instrumented hybrid M8/padded-M16 path.
Bypassed MMA includes SIMT substitution; SIMT product counts are separate.
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


def hybrid_counts(value,fast_weights=True,skip=True):
    x=value.reshape(-1,value.shape[-1]);m,k=x.shape
    potential=(m//8)*(k//16)*16
    if not skip:return [potential,0,0]
    live=x!=0;nnz=live.sum(-1)
    safe=((x==0)|((x.abs()>=2.**-50)&(x.abs()<=2.**50))).all(-1)
    simple=(nnz<=2)&safe if fast_weights else torch.zeros_like(safe)
    remaining=live & (~simple[:,None])
    active=remaining.reshape(m//8,8,k//16,16).any(1).any(-1)
    issued=int(active.sum())*16
    return [issued,potential-issued,int(nnz[simple].sum())*128]


def collect(model,native,validation,dest,emit,architecture,*,blocks=338):
    started=time.monotonic();accumulator=ActivationAccumulator((0.,.001,.01))
    weights=weight_statistics(native);histograms={};counts={};attention_counts={};joint_counts={}
    originals=[]
    def add(operation,row):
        target=counts.setdefault(operation,{})
        for key,value in row.items():target[key]=target.get(key,0)+int(value)
    with ActivationCapture(model,['a','m'],torch=torch) as capture,torch.inference_mode():
        class Joint:
            def __init__(self,original,index):self.original,self.index=original,index
            def __call__(self,h,z,residual):
                original=self.original
                for site,value,gate,threshold in [('h',h,original.gh,original.th),('z',z,original.gz,original.tz)]:
                    capture.activations[f'{site}.layer_{self.index}']=value.masked_fill(value<threshold,0) if gate else value
                previous_count=original.count;original.count=True
                try:output=original(h,z,residual)
                finally:original.count=previous_count
                stat=original.stats.sum(tuple(range(original.stats.ndim-1))).cpu().tolist()
                if len(stat)!=6 or any(type(n)!=int or n<0 for n in stat):
                    raise ValueError('Six nonnegative integer joint counters required')
                if stat[0]+stat[1]!=131072 or stat[2]+stat[3]!=32768:
                    raise ValueError('M8 padded-M16 joint counter conservation failed')
                target=joint_counts.setdefault(str(self.index),[0]*6)
                for j,value in enumerate(stat):target[j]+=value
                for site,op,offset,simt in [('h','mlp_w2',0,4),('z','attention_output_projection',2,5)]:
                    value=capture.activations[f'{site}.layer_{self.index}']
                    expected=hybrid_counts(value,original.fast_weights,original.skip)
                    if [stat[offset],stat[offset+1],stat[simt]]!=expected:
                        raise ValueError('Independent hybrid work-count mismatch')
                    add(op,{'issued_mmas':stat[offset],'skipped_mmas':stat[offset+1],'simt_products':stat[simt]})
                return output
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
                        row=projection_counts(value,n)
                        if site in {'h','z'}:
                            row={k:row[k] for k in ['product_count','zero_product_count']}
                        add(operation,row)
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
        'projection_counter_method':'a/m: operand-derived unpadded A-fragment tests; h/z: instrumented M8 padded-M16 hybrid with independent operand verification',
        'hybrid_counts_by_layer':joint_counts,
        'hybrid_counter_fields':['h_mma_issued','h_mma_bypassed','z_mma_issued','z_mma_bypassed','h_simt_products','z_simt_products'],
        'hybrid_limits':'MMA bypass includes SIMT substitution; h/z dense potential has twofold row padding. Enabling/disabling paths is not a pure zero-multiply ablation.',
        'attention_counter_method':'CUDA instrumented issued/skipped16x8x16 MMA atoms, including causal padding',
        'attention_counts_by_layer':attention_counts,
        'attention_counter_fields':['qk_issued','qk_skipped','pv_issued','pv_skipped'],
        'mma_fma_equivalent_terms_per_atom':2048,
        'per_site_layer':accumulator.rows(),'pooled_by_site':accumulator.pooled_by_site(),
        'active_features_per_row':{name:value.cpu().tolist() for name,value in histograms.items()},
        'native_weight_statistics':weights,'elapsed_seconds':time.monotonic()-started})
