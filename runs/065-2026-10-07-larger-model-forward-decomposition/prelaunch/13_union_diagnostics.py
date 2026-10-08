"""Post-cohort actual70M union counts; original frozen calls remain unchanged."""
import sys
from pathlib import Path
import math
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runio import read,write,verify,record

def collect_short(model,replay,validation):
    import torch
    stats={};originals=[]
    for i,layer in enumerate(model.gpt_neox.layers):
        joint=layer._run026_joint;originals.append(joint)
        if not joint.skip or not joint.fast_weights:raise ValueError('Expected active short-row prepass')
        key=f'joint.{i}';stats[key]={'rows':0,'blocks8':0,'both_short_rows':0,'all_short_blocks8':0,'capped_hz_state_histogram':[0]*100}
        def wrapped(h,z,r,j=joint,k=key):
            out=j(h,z,r);counts=j.row_counts
            histogram=torch.bincount((counts[:,0]*10+counts[:,1]).long(),minlength=100).cpu().tolist()
            if len(histogram)!=100:raise ValueError('Unexpected short-row state')
            both=(counts<=8).all(1);s=stats[k]
            s['rows']+=2048;s['blocks8']+=256;s['both_short_rows']+=int(both.sum());s['all_short_blocks8']+=int(both.reshape(256,8).all(1).sum())
            s['capped_hz_state_histogram']=[a+b for a,b in zip(s['capped_hz_state_histogram'],histogram)]
            return out
        layer._run026_joint=wrapped
    for block in range(338):
        ids=torch.tensor(validation[block*2048:(block+1)*2048].copy(),device='cuda',dtype=torch.long)[None];replay.scaffold.forward(model,ids)
    for layer,joint in zip(model.gpt_neox.layers,originals):layer._run026_joint=joint
    for s in stats.values():
        if sum(s['capped_hz_state_histogram'])!=338*2048 or s['rows']!=338*2048:raise ValueError('Row coverage')
        s.update(both_short_row_fraction=s['both_short_rows']/s['rows'],early_return_block_fraction=s['all_short_blocks8']/s['blocks8'])
    return stats

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--size',choices=['31m','70m'],default='70m');args=parser.parse_args()
    import torch,numpy as np,transformers
    import bindings
    replay,install=bindings.setup(args.size)
    from sparsity_research.pythia import load_checkpoint_pythia
    if torch.cuda.get_device_name()!='NVIDIA GeForce RTX 5090':raise ValueError('GPU identity')
    torch.manual_seed(2801);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    inputs=read(ROOT/'provenance/inputs.json');validation=np.memmap(verify(inputs['validation']),dtype=np.int32,mode='r')
    for pair in ['0','0p1']:
        start=time.monotonic();cp=next(r for r in inputs['checkpoints'] if r['size']==args.size and r['pair']==pair)
        for row in cp['files']:verify(row)
        with torch.inference_mode():
            model=load_checkpoint_pythia(transformers.AutoModelForCausalLM,ROOT/cp['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
            model.set_attn_implementation('sdpa');model.config.use_cache=False;meta=install(model,'sparse');stats={};originals=[]
            if args.size=='31m':
                stats=collect_short(model,replay,validation)
                write(ROOT/f'artifacts/short-rows-31m-k{pair}.json',{'status':'complete','coverage':{'blocks':338,'documents':500,'excluded_tail_tokens':1444},'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),'checkpoint':cp,'implementation':meta,'source':record(Path(__file__)),'sites':stats,'elapsed_seconds':time.monotonic()-start,'unit':'Actual prepass state pair per token:0..8 safe nonzeros;9 means longer/unsafe. Eight-token groups with both branches short for every row allow the fallback CTA to return early.'})
                del model
                continue
            for layer_index,layer in enumerate(model.gpt_neox.layers):
                joint=layer._run026_joint;originals.append(joint)
                if joint.h.spec['family']!='c':continue
                linear=joint.h;gm=linear.spec['gm'];bk=linear.spec['bk'];key=f'h.{layer_index}'
                stats[key]={'groups':0,'gm':gm,'bk':bk,'k':linear.k,'n':linear.n,'union_histogram':[0]*(linear.k+1),'union_features':0,'empty_groups':0,'rounded_k_sum':0}
                def wrapped(h,z,r,j=joint,k=key):
                    out=j(h,z,r);counts=j.h.count
                    histogram=torch.bincount(counts.long(),minlength=j.h.k+1).cpu().tolist();s=stats[k]
                    s['groups']+=counts.numel();s['union_features']+=int(counts.sum());s['empty_groups']+=int((counts==0).sum());s['rounded_k_sum']+=int(((counts+s['bk']-1)//s['bk']*s['bk']).sum())
                    s['union_histogram']=[a+b for a,b in zip(s['union_histogram'],histogram)]
                    return out
                layer._run026_joint=wrapped
            for block in range(338):
                ids=torch.tensor(validation[block*2048:(block+1)*2048].copy(),device='cuda',dtype=torch.long)[None];replay.scaffold.forward(model,ids)
                if (block+1)%64==0 or block==337:
                    write(ROOT/'artifacts/union-status.json',{'stage':'union_counts','pair':pair,'blocks':block+1,'target_blocks':338,'elapsed_seconds':time.monotonic()-start})
            for layer,joint in zip(model.gpt_neox.layers,originals):layer._run026_joint=joint
            for s in stats.values():
                if sum(s['union_histogram'])!=s['groups'] or s['groups']!=338*2048//s['gm']:raise ValueError('Group coverage')
                s.update(mean_union_features=s['union_features']/s['groups'],mean_rounded_k=s['rounded_k_sum']/s['groups'],retained_feature_fraction=s['union_features']/(s['groups']*s['k']),tensorcore_k_fraction=s['rounded_k_sum']/(s['groups']*s['k']),padded_tensorcore_products=16*s['rounded_k_sum']*s['n'],no_skip_padded_tensorcore_products=16*s['groups']*s['k']*s['n'])
            write(ROOT/f'artifacts/feature-unions-70m-k{pair}.json',{'status':'complete','coverage':{'blocks':338,'documents':500,'excluded_tail_tokens':1444},'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),'checkpoint':cp,'implementation':meta,'source':record(Path(__file__)),'sites':stats,'elapsed_seconds':time.monotonic()-start,'unit':'feature union/count per actual GM-token consumer group; rounded K counts32-wide MMA operands including padding. Product counters are logical executed tensorcore operands, not measured speedups.'})
            del model
    write(ROOT/'artifacts/union-status.json',{'stage':'complete','pairs':['0','0p1']})

if __name__=='__main__':main()
