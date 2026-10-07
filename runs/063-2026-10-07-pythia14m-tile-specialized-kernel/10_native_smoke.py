"""Real-checkpoint development qualification; exact cloud path remains mandatory."""
import argparse
import time
import numpy as np
import torch
import transformers
import replay
import topology
from io_utils import RUN,read,write,verify
from sparsity_research.pythia import load_checkpoint_pythia

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',required=True)
    parser.add_argument('--blocks',type=int,choices=[2,338],default=2)
    parser.add_argument('--attention',choices=['frozen','native-sdpa'],default='frozen')
    args=parser.parse_args()
    if not args.tag.isalnum() or (RUN/f'prelaunch/native-cuda-{args.tag}.json').exists():
        parser.error('Fresh simple tag required')
    started=time.monotonic();topology.register();checks=[]
    cfg=read(RUN/'config.json');inputs=read(RUN/'provenance/inputs.json')
    torch.manual_seed(cfg['runtime_seed']);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    validation=np.memmap(verify(inputs['validation']),dtype=np.int32,mode='r')
    with torch.inference_mode():
        for source in inputs['checkpoints']:
            model=load_checkpoint_pythia(transformers.AutoModelForCausalLM,RUN/source['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
            model.set_attn_implementation('sdpa');model.config.use_cache=False
            ids=torch.tensor(validation[:2048].copy(),device='cuda',dtype=torch.long)[None]
            anchor=replay.dense.DenseRunner(lambda x:model(input_ids=x,use_cache=False).logits,ids.clone(),'native');anchor.prepare()
            groups={'native':anchor}
            for mode in cfg['modes']:
                if mode!='native':
                    m=load_checkpoint_pythia(transformers.AutoModelForCausalLM,RUN/source['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
                    m.set_attn_implementation('sdpa');m.config.use_cache=False
                    replay.install(m,mode)
                    if args.attention=='native-sdpa':
                        # K035 freezes the RTX5090 two-split KV schedule. Local
                        # SDPA uses a different schedule; qualify the consumer
                        # with the same native attention on both sides locally.
                        for layer in m.gpt_neox.layers:
                            layer.attention._run028_attention=NativeAttention()
                else:m=model
                runner=replay.dense.DenseRunner(lambda x,m=m:replay.scaffold.forward(m,x),ids.clone(),'graph');runner.prepare();groups[mode+'_graph']=runner
            stream=(torch.tensor(validation[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in range(args.blocks))
            quality=replay.dense.compare_inputs(groups,stream,cfg['numerical_bounds'])
            quality.update(blocks=args.blocks,documents=500 if args.blocks==338 else None,
                           input_tokens=args.blocks*2048,excluded_tail_tokens=1444 if args.blocks==338 else None)
            write(RUN/f'prelaunch/smoke-quality-{source["id"]}-{args.tag}.json',quality)
            print(quality['pass'],flush=True)
            if not all(quality['pass'].values()):raise ValueError('Real checkpoint smoke failed')
            checks.append({'checkpoint':source['id'],'quality':quality});print(source['id'],'passed',flush=True)
            del groups,anchor,runner,m,model;torch.cuda.empty_cache()
    write(RUN/f'prelaunch/native-cuda-{args.tag}.json',{'status':'passed','checks':checks,'gpu':torch.cuda.get_device_name(),
          'seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
          'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'attention':args.attention,
          'scope':f'{args.blocks} real B1/T2048 blocks per checkpoint, five graphs vs own eager; development qualification, not RTX5090 path or accepted latency'})


class NativeAttention:
    def __call__(self,q,k,v,scale,*,count=False):
        if count:raise ValueError('Development native attention has no K035 counters')
        return torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,scale=scale)

if __name__=='__main__':main()
