"""Four real-checkpoint B1/T2048 native CUDA smoke and memory calibration."""
import time
import sys
from io_utils import RUN, ARCHIVE, read, verify, write
sys.path.insert(0,str(ARCHIVE/'src'))


def main():
    started=time.monotonic()
    import replay
    import numpy as np
    import torch
    import transformers
    import topology
    topology.register()
    from sparsity_research.pythia import load_checkpoint_pythia
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    inputs=read(RUN/'provenance/inputs.json')
    cache=np.memmap(verify(inputs['validation']),dtype=np.int32,mode='r')
    ids=[torch.tensor(cache[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in range(2)]
    models=[];runners=[];rows=[]
    with torch.inference_mode():
        for source in inputs['checkpoints']:
            for row in source['files']:
                verify(row)
            model=load_checkpoint_pythia(transformers.AutoModelForCausalLM,RUN/source['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
            model.set_attn_implementation('sdpa');model.config.use_cache=False
            runner=replay.dense.DenseRunner(lambda x,m=model:replay.scaffold.forward(m,x),ids[0].clone(),'graph')
            runner.prepare()
            group={'native':replay.dense.DenseRunner(lambda x,m=model:m(input_ids=x,use_cache=False).logits,ids[0].clone(),'native'),
                   'native_graph':runner}
            quality=replay.dense.compare_inputs(group,ids,read(RUN/'config.json')['numerical_bounds'])
            if not all(quality['pass'].values()):
                raise ValueError('Native graph failed its own eager anchor')
            before=time.monotonic()
            for _ in range(32):
                runner()
            torch.cuda.synchronize()
            rows.append({'id':source['id'],'blocks':2,'quality':quality,
                         'calibration_graph_ms':(time.monotonic()-before)*1000/32,'setup_seconds':runner.setup_seconds})
            models.append(model);runners.append(runner)
    write(RUN/'prelaunch/native-cuda-smoke.json',{'status':'passed','gpu':torch.cuda.get_device_name(),
          'rows':rows,'torch':torch.__version__,'cuda':torch.version.cuda,'seconds':time.monotonic()-started,
          'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
          'scope':'four real native checkpoints, two B1/T2048 blocks each; local memory/ETC calibration only, no K050/full-validation qualification'})
    print(f'Passed four native checkpoints in {time.monotonic()-started:.2f}s; peak {torch.cuda.max_memory_reserved()/2**30:.3f}GiB')


if __name__=='__main__':
    main()
