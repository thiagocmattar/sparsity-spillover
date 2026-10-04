"""Non-evidence full-model GPU integration: retained scalars and untrained tiles."""
import gc
import sys
import time
from io_utils import RUN, ARCHIVE, read, write, record, verify
sys.path.append(str(ARCHIVE / 'src'))
import kernel
import diagnostics
import replay
from topology import register
register()
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from sparsity_research.pythia import load_checkpoint_pythia


def main():
    started = time.monotonic()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    cfg = read(RUN / 'config.json')
    inputs = read(RUN / 'provenance/development-inputs.json')
    validation = np.memmap(verify(inputs['validation']), dtype=np.int32, mode='r')
    rows = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for checkpoint in inputs['checkpoints']:
            for entry in checkpoint['files']:
                verify(entry)
            group = {}; models = []
            ids = torch.tensor(validation[:2048].copy(), device='cuda', dtype=torch.long)[None]
            for mode in cfg['modes']:
                model = load_checkpoint_pythia(AutoModelForCausalLM, RUN / checkpoint['checkpoint'], torch=torch)
                model = model.to('cuda', dtype=torch.bfloat16).eval()
                model.set_attn_implementation('sdpa'); model.config.use_cache = False
                models.append(model)
                if mode == 'native':
                    anchor = replay.dense.DenseRunner(lambda value,m=model:m(input_ids=value,use_cache=False).logits, ids.clone(), 'native')
                    anchor.prepare(); group['native'] = anchor
                else:
                    replay.install(model, mode)
                graph = replay.dense.DenseRunner(lambda value,m=model:replay.scaffold.forward(m,value), ids.clone(), 'graph')
                graph.prepare(); group[mode+'_graph'] = graph
            stream = (torch.tensor(validation[i*2048:(i+1)*2048].copy(), device='cuda', dtype=torch.long)[None] for i in range(2))
            result = replay.dense.compare_inputs(group, stream, cfg['numerical_bounds'])
            rows.append({'checkpoint': checkpoint['id'], 'family': checkpoint['family'], 'blocks': 2, 'qualification': result})
            print('Qualification', checkpoint['id'], result['pass'], flush=True)
            del group, models, model, graph, ids, anchor, stream
            gc.collect(); torch.cuda.empty_cache()
    qualified=all(all(row['qualification']['pass'].values()) for row in rows)
    write(RUN / 'prelaunch/native-cuda-smoke.json', {'status':'passed' if qualified else 'completed_with_qualification_failure',
        'qualified':qualified,'kind':'non_evidence_gpu_integration',
        'gpu':torch.cuda.get_device_name(),'rows':rows,'seconds':time.monotonic()-started,
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'limitation':'Two blocks per checkpoint; tile weights are untrained. Mandatory final RTX5090 full validation follows training.'})


if __name__ == '__main__':
    main()
