"""One full-length block checks the retained diagnostics and profile serialization."""
import benchmark
bindings = benchmark.bind()
import inspect
import json
import numpy as np
import torch
import transformers
import install
from io_utils import RUN, read, write, verify
from diagnostics import collect
from profiling import collect as profile
from sparsity_research.pythia import load_checkpoint_pythia

assert 'blocks' in inspect.signature(collect).parameters
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
cfg = read(RUN / 'config.json')
manifest = read(RUN / 'provenance/inputs.json')
checkpoint = next(c for c in manifest['checkpoints'] if c['id'] == 'c25')
data = np.memmap(verify(manifest['development']), dtype=np.int32, mode='r')
dest = RUN / 'artifacts/recovery-001/diagnostic-smoke'
dest.mkdir(parents=True, exist_ok=False)

def emit(stage, **fields):
    print(json.dumps({'stage': stage, **fields}), flush=True)

models, runners = {}, {}
with torch.inference_mode():
    ids = torch.tensor(data[:2048].copy(), device='cuda', dtype=torch.long)[None]
    for mode in cfg['implementations']:
        model = load_checkpoint_pythia(transformers.AutoModelForCausalLM, RUN/checkpoint['checkpoint'], torch=torch).to('cuda', dtype=torch.bfloat16).eval()
        model.set_attn_implementation('sdpa'); model.config.use_cache = False
        if mode != 'native': install.install(model, mode)
        models[mode] = model
        runner = benchmark.replay.dense.DenseRunner(lambda x, m=model: benchmark.replay.scaffold.forward(m, x), ids.clone(), 'graph')
        runner.prepare(); runners[mode+'_graph'] = runner
        if mode != 'native':
            path = dest / ('diagnostics-'+mode+'.json')
            collect(model, models['native'], data, path, emit, checkpoint['canonical_logical_products']['architecture_maximum'], blocks=1)
            assert len(read(path)['joint_occupancy_by_layer_limit']) == 24
            emit('diagnostics_passed', implementation=mode)
    profile(models, runners, [ids], dest)
    assert len(read(dest/'profile-summary.json')['profiles']) == 12
write(dest/'result.json', {'status':'passed','blocks':1,'implementations':6,'diagnostics':5,'profiles':12,'bindings':bindings})
print('DIAGNOSTIC_SMOKE_PASSED', flush=True)
