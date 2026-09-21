"""Load the real endpoint and verify unchanged h/z gate installation on CPU."""
import torch
from io_utils import RUN, read, write
import replay
from run041_topology import register
from controls import MODES
register()
from transformers import AutoModelForCausalLM
from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata

checkpoint, = read(RUN/'provenance/inputs.json')['checkpoints']
catalog = read(RUN/'provenance/candidates.json')['configurations']
rows = []
with torch.inference_mode():
    for mode in ['frozen', *MODES]:
        model = load_checkpoint_pythia(AutoModelForCausalLM, RUN/checkpoint['checkpoint'], torch=torch).eval()
        assert set(topology_metadata(model)['active_sites']) == {'h', 'z'}
        output = model(input_ids=torch.tensor([[0,1,2,3]]), use_cache=False).logits
        assert output.shape == (1,4,50304) and torch.isfinite(output).all()
        model.to(dtype=torch.bfloat16)
        installation = replay.install(model, mode, catalog)
        rows.append(installation)
write(RUN/'prelaunch/cpu-smoke.json', {'status':'passed','installations':rows,
    'scope':'Real CPU checkpoint native forward and all five installations; no CUDA or performance claim'})
print('Passed real-checkpoint CPU forward and all five gate-preserving installations')
