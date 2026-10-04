"""Exercise the existing HZ adapter on all four actual retained checkpoints."""
import torch
import transformers
from io_utils import RUN,read,write
import replay_hz as replay
from sparsity_research.sites import TOPOLOGIES,Topology
from sparsity_research.pythia import load_checkpoint_pythia,topology_metadata

TOPOLOGIES['HZ']=Topology('HZ',('h','z'))
catalog=read(RUN/'provenance/candidates.json')['configurations']
row=replay.final_row('k050',catalog);output=[]
with torch.inference_mode():
    for checkpoint in read(RUN/'provenance/inputs.json')['checkpoints']:
        model=load_checkpoint_pythia(transformers.AutoModelForCausalLM,RUN/checkpoint['checkpoint'],torch=torch).eval()
        assert set(topology_metadata(model)['active_sites'])=={'h','z'}
        assert all(not hasattr(layer,'a_gate') and not hasattr(layer,'m_gate') for layer in model.gpt_neox.layers)
        logits=model(input_ids=torch.arange(16)[None],use_cache=False).logits
        assert logits.shape==(1,16,50304) and torch.isfinite(logits).all()
        model.to(dtype=torch.bfloat16)
        metadata=replay.install(model,row)
        assert all(layer._run026_joint.gh and layer._run026_joint.gz for layer in model.gpt_neox.layers)
        output.append({'checkpoint':checkpoint['id'],'installation':metadata,'native_forward_finite':True})
        print(checkpoint['id']+' CPU native and HZ K050 installation passed',flush=True)
write(RUN/'prelaunch/cpu-hz-install-002.json',{'status':'passed','conditions':output,
    'scope':'All four actual checkpoint native FP32 CPU forwards and BF16 K050 installations; no CUDA or speed claim'})
