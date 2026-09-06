import sys
from pathlib import Path
import pytest
import torch
import transformers
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
sys.path.insert(0,str(Path(__file__).parent))
from common import HERE,module
from sparsity_research.pythia import apply_activation_topology
from test_joint import reference as joint_reference

candidate=module('test_k019',HERE/'candidates/k019/candidate.py')


def reference(x,c,s,q,k,v,heads,dimension,tq,tk,tv):
    qr,kr,vr=x.view(x.shape[0],x.shape[1],heads,3*dimension).transpose(1,2).chunk(3,dim=-1)
    qr,kr=apply_rotary_pos_emb(qr,kr,c,s)
    for output,value,t in [(q,qr,tq),(k,kr,tk),(v,vr,tv)]:
        output.copy_(value.masked_fill(value.abs()<t,0))


@pytest.mark.parametrize('joint',[False,True])
def test_fused_qkv_ports_and_composition(joint):
    torch.manual_seed(2619)
    cfg=transformers.GPTNeoXConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,vocab_size=97,max_position_embeddings=64,hidden_dropout=0,attention_dropout=0)
    cfg.topology_id='A7-Z-POST'
    cfg.site_gates={s:{'operator':'one_sided_threshold' if s in {'a','m','h','z'} else 'symmetric_threshold',
                       'kappa':.5} for s in ['a','m','h','q_post','k_post','v','z']}
    model=apply_activation_topology(transformers.GPTNeoXForCausalLM(cfg),torch=torch).bfloat16().eval()
    ids=[torch.randint(0,97,(1,16)) for _ in range(3)]
    with torch.inference_mode():
        expected=[model(x,use_cache=False).logits.clone() for x in ids]
        candidate.Adapter(model,joint=joint,backend=reference,joint_backend=joint_reference)
        actual=[model(x,use_cache=False).logits.clone() for x in ids]
    for a,b in zip(actual,expected): torch.testing.assert_close(a,b,rtol=.02,atol=.02)
