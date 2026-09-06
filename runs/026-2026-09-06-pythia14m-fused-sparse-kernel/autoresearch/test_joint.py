import sys
from pathlib import Path
import torch
import transformers

sys.path.insert(0,str(Path(__file__).parent))
from common import HERE,module
from sparsity_research.pythia import apply_activation_topology

candidate=module('test_k018',HERE/'candidates/k018/candidate.py')


def reference(h,z,wh,wz,bh,bz,r,out,th,tz):
    a=(h.masked_fill(h<th,0).float()@wh.float()+bh.float()).bfloat16()
    b=(z.masked_fill(z<tz,0).float()@wz.float()+bz.float()).bfloat16()
    out.copy_((a+b)+r)


def test_joint_preserves_parallel_bf16_rounding_and_new_inputs():
    torch.manual_seed(2618)
    cfg=transformers.GPTNeoXConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,vocab_size=97,max_position_embeddings=64,hidden_dropout=0,attention_dropout=0)
    cfg.topology_id='A7-Z-POST'
    cfg.site_gates={s:{'operator':'one_sided_threshold' if s in {'a','m','h','z'} else 'symmetric_threshold',
                       'kappa':.5} for s in ['a','m','h','q_post','k_post','v','z']}
    model=apply_activation_topology(transformers.GPTNeoXForCausalLM(cfg),torch=torch).bfloat16().eval()
    ids=[torch.randint(0,97,(1,16)) for _ in range(3)]
    with torch.inference_mode():
        expected=[model(x,use_cache=False).logits.clone() for x in ids]
        candidate.Adapter(model,backend=reference).install()
        actual=[model(x,use_cache=False).logits.clone() for x in ids]
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=.02,atol=.02)
