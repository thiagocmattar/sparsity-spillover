import sys
from pathlib import Path
import torch
import transformers
sys.path.insert(0,str(Path(__file__).parent))
from common import HERE
from sparsity_research.pythia import apply_activation_topology
import dense_compatible


def test_import_hoist_is_exact_and_fullgraph_traceable():
    cfg=transformers.GPTNeoXConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=1,
        num_attention_heads=4,vocab_size=97,max_position_embeddings=64,hidden_dropout=0,attention_dropout=0)
    cfg.topology_id='A7-Z-POST'
    cfg.site_gates={s:{'operator':'one_sided_threshold' if s in {'a','m','h','z'} else 'symmetric_threshold',
                       'kappa':.5} for s in ['a','m','h','q_post','k_post','v','z']}
    model=apply_activation_topology(transformers.GPTNeoXForCausalLM(cfg),torch=torch).bfloat16().eval()
    ids=torch.randint(0,97,(1,16))
    with torch.inference_mode():
        expected=model(ids,use_cache=False).logits
        dense_compatible.install(model)
        torch.testing.assert_close(model(ids,use_cache=False).logits,expected,rtol=0,atol=0)
        compiled=torch.compile(lambda x:model(x,use_cache=False).logits,backend='eager',fullgraph=True)
        torch.testing.assert_close(compiled(ids),expected,rtol=0,atol=0)
