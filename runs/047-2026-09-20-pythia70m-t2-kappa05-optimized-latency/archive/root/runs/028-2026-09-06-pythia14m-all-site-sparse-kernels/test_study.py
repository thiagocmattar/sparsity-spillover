"""Mathematical unit tests; full-model CUDA validation remains mandatory."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import torch

RUN=Path(__file__).parent
sys.path.insert(0,str(RUN))


def load(name):
    spec=importlib.util.spec_from_file_location('test_'+name,RUN/name)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def test_projection_mma_and_scalar_denominators():
    diagnostic=load('58_study_diagnostics.py')
    x=torch.ones(32,32);x[:16,:16]=0
    counts=diagnostic.projection_counts(x,16)
    assert counts=={'product_count':16384,'zero_product_count':4096,'issued_mmas':6,'skipped_mmas':2}
    assert (counts['issued_mmas']+counts['skipped_mmas'])*2048==counts['product_count']


def test_causal_zero_union_and_v_only_lower_bound():
    diagnostic=load('58_study_diagnostics.py')
    q=torch.tensor([[[[1.,0.],[0.,1.],[1.,1.]]]])
    k=torch.tensor([[[[0.,1.],[1.,1.],[0.,0.]]]])
    v=torch.tensor([[[[1.,0.],[0.,1.],[1.,1.]]]])
    counts=diagnostic.attention_opportunities(q,k,v)
    qk=sum(q[0,0,i,d]==0 or k[0,0,j,d]==0 for i in range(3) for j in range(i+1) for d in range(2))
    pv=sum(v[0,0,j,d]==0 for i in range(3) for j in range(i+1) for d in range(2))
    assert counts['qk_scores']=={'product_count':12,'zero_product_count':int(qk)}
    assert counts['probability_value']=={'product_count':12,'zero_product_count':int(pv)}


def test_graph_scaffold_preserves_dynamic_input_and_fixed_causality():
    scaffold=load('45_graph_forward.py')
    class Layer(torch.nn.Module):
        def forward(self,x,**kwargs):
            assert kwargs['attention_mask'] is None and kwargs['layer_past'] is None
            assert kwargs['use_cache'] is False and kwargs['position_ids'].shape==(1,2048)
            return x+1
    body=SimpleNamespace(embed_in=torch.nn.Embedding(2,4),emb_dropout=torch.nn.Identity(),
                         rotary_emb=lambda x,position_ids:(position_ids,position_ids),
                         layers=[Layer(),Layer()],final_layer_norm=torch.nn.Identity())
    model=SimpleNamespace(gpt_neox=body,embed_out=torch.nn.Linear(4,8,bias=False),training=False)
    with torch.inference_mode():
        ids=torch.zeros((1,2048),dtype=torch.long)
        first=scaffold.forward(model,ids)
        assert torch.equal(first,model.embed_out(body.embed_in(ids)+2))
        second=scaffold.forward(model,torch.ones_like(ids))
        assert not torch.equal(first,second)
