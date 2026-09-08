import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
from sparsity_research.pythia import apply_activation_topology

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('run030_clipping', HERE/'clipping.py')
clipping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clipping)


def test_empirical_quantiles_keep_zero_ties_and_clip_threshold_equality():
    result = clipping.empirical_thresholds(np.array([0,0,0,1,2,3,4,5,6,7]),(0,.1,.3,.4,.9),np=np)
    assert [r['threshold'] for r in result['targets']]==[0,0,0,1,6]
    assert [r['calibration_zero_count'] for r in result['targets']]==[3,3,3,4,9]
    with pytest.raises(ValueError):
        clipping.empirical_thresholds(np.array([np.nan]),(.1,),np=np)


@pytest.mark.parametrize('layers',[1,2])
def test_clipping_is_after_trained_gates_and_zero_target_preserves_logits(layers):
    config = GPTNeoXConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=layers,
                          num_attention_heads=4,vocab_size=32,max_position_embeddings=32,
                          hidden_dropout=0,attention_dropout=0)
    config.topology_id='A7-Z-POST'
    config.site_gates={site:{'operator':'one_sided_threshold' if site in ('a','m','h','z')
                            else 'symmetric_threshold','kappa':.2}
                       for site in ('a','m','h','q_post','k_post','v','z')}
    config._attn_implementation='eager'
    model=apply_activation_topology(GPTNeoXForCausalLM(config),torch=torch).eval()
    tokens=torch.tensor([[1,2,3,4]])
    original=model(tokens).logits.detach()
    thresholds={name:0. for name in clipping._module_map(model)}
    with clipping.threshold_hooks(model,thresholds) as captured:
        assert torch.equal(model(tokens).logits,original)
        assert len(captured)==4*layers
    # Directly test the registered trained LayerNorm gate and subsequent clipping.
    layer=model.gpt_neox.layers[0]
    x=torch.linspace(-1,1,16).reshape(1,1,16)
    trained=layer.input_layernorm(x).detach()
    cutoff=float(trained.max()/2)
    thresholds['a.layer_0']=cutoff
    with clipping.threshold_hooks(model,thresholds):
        actual=layer.input_layernorm(x)
        assert torch.equal(actual,trained.masked_fill(trained.abs()<=cutoff,0))
        assert bool((actual>=0).all())
    assert torch.equal(layer.input_layernorm(x),trained)


def test_frontier_preserves_ties_and_rejects_dominated_points():
    rows=[{'validation':{'loss':l},'logical_products':{'R_model':s}}
          for l,s in [(1,.1),(1,.1),(2,.3),(2,.2),(3,.3)]]
    assert clipping.nondominated(rows)==[True,True,True,False,False]


def test_complete_missing_cohort_and_source_identities():
    data=json.loads((HERE/'input-manifest.json').read_text())
    rows=data['checkpoints']
    assert {s:sum(r['scale']==s for r in rows) for s in ('14M','70M','410M')}=={'14M':30,'70M':12,'410M':12}
    assert {s:sum(r['scale']==s and r['evaluate'] for r in rows) for s in ('14M','70M','410M')}=={'14M':15,'70M':10,'410M':10}
    assert len({r['id'] for r in rows})==54
    assert all(r['identity']['seeds']=={'model':1234,'data_order':1234} for r in rows)
    assert data['targets']==[i/10 for i in range(10)]
