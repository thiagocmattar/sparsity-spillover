"""CPU guard, gate-semantics and paired-module wiring regression checks."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
import torch
from sparsity_research.sites import FixedOneSidedThreshold,FixedSymmetricThreshold

RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(RUN))
spec=importlib.util.spec_from_file_location('k050_pair_test',RUN/'candidates/k050/candidate.py')
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)


def test_gate_scalar_dtype_and_boundary():
    for threshold in [0.,.01,.05,.1,.5]:
        gate=FixedOneSidedThreshold(threshold);enabled,rounded=candidate.gate_spec(gate)
        x=torch.tensor([rounded-.0078125,rounded,rounded+.0078125],dtype=torch.bfloat16)
        assert enabled and torch.equal(gate(x),torch.where(x.float()<rounded,0,x.float()).bfloat16())
    assert candidate.gate_spec(None)==(False,0.)
    with pytest.raises(ValueError):candidate.gate_spec(FixedSymmetricThreshold(.1))


def test_norm_pair_validation_and_order():
    pair=candidate.NormPair(torch.nn.LayerNorm(128),torch.nn.LayerNorm(128))
    x=torch.zeros(1,128,dtype=torch.bfloat16)
    with pytest.raises(ValueError):pair.finish(x)
    with torch.inference_mode(),pytest.raises(ValueError):pair.start(x) # CPU rejected
    pair.pending=x;pair.out_m=torch.ones_like(x)
    with pytest.raises(ValueError):pair.finish(x.clone())
    assert pair.finish(x) is pair.out_m and pair.pending is None
    with pytest.raises(ValueError):pair.finish(x)
    with pytest.raises(ValueError):candidate.NormPair(torch.nn.LayerNorm(64),torch.nn.LayerNorm(64))
    with pytest.raises(ValueError):candidate.NormPair(torch.nn.LayerNorm(128,eps=1e-4),torch.nn.LayerNorm(128))


def test_pair_keeps_modules_hooks_and_refreshes_each_call():
    layer=SimpleNamespace(input_layernorm=torch.nn.LayerNorm(128),post_attention_layernorm=torch.nn.LayerNorm(128),
        a_gate=FixedOneSidedThreshold(.5),m_gate=FixedOneSidedThreshold(.5))
    observed=[]
    layer.input_layernorm.register_forward_hook(lambda mod,args,out:layer.a_gate(out))
    layer.post_attention_layernorm.register_forward_hook(lambda mod,args,out:layer.m_gate(out))
    layer.input_layernorm.register_forward_hook(lambda mod,args,out:observed.append(out.clone()))
    pair=SimpleNamespace(start=lambda x:x+2,finish=lambda x:x+3)
    candidate.bind_pair(layer,pair)
    for value in [0.,10.]:
        x=torch.full((1,128),value)
        assert torch.equal(layer.input_layernorm(x),x+2)
        assert torch.equal(layer.post_attention_layernorm(x),x+3)
    assert len(observed)==2 and observed[1][0,0]==12
    assert isinstance(layer.a_gate,FixedOneSidedThreshold) and layer.a_gate.kappa==.5


@pytest.mark.parametrize('skip,projection_skip',[(True,True),(False,True),(False,False)])
def test_same_fusion_for_all_skip_controls(monkeypatch,skip,projection_skip):
    calls=[]
    def module(name,path):
        assert path==RUN/'candidates/k049/candidate.py'
        return SimpleNamespace(install=lambda model,**kwargs:calls.append(kwargs) or {})
    monkeypatch.setattr(candidate,'module',module)
    layers=[SimpleNamespace(input_layernorm=torch.nn.LayerNorm(128),post_attention_layernorm=torch.nn.LayerNorm(128)) for _ in range(6)]
    model=SimpleNamespace(config=SimpleNamespace(use_parallel_residual=True),gpt_neox=SimpleNamespace(layers=layers))
    result=candidate.install(model,shortcut=False,skip=skip,projection_skip=projection_skip)
    assert calls==[{'shortcut':False,'skip':skip,'projection_skip':projection_skip}]
    assert all(hasattr(layer,'_run028_norm_pair') for layer in layers)
    assert 'Identical fusion' in result['normalization_attribution']
