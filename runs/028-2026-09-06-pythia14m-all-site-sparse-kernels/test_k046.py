"""Regression test for the actual projection composition and control forwarding."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(RUN))
spec=importlib.util.spec_from_file_location('k046_composition',RUN/'candidates/k046/candidate.py')
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)


@pytest.mark.parametrize('skip,projection_skip',[(True,True),(False,True),(False,False)])
def test_k036_projection_base_and_skip_controls(monkeypatch,skip,projection_skip):
    calls=[]
    def install(model,**kwargs):
        calls.append(kwargs)
        return {'input_projections':'k036 sentinel'}
    def module(name,path):
        assert path==RUN/'candidates/k036/candidate.py'
        return SimpleNamespace(install=install)
    monkeypatch.setattr(candidate,'module',module)
    monkeypatch.setattr(candidate,'Attention',lambda selected,prefix:SimpleNamespace(skip=selected,shortcut=prefix))
    layers=[SimpleNamespace(attention=SimpleNamespace()) for _ in range(6)]
    model=SimpleNamespace(gpt_neox=SimpleNamespace(layers=layers))
    result=candidate.install(model,shortcut=False,skip=skip,projection_skip=projection_skip)
    assert calls==[{'shortcut':False,'skip':skip,'projection_skip':projection_skip}]
    assert result['input_projections']=='k036 sentinel'
    for layer in layers:
        assert layer.attention._run028_attention.skip==skip
        assert layer.attention._run028_attention.shortcut is False
