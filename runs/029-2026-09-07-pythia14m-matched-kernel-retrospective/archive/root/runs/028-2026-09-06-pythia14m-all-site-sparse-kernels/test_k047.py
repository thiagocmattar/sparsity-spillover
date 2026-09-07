import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(RUN))
spec=importlib.util.spec_from_file_location('k047_composition',RUN/'candidates/k047/candidate.py')
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)


@pytest.mark.parametrize('skip,projection_skip',[(True,True),(False,True),(False,False)])
def test_k042_inputs_and_k039_outputs(monkeypatch,skip,projection_skip):
    assert Path(candidate.output.__file__)==RUN/'candidates/k039/candidate.py'
    calls=[]
    def install(model,**kwargs):
        calls.append(kwargs)
        for layer in model.gpt_neox.layers:layer._run026_joint=SimpleNamespace(skip=kwargs['projection_skip'])
        return {'input_projections':'k042 sentinel','attention':'k035 sentinel'}
    def module(name,path):
        assert path==RUN/'candidates/k042/candidate.py'
        return SimpleNamespace(install=install)
    monkeypatch.setattr(candidate,'module',module)
    monkeypatch.setattr(candidate.output,'Joint',lambda old:SimpleNamespace(skip=old.skip,implementation='k039'))
    layers=[SimpleNamespace() for _ in range(6)];model=SimpleNamespace(gpt_neox=SimpleNamespace(layers=layers))
    result=candidate.install(model,shortcut=False,skip=skip,projection_skip=projection_skip)
    assert calls==[{'shortcut':False,'skip':skip,'projection_skip':projection_skip}]
    assert result['input_projections']=='k042 sentinel' and result['attention']=='k035 sentinel'
    assert all(layer._run026_joint.skip==projection_skip and layer._run026_joint.implementation=='k039' for layer in layers)
