"""Focused mathematical certificates and frozen-parameter adapter contracts."""
import importlib.util
from pathlib import Path
import sys
import pytest
import torch

RUN=Path(__file__).resolve().parent


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


@pytest.fixture(scope='module')
def modules():
    previous=sys.modules.get('io_utils')
    sys.modules['io_utils']=module('_run067_io',RUN/'io_utils.py')
    try:
        yield module('_run067_head',RUN/'head.py'),module('_run067_diagnostics',RUN/'diagnostics.py')
    finally:
        if previous is None:sys.modules.pop('io_utils',None)
        else:sys.modules['io_utils']=previous


def gate():
    source=RUN.parent/'066-2026-10-08-pythia14m-vocabulary-tile-gates/tile_gate.py'
    spec=importlib.util.spec_from_file_location('run067_frozen_gate',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.CornerTileThreshold


@pytest.mark.parametrize('kappa',[0.,.1])
def test_native_corner_certificate_survives_fp16_rounding(kappa,modules):
    _,diagnostics=modules
    value=torch.ones(1,16,128)
    value[:,::8,::16]=kappa
    survivor=torch.nextafter(torch.tensor(kappa),torch.tensor(float('inf')))
    value[:,0,0]=survivor;value[:,:,1]=-3.
    result=gate()(kappa)(value)
    assert result[0,0,0]>kappa and result[0,0,0].half()<=kappa
    assert result[0,1,1]==-3 and result[0,8:].count_nonzero()==0
    counts=diagnostics.counts(result,kappa,16,256,vocabulary=263)
    assert counts['gate_tiles']==16 and counts['empty_gate_tiles']==15
    assert counts['consumer_tiles']==8 and counts['empty_consumer_tiles']==7
    assert counts['activation_zero_products']==int(result.eq(0).sum())*263


def test_certificate_rejects_a_nonempty_rejected_tile(modules):
    _,diagnostics=modules
    value=torch.zeros(1,16,128);value[0,1,1]=1
    with pytest.raises(ValueError,match='certificate'):diagnostics.counts(value,0.,16,256)


def test_integer_pooling_does_not_average_tile_fractions(modules):
    _,diagnostics=modules
    values=[gate()(0.)(torch.ones(1,t,128)) for t in (16,32)]
    values[1].zero_()
    pooled=diagnostics.pool([diagnostics.counts(x,0.,16,256) for x in values])
    assert pooled['empty_gate_tiles']==32 and pooled['gate_tiles']==48
    assert pooled['skipped_mma']+pooled['issued_mma']==pooled['potential_mma']


@pytest.mark.parametrize('bad',[{'operator':'scalar','kappa':0,'tile_tokens':8,'tile_features':16},
                              {'operator':'tile_corner_threshold','kappa':True,'tile_tokens':8,'tile_features':16},
                              {'operator':'tile_corner_threshold','kappa':float('nan'),'tile_tokens':8,'tile_features':16}])
def test_rejects_other_gate_semantics(bad,modules):
    head,_=modules
    with pytest.raises(ValueError):head.validate_gate(bad)


def test_dense_adapter_retains_parameter_and_checkpoint_names(modules):
    head_module,_=modules
    linear=torch.nn.Linear(128,263,bias=False)
    head=head_module.DenseHead(linear)
    assert head.weight is linear.weight
    assert set(head.state_dict())=={'weight'}
    assert torch.equal(head.state_dict()['weight'],linear.state_dict()['weight'])


@pytest.mark.parametrize('vocabulary',[8,263,50304])
def test_weight_fragment_packing_preserves_every_weight_and_zero_pads(modules,vocabulary):
    head,_=modules
    weight=torch.randn(vocabulary,128).half()
    packed=head.pack_weight(weight)
    inverse=packed.reshape(8,-1,2,8,8).permute(1,3,0,2,4).reshape(-1,128)
    assert torch.equal(inverse[:vocabulary],weight)
    assert inverse[vocabulary:].count_nonzero()==0


def test_accepted_manifests_require_deployment_commit_and_runtime_identity(modules,monkeypatch):
    io=sys.modules['io_utils']
    monkeypatch.delenv('RUN067_SOURCE_COMMIT',raising=False)
    with pytest.raises(ValueError,match='Git commit'):io.source_identity(development=False)
    synthetic_commit='f'*40
    monkeypatch.setenv('RUN067_SOURCE_COMMIT',synthetic_commit)
    identity=io.source_identity(development=False)
    assert identity['git_commit']==synthetic_commit
    assert identity['runtime_inventory']==io.record(RUN/'provenance/upload-inventory.json')
