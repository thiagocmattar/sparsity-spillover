import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

RUN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RUN))
from io_utils import read, verify, module, record, write, fs
from candidate_catalog import configurations


def test_catalog_complete_and_preselected():
    rows=configurations()
    assert {r['original_iteration'] for r in rows}==set(range(51))
    eligible=[r for r in rows if r['status']=='eligible']
    assert len(eligible)==54
    assert sorted({r['paper_iteration'] for r in eligible if r['candidate']!='p0'})==list(range(1,43))
    masks=[r for r in rows if r['candidate']=='k011']
    assert len(masks)==12 and len({r['paper_iteration'] for r in masks})==1
    assert {r['original_iteration'] for r in rows if r['status']=='excluded'}=={2,7,8,9,10,14,15,16}
    assert read(RUN/'provenance/candidates.json')['configurations']==rows


def test_frozen_source_and_input_identities():
    value=read(RUN/'provenance/archive.json')
    assert len(value['historical_candidate_checks'])==88
    for row in value['files']:
        verify(row['snapshot'])
        assert row['origin']['sha256']==row['snapshot']['sha256']
    inputs=read(RUN/'provenance/inputs.json')
    assert len(inputs['checkpoints'])==35
    assert inputs['validation']['bytes']//4==338*2048+1444
    for row in [inputs['validation']]+[f for c in inputs['checkpoints'] for f in c['files']+c['provenance']]:verify(row)


def test_matrix_coverage_and_determinism():
    matrix=module('run029_matrix_test',RUN/'03_matrix.py')
    args=(read(RUN/'config.json'),configurations(),read(RUN/'provenance/inputs.json')['checkpoints'],'scientific')
    rows=matrix.jobs(*args)
    assert rows==matrix.jobs(*args)
    assert len(rows)==1173==len({r['key'] for r in rows})
    assert sum(r['final'] for r in rows)==525
    assert all({r['replicate'] for r in rows if r['candidate']==c and r['condition']=='c30' and not r['final']}=={1,2,3}
               for c in {r['candidate'] for r in rows if not r['final']})


def test_private_replay_imports():
    script="import replay, pathlib; from sparsity_research import pythia; a=replay.ARCHIVE; assert all(pathlib.Path(m.__file__).resolve().is_relative_to(a) for m in [replay.common,replay.dense,replay.scaffold,pythia]);print('archive-only imports verified')"
    subprocess.run([sys.executable,'-c',script],cwd=RUN,check=True,timeout=120)


def test_timer_and_numerical_contract():
    import replay
    import torch
    samples=[{'mode':m,'repeat':i,'input_index':0,'host_ms':v}
             for i,pair in enumerate([(2.,1.),(8.,2.)]) for m,v in zip(['native_graph','candidate_graph'],pair)]
    assert replay.dense.timing_summary(samples,'native_graph')['candidate_graph']['paired_geomean_speedup']==pytest.approx(8**.5)
    with pytest.raises(ValueError):replay.dense.timing_summary(samples[:-1],'native_graph')
    x=torch.ones(8)
    assert replay.dense.numerical_gate(x,x,relative_l2=.02,atol=.25,rtol=.02)['pass']
    assert not replay.dense.numerical_gate(x,x+1,relative_l2=.02,atol=.25,rtol=.02)['pass']
    assert not replay.dense.numerical_gate(x,x*float('nan'),relative_l2=.02,atol=.25,rtol=.02)['pass']


def test_long_path_atomic_roundtrip(tmp_path):
    target=tmp_path/('a'*80)/('b'*80)/('c'*80)/'identity.json'
    write(target,{'x':1})
    row=record(target,tmp_path)
    assert read(verify(row,tmp_path))=={'x':1}


@pytest.mark.parametrize('row',[r for r in configurations() if r['status']=='eligible'],ids=lambda r:r['id'])
def test_installation_interfaces_cpu(row):
    import copy
    import replay
    import torch
    import transformers
    from sparsity_research.pythia import load_checkpoint_pythia
    with torch.inference_mode():
        model=load_checkpoint_pythia(transformers.AutoModelForCausalLM,RUN/'inputs/checkpoints/c30',torch=torch).to(dtype=torch.bfloat16).eval()
        try:replay.install(copy.deepcopy(model),row)
        except ModuleNotFoundError as exc:
            if sys.platform=='win32' and exc.name=='triton':pytest.skip('Linux-only Triton: required remote test')
            raise


def test_native_control_interfaces_cpu():
    import copy
    import replay
    import torch
    import transformers
    from sparsity_research.pythia import load_checkpoint_pythia
    with torch.inference_mode():
        native=load_checkpoint_pythia(transformers.AutoModelForCausalLM,RUN/'inputs/checkpoints/c01',torch=torch).to(dtype=torch.bfloat16).eval()
        for name in ['p0','k001','k019','k050','k050-no-skip','k050-attention-dense']:
            replay.install(copy.deepcopy(native),replay.final_row(name,configurations()))
        with pytest.raises(ValueError,match='UNSUPPORTED'):
            replay.install(native,replay.final_row('k017',configurations()))
