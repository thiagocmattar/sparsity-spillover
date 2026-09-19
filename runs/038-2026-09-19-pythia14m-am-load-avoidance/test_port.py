"""Focused CPU checks of masks, output mapping and independent sparse counters."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from controls import MODES, settings, contrasts
from site_port import expected_counts


@pytest.mark.parametrize('mode,a,m,skip',[
    ('frozen',False,False,True),('port-a',True,False,True),
    ('port-m',False,True,True),('port-am',True,True,True),('port-am-dense',True,True,False)])
def test_selection(mode,a,m,skip):
    assert settings(mode)=={'a':a,'m':m,'skip':skip}


def test_signed_comparisons():
    values=dict.fromkeys(MODES,2.)
    values.update({'port-a':1.,'port-m':4.,'port-am-dense':3.})
    result=contrasts(values)
    assert result['port-a']=={'saved_ms':1.,'speedup':2.}
    assert result['port-m']=={'saved_ms':-2.,'speedup':.5}
    assert result['port_sparse_effect']=={'saved_ms':1.,'speedup':1.5}
    with pytest.raises(ValueError):contrasts({'frozen':2.})


@pytest.mark.parametrize('n',[384,512])
@pytest.mark.parametrize('skip',[False,True])
def test_independent_counter_patterns(n,skip):
    x=torch.zeros((16,128))
    potential=2*8*(n//8)
    assert expected_counts(x,n,skip=skip)==([0,potential,0] if skip else [potential,0,0])
    x[:,0]=.75;x[:,17]=-.5
    assert expected_counts(x,n,skip=skip)==([0,potential,32*n] if skip else [potential,0,0])
    x[8:,32:48]=.625
    issued=3*(n//8)
    assert expected_counts(x,n,skip=skip)==([issued,potential-issued,16*n] if skip else [potential,0,0])
    x[0,0]=2.**-60
    if skip:
        assert expected_counts(x,n)[2]==14*n


@pytest.mark.parametrize('n',[384,512])
def test_mapping_covers_each_output_once(n):
    coords=[]
    for block_y in range(n//128):
        for warp in range(4):
            for lane in range(32):
                for atom in range(4):
                    for e in range(2):
                        coords.append((lane//4,block_y*128+warp*32+atom*8+(lane%4)*2+e))
    assert len(coords)==len(set(coords))==8*n
    assert set(coords)=={(r,c) for r in range(8) for c in range(n)}


def test_known_checkpoint_and_protocol():
    cfg=json.loads((HERE/'config.json').read_text())
    inputs=json.loads((HERE/'provenance/inputs.json').read_text())
    assert cfg['modes']==list(MODES)
    assert len(inputs['checkpoints'])==1 and inputs['checkpoints'][0]['id']=='c30'
    weight=next(r for r in inputs['checkpoints'][0]['files'] if r['path'].endswith('model.safetensors'))
    assert weight['sha256']=='f83aef36ddbddbd94efb915d574c4645c687206bb68f51886ffc6e979985b425'
    assert inputs['development']['sha256']=='4bd751109b82f203ed904946874d12f1d9bb576874689a7990654a39c2644378'
    assert (cfg['validation_blocks'],cfg['excluded_tail_tokens'])==(338,1444)
    assert (cfg['timing_inputs'],cfg['timing_passes'],cfg['process_replicates'])==(64,7,3)


def test_realized_process_matrix():
    spec=importlib.util.spec_from_file_location('run038_execute',HERE/'03_execute.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    jobs=mod.jobs()
    assert len(jobs)==15 and set(jobs)=={(m,r) for m in MODES for r in [1,2,3]}
    assert jobs==mod.jobs()
