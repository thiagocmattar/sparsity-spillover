import copy
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('run065_copy_mapping',Path(__file__).with_name('trace_copy_mapping.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def trace():
    eager=[];graph=[]
    for layer in range(6):
        for site in ['h','z']:
            stage=f'layer{layer}.{site}_gate'
            a={'name':stage+'.a','stage':stage};b={'name':stage+'.b','stage':stage}
            eager += [a,b];graph += [dict(a),{'name':'memcpy128' if site=='h' else 'memcpy32_post'},dict(b)]
    return {'mode':'native','mapping':{'mapped':False},'eager':{'kernels':eager},'graph':{'kernels':graph*3,'repeats':3}}

def test_copy_attribution_retains_every_activity():
    result=m.map_native_copies(trace())
    assert result['mapped']
    assert len(result['kernels'])==108
    assert len(result['graph_only_gate_copies'])==12
    assert all(row['stage'] for row in result['kernels'])

def test_reject_changed_computational_kernel():
    value=copy.deepcopy(trace());value['graph']['kernels'][0]['name']='different_computation'
    assert not m.map_native_copies(value)['mapped']

def test_reject_copy_crossing_stage_boundary():
    value=copy.deepcopy(trace());value['eager']['kernels'][1]['stage']='layer0.attention'
    assert not m.map_native_copies(value)['mapped']
