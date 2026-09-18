"""Prelaunch protocol, cohort and work-count checks; CUDA tests are separate."""
import importlib.util
import json
from pathlib import Path
import sys
import torch

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'033-2026-09-17-a7-h-only-k050-benchmark'

def read(path):return json.loads(path.read_text())
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_matched_contract():
    old,new=read(BASE/'config.json'),read(HERE/'config.json')
    for key in ['gpu','runtime','precision','batch_size','sequence_length','vocabulary','validation_blocks',
        'validation_documents','excluded_tail_tokens','timing_inputs','timing_passes','timing_seed',
        'runtime_seed','process_replicates','numerical_bounds','near_zero_thresholds']:assert new[key]==old[key]
    assert new['final_candidates']==['k050-70m-v1']
    assert read(HERE/'provenance/inputs.json')['validation']['sha256']==read(BASE/'provenance/inputs.json')['validation']['sha256']

def test_complete_available_cohort():
    rows=read(HERE/'provenance/inputs.json')['checkpoints']
    assert len(rows)==22
    assert {r['id'] for r in rows}=={f'c{i:02d}' for i in range(22)}
    for family in ['A4+OL1@all','A7+OL1@all','A4+OL1@h','A7+OL1@h']:
        assert sorted(r['dose'] for r in rows if r['family']==family)==[0,.01,.05,.1,.5]
    for r in rows:
        assert r['canonical_logical_products']['coverage']['sequences']==338
        for f,o in zip(r['files'],r['original_files']):assert (f['bytes'],f['sha256'])==(o['bytes'],o['sha256'])

def test_fresh_process_schedule():
    m=load('run035_queue',HERE/'03_execute.py');jobs=m.jobs()
    assert len(jobs)==len(set(jobs))==66
    assert set(jobs)=={(f'c{i:02d}',r) for i in range(22) for r in range(1,4)}

def test_independent_hybrid_integer_counts():
    # Load only the pure counter function, avoiding CUDA/runtime import changes.
    import ast
    tree=ast.parse((HERE/'diagnostics.py').read_text())
    f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='hybrid_counts')
    ns={'torch':torch};exec(compile(ast.Module(body=[f],type_ignores=[]),'<counter>','exec'),ns)
    count=ns['hybrid_counts'];x=torch.zeros(8,2048)
    assert count(x)==[0,8192,0]
    x[0,2047]=1;x[1,1025]=2
    assert count(x)==[0,8192,1024]
    x[2,[16,1024,2047]]=1
    assert count(x)==[192,8000,1024]
    assert count(x,skip=False)==[8192,0,0]

def test_measurement_driver_changes_are_only_identity_and_diagnostics():
    # All timing/qualification code remains the retained driver, including bounds.
    old=(BASE/'02_benchmark.py').read_text();new=(HERE/'02_benchmark.py').read_text()
    new=new.replace("a.candidate.startswith('k050-70m-')","a.candidate=='k050'")
    new=new.replace("RUN/'diagnostics.py'","replay.R28/'115_hybrid_diagnostics.py'").replace('run035_final_diagnostics','run029_final_diagnostics')
    new=new.replace(",\n            'port_sources': {str(p.relative_to(RUN)): record(p) for p in sorted((RUN/'kernel').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}",'')
    assert new==old
