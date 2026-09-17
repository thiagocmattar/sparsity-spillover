"""Focused scope, identity, coverage and paired-ratio checks; no CUDA claims."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'runs/029-2026-09-07-pythia14m-matched-kernel-retrospective'


def read(path):return json.loads(path.read_text())


def test_scope_has_only_new_checkpoints_and_final_kernel():
    config=read(HERE/'config.json');inputs=read(HERE/'provenance/inputs.json')
    assert config['final_candidates']==['k050']
    assert config['conditions']==[f'c{i}' for i in range(36,41)]
    assert [r['dose'] for r in inputs['checkpoints']]==[0,.01,.05,.1,.5]
    for row in inputs['checkpoints']:
        assert row['source'].startswith('runs/032-')
        assert row['source_condition']['pressure_sites']==['h']
        assert row['checkpoint'].endswith(row['id'])
        counts=row['canonical_logical_products']
        assert counts['coverage']['sequences']==338
        assert counts['coverage']['excluded_tail_tokens']==1444
        measured=counts['measured']
        assert math.isclose(measured['R_model'],measured['block_zero_product_count']/measured['model_product_count'],abs_tol=1e-15)


def test_scientific_driver_and_kernel_settings_unchanged():
    for name in ['02_benchmark.py','replay.py','io_utils.py']:
        assert (HERE/name).read_bytes()==(BASE/name).read_bytes()
    new=read(HERE/'config.json');old=read(BASE/'config.json')
    for key in ['runtime','precision','batch_size','sequence_length','vocabulary','validation_blocks',
                'validation_documents','excluded_tail_tokens','timing_inputs','timing_passes','timing_seed',
                'runtime_seed','process_replicates','numerical_bounds','near_zero_thresholds']:
        assert new[key]==old[key]
    catalog=read(HERE/'provenance/candidates.json')['configurations']
    assert len(catalog)==1 and catalog[0]['id']=='k050'
    assert catalog[0]['settings']=={'round_p':False,'shortcut':False}


def test_original_weights_and_validation_identity():
    inputs=read(HERE/'provenance/inputs.json')
    assert inputs['validation']['sha256']=='51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451'
    for checkpoint in inputs['checkpoints']:
        for new,old in zip(checkpoint['files'],checkpoint['original_files']):
            assert new['bytes']==old['bytes'] and new['sha256']==old['sha256']
            assert hashlib.sha256((HERE/new['path']).read_bytes()).hexdigest()==new['sha256']


def test_schedule_is_fifteen_unique_fresh_processes():
    spec=importlib.util.spec_from_file_location('run033_queue',HERE/'03_execute.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    rows=mod.jobs()
    assert len(rows)==len(set(rows))==15
    assert set(rows)=={(f'c{i}',r) for i in range(36,41) for r in range(1,4)}
    assert mod.jobs(True)==[('c36',1),('c40',1)]
