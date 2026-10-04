"""Focused coverage, checkpoint identity and paired-estimator contract checks."""
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from io_utils import read,verify,module


def test_four_original_conditions_and_gate_sites():
    manifest=read(HERE/'provenance/inputs.json')
    assert {row['id'] for row in manifest['checkpoints']}==set(read(HERE/'config.json')['conditions'])
    identities=set();orders=set()
    for row in manifest['checkpoints']:
        config=row['topology'];condition=row['source_condition']
        assert config['topology_id']=='HZ' and set(config['site_gates'])=={'h','z'}
        assert all(spec=={'operator':'one_sided_threshold','kappa':row['dose']} for spec in config['site_gates'].values())
        assert condition['pressure_sites']==(['h'] if condition['pressure_recipe']=='ph' else ['h','z'])
        for item in row['files']+row['provenance']:verify(item)
        original=read(HERE/f"provenance/originals/{row['id']}/manifest.json")
        assert original['completed_steps']==712
        identities.add(original['initial_parameter_sha256']);orders.add(original['training_schedule_hash'])
    assert len(identities)==len(orders)==1


def test_full_coverage_and_measurement_bounds():
    cfg=read(HERE/'config.json')
    assert (cfg['batch_size'],cfg['sequence_length'],cfg['vocabulary'])==(1,2048,50304)
    assert (cfg['validation_blocks'],cfg['validation_documents'],cfg['excluded_tail_tokens'])==(338,500,1444)
    assert (cfg['timing_inputs'],cfg['timing_passes'],cfg['process_replicates'])==(64,7,3)
    assert cfg['numerical_bounds']==dict(logit_atol=.25,logit_rtol=.02,logit_relative_l2=.02,validation_loss_atol=.001)
    assert verify(read(HERE/'provenance/inputs.json')['validation']).stat().st_size==693668*4


def test_unique_sequential_fresh_process_pairs():
    controller=module('run059_controller_test',HERE/'03_execute.py')
    assert len(controller.jobs())==6 and len(set(controller.jobs()))==6
    assert set(controller.jobs())=={(pair,r) for pair in ['0','0p1'] for r in [1,2,3]}
    assert controller.jobs(True)==[('0',1),('0p1',1)]


def test_unchanged_frozen_kernel_inventory():
    archive=read(HERE/'provenance/archive.json')
    for row in archive['files']:verify(row['snapshot'])
    settings=read(HERE/'provenance/candidates.json')['configurations']
    assert len(settings)==1 and settings[0]['id']=='k050'
    assert settings[0]['settings']=={'round_p':False,'shortcut':False}


def test_pair_estimator_and_invalid_samples():
    import pytest
    measurement=module('run059_measurement_test',HERE/'archive/root/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/measurement.py')
    samples=[{'mode':mode,'repeat':0,'input_index':i,'host_ms':value}
             for i,(ph,p2) in enumerate([(1.,2.),(4.,2.)]) for mode,value in [('ph',ph),('p2',p2)]]
    summary=measurement.timing_summary(samples,reference='ph')
    assert summary['p2']['paired_geomean_speedup']==pytest.approx(1.)
    assert summary['ph']['median_host_ms']==2.5
    with pytest.raises(ValueError):measurement.timing_summary(samples[:-1],reference='ph')
    samples[0]['host_ms']=0
    with pytest.raises(ValueError):measurement.timing_summary(samples,reference='ph')


def test_hz_checkpoint_roundtrip_native():
    import torch
    import transformers
    from sparsity_research.sites import TOPOLOGIES,Topology
    from sparsity_research.pythia import load_checkpoint_pythia,topology_metadata
    prior=TOPOLOGIES.get('HZ');TOPOLOGIES['HZ']=Topology('HZ',('h','z'))
    try:
        checkpoint=HERE/'inputs/checkpoints/t2-p2-kappa-0p1'
        model=load_checkpoint_pythia(transformers.AutoModelForCausalLM,checkpoint,torch=torch).eval()
        assert topology_metadata(model)['active_sites']==['h','z']
        assert all(not hasattr(layer,'a_gate') and not hasattr(layer,'m_gate') for layer in model.gpt_neox.layers)
        with torch.inference_mode():output=model(input_ids=torch.arange(16)[None],use_cache=False).logits
        assert output.shape==(1,16,50304) and torch.isfinite(output).all()
    finally:
        if prior is None:TOPOLOGIES.pop('HZ')
        else:TOPOLOGIES['HZ']=prior
