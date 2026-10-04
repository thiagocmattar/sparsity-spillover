"""Gate semantics, original checkpoints, exact coverage and paired evidence."""
import sys
from pathlib import Path
import pytest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from io_utils import read,verify,module,write


def test_checkpoint_identity_and_native_cpu_forward():
    import transformers
    import topology
    topology.register()
    from sparsity_research.pythia import load_checkpoint_pythia,topology_metadata
    inputs=read(HERE/'provenance/inputs.json')
    expected={c for pair in read(HERE/'config.json')['pair_conditions'].values() for c in pair}
    assert {c['id'] for c in inputs['checkpoints']}==expected
    initial=set();schedule=set()
    for c in inputs['checkpoints']:
        for row in c['files']+c['provenance']:
            verify(row)
        original=read(HERE/f"provenance/originals/{c['id']}/manifest.json")
        assert original['status']=='completed' and original['completed_steps']==712
        assert original['condition']['pressure_sites']==['h']
        initial.add(original['initial_parameter_sha256']);schedule.add(original['training_schedule_hash'])
        model=load_checkpoint_pythia(transformers.AutoModelForCausalLM,HERE/c['checkpoint'],torch=torch).eval()
        assert topology_metadata(model)['active_sites']==['h','z']
        assert all(not hasattr(l,'a_gate') and not hasattr(l,'m_gate') for l in model.gpt_neox.layers)
        with torch.inference_mode():
            out=model(input_ids=torch.arange(16)[None],use_cache=False).logits
        assert out.shape==(1,16,50304) and torch.isfinite(out).all()
        if c['family']=='tile':
            with pytest.raises(ValueError,match='cached decoding'):
                model(input_ids=torch.arange(8)[None],use_cache=True)
    assert len(initial)==len(schedule)==1


def test_corner_equality_signed_interior_and_causal_prefix():
    from tile_gate import CornerTileThreshold
    from kernel import gate_spec,gated_value
    from sparsity_research.sites import FixedOneSidedThreshold
    gate=CornerTileThreshold(.1)
    threshold=gate_spec(gate)['threshold']
    x=torch.ones(2,16,32,dtype=torch.bfloat16)
    x[:,0,0]=threshold;x[:,0,16]=1
    x[:,0,17]=-3;x[:,7,18]=-2;x[:,8,0]=1;x[:,8,1]=-4
    actual=gated_value(x,2,threshold)
    assert not actual[:,:8,:16].any()
    assert actual[0,0,17]==-3 and actual[0,7,18]==-2 and actual[0,8,1]==-4
    assert torch.equal(actual,gate(x))
    changed=x.clone();changed[:,12:]=0
    assert torch.equal(gate(x)[:,:12],gate(changed)[:,:12])
    assert torch.equal(gate(x[:,:5]),gate(x)[:,:5])
    scalar=FixedOneSidedThreshold(.1)
    scalar_result=gated_value(x,1,gate_spec(scalar)['threshold'])
    assert scalar_result[0,0,0]==x[0,0,0] and scalar_result[0,0,17]==0


def test_optimized_installation_retains_weights_and_fuses_exact_gate():
    import replay
    import topology
    import transformers
    topology.register()
    from sparsity_research.pythia import load_checkpoint_pythia
    from kernel import gate_spec
    for source in read(HERE/'provenance/inputs.json')['checkpoints']:
        for mode in ['sparse','no_skip']:
            model=load_checkpoint_pythia(transformers.AutoModelForCausalLM,HERE/source['checkpoint'],torch=torch).eval()
            weights=[(l.mlp.dense_4h_to_h.weight,l.attention.dense.weight) for l in model.gpt_neox.layers]
            gates=[(gate_spec(l.mlp.act),gate_spec(l.attention.z_gate)) for l in model.gpt_neox.layers]
            with torch.inference_mode():
                metadata=replay.install(model,mode)
            assert metadata['hz_skip']==(mode=='sparse')
            for layer,(wh,wz),(gh,gz) in zip(model.gpt_neox.layers,weights,gates):
                joint=layer._run026_joint
                assert joint.w2.weight is wh and joint.wo.weight is wz
                assert joint.gh==gh['mode'] and joint.gz==gz['mode']
                assert joint.th==gh['threshold'] and joint.tz==gz['threshold']
                assert joint.skip==(mode=='sparse')
                x=torch.tensor([[[-3.,0.,.1,1.]]])
                assert torch.equal(layer.mlp.act(x),x) and torch.equal(layer.attention.z_gate(x),x)
                assert layer.attention._run028_attention.skip is True
                assert layer.attention._run028_attention.shortcut is False


def test_exact_full_data_coverage_and_frozen_sources():
    inputs=read(HERE/'provenance/inputs.json')
    path=verify(inputs['validation'])
    assert divmod(path.stat().st_size//4,2048)==(338,1444)
    assert inputs['validation']['sha256']=='51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451'
    for row in read(HERE/'provenance/archive.json')['files']:
        verify(row['snapshot'])


def test_unique_fresh_process_cohort_and_staging_scope():
    controller=module('run061_controller_test',HERE/'03_execute.py')
    assert set(controller.jobs())=={(p,r) for p in ['0','0p1'] for r in [1,2,3]}
    assert len(controller.jobs())==len(set(controller.jobs()))==6
    assert controller.jobs(True)==[('0',1),('0p1',1)]
    cfg=read(HERE/'config.json')
    assert cfg['modes']==['native','no_skip','sparse']
    assert (cfg['timing_inputs'],cfg['timing_passes'],cfg['process_replicates'])==(64,7,3)


def test_paired_ratio_and_invalid_samples():
    measurement=module('run061_measurement_test',HERE/'archive/root/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/measurement.py')
    samples=[{'mode':m,'repeat':0,'input_index':i,'host_ms':v}
             for i,(a,b) in enumerate([(1.,2.),(4.,2.)]) for m,v in [('scalar',a),('tile',b)]]
    assert measurement.timing_summary(samples,reference='scalar')['tile']['paired_geomean_speedup']==pytest.approx(1.)
    with pytest.raises(ValueError):
        measurement.timing_summary(samples[:-1],reference='scalar')
    samples[0]['host_ms']=float('nan')
    with pytest.raises(ValueError):
        measurement.timing_summary(samples,reference='scalar')


def test_count_pooling_and_evidence_serialization(tmp_path):
    from sparsity_research.metrics import ActivationAccumulator
    acc=ActivationAccumulator((0.,.001,.01))
    acc.update({'h.layer_0':torch.tensor([0.,1.]),'h.layer_1':torch.tensor([0.,0.,0.,1.,1.,1.])},torch=torch)
    row=acc.pooled_by_site()[0]
    assert row['exact_zero_count']==4 and row['total']==8
    path=tmp_path/'evidence.json';write(path,{'counts':row})
    assert read(path)['counts']==row
    assert b'\r' not in path.read_bytes()
