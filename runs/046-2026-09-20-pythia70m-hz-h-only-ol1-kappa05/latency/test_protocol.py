import importlib.util
import json
import hashlib
from pathlib import Path

RUN=Path(__file__).resolve().parent


def test_single_checkpoint_latency_matrix():
    spec=importlib.util.spec_from_file_location('run046_latency_jobs',RUN/'03_execute.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert set(module.jobs())=={(f'c{i:02d}',r) for i in range(1) for r in range(1,4)}
    c=json.loads((RUN/'config.json').read_text())
    assert c['validation_blocks']==338 and c['excluded_tail_tokens']==1444
    assert (c['timing_inputs'],c['timing_passes'],c['process_replicates'])==(64,7,3)
    assert c['numerical_bounds']==dict(logit_atol=.25,logit_rtol=.02,logit_relative_l2=.02,validation_loss_atol=.001)


def test_adapter_changes_only_topology_acceptance():
    original=(RUN/'archive/root/runs/027-2026-09-06-pythia14m-kernel-sparsity-characterization/adapter.py').read_text()
    changed=original.replace("{'A0','A1-H','A4-Z','A7-Z-POST'}","{'A0','A1-H','A4-Z','A7-Z-POST','HZ'}")
    assert changed!=original
    assert (RUN/'hz_adapter.py').read_text()==changed


def test_qualified_70m_kernel_identity_and_local_bridge():
    frozen=json.loads((RUN/'provenance/run035-frozen-final-port.json').read_text())
    source=RUN.parents[2]/'runs/035-2026-09-18-pythia70m-k050-port'
    for name,row in frozen['sources'].items():
        local=RUN/name
        if name=='kernel/candidate.py':
            original=(source/name).read_text()
            assert local.read_text()==original.replace("replay.R27/'adapter.py'", "replay.RUN/'hz_adapter.py'")
        else:
            assert hashlib.sha256(local.read_bytes()).hexdigest()==row['sha256']
    assert (RUN/'diagnostics.py').read_bytes()==(source/'diagnostics.py').read_bytes()
    cfg=json.loads((RUN/'config.json').read_text())
    assert cfg['final_candidates']==['k050-70m-v2']
