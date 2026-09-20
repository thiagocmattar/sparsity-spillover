import importlib.util
import json
from pathlib import Path

RUN=Path(__file__).resolve().parent


def test_single_checkpoint_latency_matrix():
    spec=importlib.util.spec_from_file_location('run044_latency_jobs',RUN/'03_execute.py')
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
