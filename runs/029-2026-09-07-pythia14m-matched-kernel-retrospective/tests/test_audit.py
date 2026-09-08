"""Independent audit checks and a documented, out-of-cohort gate counterexample."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN))
from io_utils import module, read
audit = module('run029_independent_audit_test', RUN / '19_audit_evidence.py')


def test_actual_cohort_thresholds_have_no_float_vs_bf16_boundary_mismatch():
    thresholds = set()
    for checkpoint in read(RUN / 'provenance/inputs.json')['checkpoints']:
        cfg = read(RUN / checkpoint['checkpoint'] / 'config.json')
        specs = [cfg.get('site_gate')] + list((cfg.get('site_gates') or {}).values())
        thresholds.update(s['kappa'] for s in specs if s and 'kappa' in s)
    assert thresholds == {0., .01, .05, .1, .5}
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    values = values[torch.isfinite(values)]
    assert values.numel() == 65280
    for threshold in thresholds:
        # Compare all representable finite BF16 operands, not a few decimal samples.
        assert torch.equal(values < threshold, values.float() < threshold)
        assert torch.equal(values.abs() < threshold, values.float().abs() < threshold)


def test_noncohort_threshold_counterexample_is_real_and_not_a_fixed_kernel():
    value = torch.tensor([.7], dtype=torch.bfloat16)
    assert value.item() == .69921875
    native = value.masked_fill(value < .7, 0.)
    fused_semantics = value.masked_fill(value.float() < .7, 0.)
    assert native.item() == value.item() and fused_semantics.item() == 0.
    # K050's a/m pair already rounds explicitly; inherited q/k/v and h/z do not.
    rounded = float(torch.tensor(.7, dtype=torch.bfloat16))
    assert torch.equal(native, value.masked_fill(value.float() < rounded, 0.))


def test_independent_fit_and_duplicate_timer_detection():
    rows = [{'R_model': x, 'speedup': 1 + 2*x} for x in [0., .1, .2, .3]]
    fit = audit.regression(rows)
    assert fit['intercept'] == pytest.approx(1.)
    assert fit['slope'] == pytest.approx(2.) and fit['r_squared'] == pytest.approx(1.)
    # Synthetic in-memory test data only; never serialized as benchmark evidence.
    sample = {'repeat': 0, 'input_index': 0, 'output_shape': [1, 2048, 50304], 'cuda_ms': 1.}
    timing = {'samples': [{**sample, 'mode': 'native_graph', 'host_ms': 2.},
                          {**sample, 'mode': 'candidate_graph', 'host_ms': 1.}]}
    assert audit.timing_ratios(timing, {'timing_passes': 1, 'timing_inputs': 1})['host_ms'] == [2.]
    timing['samples'].append(timing['samples'][0])
    with pytest.raises(AssertionError):
        audit.timing_ratios(timing, {'timing_passes': 1, 'timing_inputs': 1})


def test_recomputed_quality_rejects_tampered_summary_flags():
    cfg = read(RUN / 'config.json')
    gate = {'finite': True, 'elementwise_gate': True, 'relative_l2': 0., 'pass': True}
    quality = {'blocks': 338, 'documents': 500, 'excluded_tail_tokens': 1444,
               'prediction_tokens': 338*2047,
               'loss': {m: 5. for m in ['native', 'native_graph', 'candidate_graph']},
               'loss_delta': {m: 0. for m in ['native', 'native_graph', 'candidate_graph']},
               'pass': {m: True for m in ['native', 'native_graph', 'candidate_graph']},
               'gates': {m: [{**gate, 'input_index': i} for i in range(338)]
                         for m in ['native_graph', 'candidate_graph']}}
    assert audit.qualify(quality, cfg)
    altered = deepcopy(quality)
    altered['gates']['candidate_graph'][0]['relative_l2'] = .1
    with pytest.raises(AssertionError):
        audit.qualify(altered, cfg)
