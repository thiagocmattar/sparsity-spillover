"""The scientific denominator is a fixed native base, not each recipe's native time."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


@pytest.mark.parametrize('duplicate', [False, True])
def test_fixed_base_reference_and_duplicate_rejection(tmp_path, monkeypatch, duplicate):
    spec = importlib.util.spec_from_file_location('run040_reduce', HERE / '07_reduce.py')
    reducer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reducer)
    monkeypatch.setattr(reducer, 'RUN', tmp_path)
    monkeypatch.setattr(reducer, 'record', lambda path: {'path': str(path.relative_to(tmp_path))})
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    save(tmp_path / 'config.json', {'conditions': ['c00', 'c21'], 'diagnostic_modes': ['full']})
    save(tmp_path / 'provenance/timing-indices.json', list(range(64)))
    for condition, native, custom in [('c00', 1.6, 3.2), ('c21', 2., 1.6)]:
        for rep in (1, 2, 3):
            folder = tmp_path / 'artifacts/attempts' / f'{condition}-r{rep}'
            save(folder / 'result.json', {'arguments': {'replicate': rep}, 'status': 'complete',
                 'qualified': True, 'validation_blocks': 338, 'runtime': {'device_uuid': 'same-gpu'},
                 'condition': condition, 'candidate': 'full'})
            save(folder / 'quality.json', {'blocks': 338, 'documents': 500, 'excluded_tail_tokens': 1444,
                 'prediction_tokens': 691886, 'pass': {'native_graph': True, 'candidate_graph': True}})
            samples = [{'input_index': i, 'repeat': p, 'mode': mode, 'host_ms': t, 'cuda_ms': t,
                        'output_shape': [1, 2048, 50304]} for i in range(64) for p in range(7)
                       for mode, t in [('native_graph', native), ('candidate_graph', custom)]]
            if duplicate: samples.append(samples[0])
            save(folder / 'timing.json', {'indices': list(range(64)), 'samples': samples})
    if duplicate:
        with pytest.raises(AssertionError, match='Duplicate'): reducer.reduce_attempts()
    else:
        result = reducer.reduce_attempts()
        sparse, = [r for r in result['rows'] if r['condition'] == 'c21']
        assert result['native_base_reference_ms'] == pytest.approx(1.6)
        assert sparse['native_base_speedup'] == pytest.approx(1.)
        assert sparse['same_checkpoint_native_speedup'] == pytest.approx(1.25)
