"""Add the complete 31M Base history to the paper's existing training figure."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'analyses/021-2026-09-10-training-results-figures'
SOURCE = OLD / '06_a0_optimization.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location('original_base_training', SOURCE)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    data_path = OLD / 'data/a0-optimization.json'
    data = json.loads(data_path.read_text())
    assert original.read_evidence() == data
    historical = list(data['runs'])
    points = json.loads((ROOT / 'analyses/037-2026-09-24-14m-31m-70m-latency-quality/data/31m-results.json').read_text())['points']
    base, = [p for p in points if p['scope'] == '0']
    attempt = ROOT / base['source_attempt']
    events_path, manifest_path = attempt / 'events.jsonl', attempt / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    events = [e for e in events if e['event'] == 'train']
    assert len(events) == manifest['completed_steps'] == 712
    assert [e['step'] for e in events] == list(range(1, 713))
    assert all(e['condition_id'] == 'base' for e in events)
    assert not manifest['activation_pressure']['sites']
    assert all(not e['optimizer_step_skipped'] and not e['gradient_overflow'] for e in events)
    assert [e['input_tokens_seen'] for e in events] == [i*1024*2048 for i in range(1, 713)]
    for e in events:
        assert np.isfinite(e['task_loss']) and e['task_loss'] > 0
        assert np.isfinite(e['adamw_gradient_norm_pre_clip']) and e['adamw_gradient_norm_pre_clip'] > 0
        assert e['adamw_gradient_clipping_enabled'] and e['adamw_gradient_clip_norm'] == 1
        assert np.isclose(e['adamw_gradient_norm_post_clip'], min(e['adamw_gradient_norm_pre_clip'], 1), atol=2e-6)
    rows = [{k: e[k] for k in original.FIELDS} for e in events]
    run = dict(scale='31M', baseline_id='Run054/base', condition_id='base',
               events_source=events_path.relative_to(ROOT).as_posix(),
               manifest_source=manifest_path.relative_to(ROOT).as_posix(),
               initial_parameter_sha256=manifest['initial_parameter_sha256'],
               training_schedule_sha256=manifest['data']['training_schedule_hash'],
               tokens_per_step=1024*2048, rows=rows,
               training_tokens_billions=[e['input_tokens_seen']/1e9 for e in rows],
               smoothed_task_loss=original.smooth([e['task_loss'] for e in rows]).tolist(),
               smoothed_gradient_norm_pre_clip=original.smooth([e['adamw_gradient_norm_pre_clip'] for e in rows]).tolist())
    data['runs'].insert(1, run)
    assert [r for r in data['runs'] if r['scale'] != '31M'] == historical
    data['coverage'].update(model_sizes=4, total_step_records=2848)
    for path in (events_path, manifest_path, SOURCE, data_path):
        data['source_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
    original.COLORS['31M'] = '#238B69'
    fig = original.make_figure(data)
    for ax, title in zip(fig.axes, (r'(a) $T_0/P_0$ training loss', r'(b) $T_0/P_0$ gradient norm')):
        ax.set_title(title, loc='left', pad=9)
        assert len(ax.lines) == 8
    # Keep all pre-existing coordinates and axis limits, and fit four legend entries.
    fig.legends[0].remove()
    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc='center', bbox_to_anchor=(.54,.045),
               ncol=4, frameon=False, fontsize=8, handlelength=2.3, columnspacing=2.5)
    assert all(3.5 <= e['task_loss'] <= 11.5 and .15 <= e['adamw_gradient_norm_pre_clip'] <= 35 for e in rows)
    output = HERE / 'figures/03-base-model-training.pdf'
    fig.savefig(output, metadata={'Title': 'Base-model optimization at 14M, 31M, 70M and 410M', 'CreationDate': None, 'ModDate': None})
    original.plt.close(fig)
    data.update(output=output.relative_to(HERE).as_posix(), output_sha256=sha(output),
                source_script_sha256=sha(Path(__file__)), historical_coordinates_unchanged=True)
    (HERE / 'data/base-model-training.json').write_text(json.dumps(data,indent=2)+'\n', encoding='utf-8', newline='\n')
    print('Verified 2,848 updates; unchanged historical raw/smoothed coordinates and axes.')
    print(output)


if __name__ == '__main__':
    main()
