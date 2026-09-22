"""One process compares all predeclared h/z policies with a common native anchor."""
import argparse
import json
import math
import os
import platform
import time
import traceback
from io_utils import RUN, read, write, record, verify


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--condition', required=True)
    p.add_argument('--replicate', type=int, required=True)
    p.add_argument('--attempt', required=True)
    p.add_argument('--phase', choices=('smoke', 'development', 'final'), required=True)
    a = p.parse_args()
    cfg = read(RUN / 'config.json')
    if a.condition not in cfg['conditions'] or not 1 <= a.replicate <= 3:
        p.error('Condition/replicate outside declared comparison')
    if not a.attempt.replace('-', '').isalnum():
        p.error('Simple unique attempt identifier required')
    dest = RUN / 'artifacts/attempts' / a.attempt
    dest.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    checkpoint = next(r for r in read(RUN / 'provenance/inputs.json')['checkpoints'] if r['id'] == a.condition)
    result = {'status': 'running', 'arguments': vars(a), 'checkpoint': checkpoint,
              'config': record(RUN / 'config.json'), 'sources': record(RUN / 'provenance/source-freeze.json')}

    def emit(stage, **fields):
        row = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'stage': stage,
               'condition': a.condition, 'elapsed_seconds': time.monotonic() - started, **fields}
        write(dest / 'status.json', row)
        with (dest / 'events.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row), flush=True)

    try:
        emit('imports')
        import replay
        import install
        import numpy as np
        import torch
        import transformers
        from sparsity_research.pythia import load_checkpoint_pythia
        versions = {'python': '.'.join(platform.python_version().split('.')[:2]),
                    'torch': torch.__version__.split('+')[0], 'transformers': transformers.__version__,
                    'numpy': np.__version__, 'cuda': torch.version.cuda}
        if versions != cfg['runtime'] or torch.cuda.get_device_name() != cfg['gpu']:
            raise RuntimeError(f'Runtime/GPU mismatch: {versions}')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG'):
            raise RuntimeError('Canonical cuBLAS workspace required')
        torch.manual_seed(cfg['runtime_seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        for item in checkpoint['files'] + checkpoint['provenance'] + read(RUN / 'provenance/source-freeze.json')['files']:
            verify(item)
        if a.phase == 'final':
            result['selection'] = record(RUN / 'provenance/selection.json')
        inputs_manifest = read(RUN / 'provenance/inputs.json')
        split = 'validation' if a.phase == 'final' else 'development'
        data_record = inputs_manifest[split]
        data = np.memmap(verify(data_record), dtype=np.int32, mode='r')
        result['data_identity'] = data_record
        if split == 'validation':
            assert divmod(len(data), 2048) == (338, 1444)
            indices = np.random.default_rng(cfg['timing_seed']).choice(338, cfg['timing_inputs'], replace=False)
            blocks, passes = cfg['validation_blocks'], cfg['timing_passes']
        else:
            indices = np.arange(4 if a.phase == 'smoke' else cfg['development_blocks'])
            blocks, passes = len(indices), 2 if a.phase == 'smoke' else cfg['development_passes']
        inputs = [torch.tensor(data[i*2048:(i+1)*2048].copy(), device='cuda', dtype=torch.long)[None] for i in indices]
        models, runners = {}, {}
        with torch.inference_mode():
            for mode in cfg['implementations']:
                emit('loading', implementation=mode)
                model = load_checkpoint_pythia(transformers.AutoModelForCausalLM, RUN / checkpoint['checkpoint'], torch=torch).to('cuda', dtype=torch.bfloat16).eval()
                model.set_attn_implementation('sdpa')
                model.config.use_cache = False
                if mode != 'native':
                    result.setdefault('implementation_metadata', {})[mode] = install.install(model, mode)
                models[mode] = model
                if mode == 'native':
                    runner = replay.dense.DenseRunner(lambda ids, m=model: m(input_ids=ids, use_cache=False).logits, inputs[0].clone(), 'native')
                    runner.prepare()
                    runners['native'] = runner
                emit('capture', implementation=mode)
                runner = replay.dense.DenseRunner(lambda ids, m=model: replay.scaffold.forward(m, ids), inputs[0].clone(), 'graph')
                runner.prepare()
                runners[mode + '_graph'] = runner
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info()
            result['runtime'] = {**versions, 'gpu': torch.cuda.get_device_name(), 'device_uuid': str(props.uuid),
                                 'free_bytes_after_capture': free, 'total_bytes': total}
            if free < 8 * 1024**3:
                raise RuntimeError('Less than 8GiB GPU headroom')
            for runner in runners.values():
                for ids in inputs:
                    runner.stage(ids)
                    runner()
            torch.cuda.synchronize()
            emit('timing', inputs=len(inputs), passes=passes)
            samples = replay.dense.paired_probe({k: v for k, v in runners.items() if k.endswith('_graph')},
                inputs, passes=passes, seed=cfg['timing_seed'] + a.replicate - 1)
            summary = replay.dense.timing_summary(samples, reference='native_graph')
            for mode, values in summary.items():
                x = [r['host_ms'] for r in samples if r['mode'] == mode]
                values['geomean_host_ms'] = math.exp(math.fsum(map(math.log, x)) / len(x))
            write(dest / 'timing.json', {'indices': indices.tolist(), 'samples': samples, 'summary': summary})
            result['timing'] = summary
            emit('validation', target_blocks=blocks)
            stream = (torch.tensor(data[i*2048:(i+1)*2048].copy(), device='cuda', dtype=torch.long)[None] for i in range(blocks))
            quality = replay.dense.compare_inputs(runners, stream, cfg['numerical_bounds'], progress=lambda **f: emit('validation', **f))
            quality.update(blocks=blocks, documents=500 if split == 'validation' else None,
                           input_tokens=blocks*2048, excluded_tail_tokens=1444 if split == 'validation' else None)
            write(dest / 'quality.json', quality)
            result.update(qualification=quality['pass'], all_qualified=all(quality['pass'].values()),
                          loss=quality['loss'], loss_delta=quality['loss_delta'], validation_blocks=blocks)
            if a.phase == 'final' and a.replicate == 1:
                from diagnostics import collect
                for mode in cfg['implementations'][1:]:
                    emit('diagnostics', implementation=mode, loss=quality['loss'])
                    collect(models[mode], models['native'], data, dest / f'diagnostics-{mode}.json', emit,
                            checkpoint['canonical_logical_products']['architecture_maximum'], blocks=blocks)
                from profiling import collect as profile
                emit('profiling', loss=quality['loss'])
                profile(models, runners, inputs[:4], dest)
            result.update(status='complete', peak_allocated_bytes=torch.cuda.max_memory_allocated())
            emit('complete', qualified=result['all_qualified'], loss=quality['loss'], timing_ms={k: v['geomean_host_ms'] for k, v in summary.items()})
    except Exception as exc:
        result.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        emit('failed', error=str(exc))
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        write(dest / 'result.json', result)
        write(dest / 'manifest.json', result)
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
