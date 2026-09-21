"""One fresh process: pair all four h/z controls, fixed K050, and native graphs."""
import argparse
import json
import os
import platform
import time
import traceback
from io_utils import RUN, read, write, record, verify, module
from controls import MODES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replicate', required=True, type=int)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = read(RUN/'config.json')
    assert args.attempt.replace('-', '').isalnum() and 1 <= args.replicate <= 3
    dest = RUN/'artifacts/attempts'/args.attempt
    dest.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    inputs_manifest = read(RUN/'provenance/inputs.json')
    checkpoint, = inputs_manifest['checkpoints']
    result = {'status': 'running', 'arguments': vars(args), 'checkpoint': checkpoint,
              'config': record(RUN/'config.json'), 'execution_sources':
              {p.name: record(p) for p in RUN.glob('*.py')}, 'installation': {}}

    def emit(stage, **fields):
        row = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'stage': stage, 'elapsed_seconds': time.monotonic()-started, **fields}
        write(dest/'status.json', row)
        with (dest/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row), flush=True)

    try:
        emit('imports')
        import replay
        from run041_topology import register
        register()
        import numpy as np
        import torch
        import transformers
        from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata
        runtime = {'python': '.'.join(platform.python_version().split('.')[:2]),
                   'torch': torch.__version__.split('+')[0], 'transformers': transformers.__version__,
                   'numpy': np.__version__, 'cuda': torch.version.cuda}
        assert runtime == cfg['runtime'], runtime
        assert torch.cuda.get_device_name() == cfg['gpu']
        assert not os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        torch.manual_seed(cfg['runtime_seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        for row in checkpoint['files'] + checkpoint['provenance']:
            verify(row)
        validation = np.memmap(verify(inputs_manifest['validation']), dtype=np.int32, mode='r')
        assert divmod(len(validation), 2048) == (338, 1444)
        indices = np.random.default_rng(cfg['timing_seed']).choice(338, cfg['timing_inputs'], replace=False)
        if args.smoke:
            indices = indices[:4]
        def ids_at(index):
            return torch.tensor(validation[index*2048:(index+1)*2048].copy(),
                                device='cuda', dtype=torch.long)[None]
        inputs = [ids_at(i) for i in indices]
        catalog = read(RUN/'provenance/candidates.json')['configurations']
        models, runners = {}, {}
        with torch.inference_mode():
            for mode in ['native', 'frozen', *MODES]:
                emit('loading', mode=mode)
                model = load_checkpoint_pythia(transformers.AutoModelForCausalLM,
                    RUN/checkpoint['checkpoint'], torch=torch).to('cuda', dtype=torch.bfloat16).eval()
                model.set_attn_implementation('sdpa')
                model.config.use_cache = False
                if mode != 'native':
                    result['installation'][mode] = replay.install(model, mode, catalog)
                models[mode] = model
                if mode == 'native':
                    anchor = replay.dense.DenseRunner(lambda ids, m=model:
                        m(input_ids=ids, use_cache=False).logits, inputs[0].clone(), 'native')
                    anchor.prepare()
                    runners['native'] = anchor
                emit('capture', mode=mode)
                runner = replay.dense.DenseRunner(lambda ids, m=model: replay.scaffold.forward(m, ids),
                                                  inputs[0].clone(), 'graph')
                runner.prepare()
                runners[mode+'_graph'] = runner
            runtime.update(gpu=torch.cuda.get_device_name(),
                gpu_properties=str(torch.cuda.get_device_properties(0)),
                device_uuid=str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unavailable')),
                torch_git=torch.version.git_version, cpu_threads=torch.get_num_threads(),
                topology=topology_metadata(models['native']),
                setup_seconds={k: r.setup_seconds for k, r in runners.items()})
            result['runtime'] = runtime
            write(dest/'manifest.json', result)
            for runner in runners.values():
                for ids in inputs:
                    runner.stage(ids)
                    runner()
            torch.cuda.synchronize()
            emit('timing', inputs=len(inputs), passes=2 if args.smoke else cfg['timing_passes'])
            samples = replay.dense.paired_probe({k: r for k, r in runners.items() if k.endswith('_graph')},
                inputs, passes=2 if args.smoke else cfg['timing_passes'],
                seed=cfg['timing_seed']+args.replicate-1)
            write(dest/'timing.json', {'indices': indices.tolist(), 'samples': samples,
                'summary': replay.dense.timing_summary(samples, reference='D_graph')})
            blocks = 8 if args.smoke else cfg['validation_blocks']
            emit('validation', target_blocks=blocks)
            quality = replay.dense.compare_inputs(runners, (ids_at(i) for i in range(blocks)),
                cfg['numerical_bounds'], progress=lambda **f: emit('validation', **f))
            quality.update(blocks=blocks, documents=500 if blocks==338 else None,
                           input_tokens=blocks*2048, excluded_tail_tokens=1444 if blocks==338 else None)
            # Zero error against a common eager anchor proves cross-mode value agreement.
            exact = {m: all(g['max_abs'] == 0 for g in gates) for m, gates in quality['gates'].items()}
            quality['exact_eager_agreement'] = exact
            write(dest/'quality.json', quality)
            result.update(qualified=all(quality['pass'].values()), exact_eager_agreement=exact,
                          loss=quality['loss'], validation_blocks=blocks)
            if not result['qualified']:
                raise RuntimeError('Full-model numerical qualification failed')
            if not all(exact.values()):
                raise RuntimeError('Nonzero output difference: investigate before causal attribution')
            if args.replicate == 1:
                diagnostic = module('run048_diagnostics', RUN/'diagnostics.py')
                for mode in MODES:
                    emit('diagnostics', mode=mode, loss=quality['loss'])
                    diagnostic.collect(models[mode], models['native'], validation,
                        dest/f'diagnostics-{mode}.json', lambda stage, **f: emit(stage, mode=mode, **f),
                        checkpoint['canonical_logical_products']['architecture_maximum'], blocks=blocks)
                diagnostics = {mode: read(dest/f'diagnostics-{mode}.json') for mode in MODES}
                hashes = {m: d['hz_operand_hashes'] for m, d in diagnostics.items()}
                same = all(h == hashes['A'] for h in hashes.values())
                write(dest/'zero-pattern-check.json', {'identical_values_and_masks': same,
                    'blocks': blocks, 'hashes': hashes})
                if not same:
                    raise RuntimeError('h/z operand values or zero masks changed across execution modes')
            result.update(status='complete', peak_allocated_bytes=torch.cuda.max_memory_allocated())
            emit('complete', qualified=True, loss=result['loss'])
    except Exception as exc:
        result.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        emit('failed', error=str(exc))
    finally:
        result['elapsed_seconds'] = time.monotonic()-started
        write(dest/'result.json', result)
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
