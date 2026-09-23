"""Local full-model resource/correctness calibration of frozen references."""
import argparse
import math
import platform
import sys
import time
import traceback
from local_support import RUN, event, load, sha, write

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--phase', choices=('smoke', 'reference'), required=True)
    args = parser.parse_args()
    cfg = load(RUN / 'config.json')
    assert args.condition in cfg['conditions']
    assert args.attempt.replace('-', '').isalnum()
    dest = RUN / 'artifacts/attempts' / args.attempt
    dest.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'status': 'running', 'arguments': vars(args), 'config': cfg,
              'config_sha256': sha(RUN / 'config.json'),
              'transfer_manifest_sha256': sha(RUN / 'input-transfer.json'),
              'source_state': load(RUN / 'source-state.json'),
              'scope': 'Local calibration; one process; smoke is not full validation.'}
    def emit(stage, **fields):
        event(dest, stage, condition=args.condition, elapsed_seconds=time.monotonic()-started, **fields)
    write(dest / 'manifest.json', result)
    try:
        emit('imports')
        import numpy as np
        import torch
        import transformers
        import triton
        import os
        versions = {'python': '.'.join(platform.python_version_tuple()[:2]),
                    'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'triton': triton.__version__, 'transformers': transformers.__version__,
                    'numpy': np.__version__}
        assert versions == cfg['runtime'], versions
        assert torch.cuda.get_device_name() == cfg['gpu']
        assert not os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        torch.manual_seed(cfg['runtime_seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        base = RUN / 'deps/run049'
        sys.path.insert(0, str(base))
        import replay
        import install as base_install
        from sparsity_research.pythia import load_checkpoint_pythia
        catalog = load(RUN / 'provenance/inputs.json')
        row = next(r for r in catalog['checkpoints'] if r['id'] == args.condition)
        split = 'development' if args.phase == 'smoke' else 'validation'
        for item in row['files'] + row['provenance'] + [catalog[split]]:
            path = RUN / item['path']
            assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], item['path']
        result.update(checkpoint=row, data_identity=catalog[split])
        data = np.memmap(RUN / catalog[split]['path'], dtype=np.int32, mode='r')
        if args.phase == 'smoke':
            blocks, passes = cfg['smoke_blocks'], cfg['smoke_passes']
            indices = np.arange(blocks)
        else:
            assert divmod(len(data), 2048) == (338, 1444)
            blocks, passes = 338, cfg['timing_passes']
            indices = np.random.default_rng(cfg['timing_seed']).choice(blocks, cfg['timing_inputs'], replace=False)
        inputs = [torch.tensor(data[i*2048:(i+1)*2048].copy(), device='cuda', dtype=torch.long)[None] for i in indices]
        models, runners, memory = {}, {}, []
        with torch.inference_mode():
            for mode in cfg['modes']:
                emit('loading', mode=mode)
                model = load_checkpoint_pythia(transformers.AutoModelForCausalLM, RUN / row['checkpoint'], torch=torch)
                model = model.to('cuda', dtype=torch.bfloat16).eval()
                model.set_attn_implementation('sdpa'); model.config.use_cache = False
                if mode == 'legacy':
                    metadata = replay.install(model, 'legacy')
                elif mode != 'native':
                    metadata = base_install.install(model, 'native_hz' if mode == 'native_hz' else 'opt073')
                    if mode == 'opt073_no_skip':
                        for layer in model.gpt_neox.layers:
                            layer._run026_joint.skip = False
                else:
                    metadata = {'mode': 'native PyTorch'}
                result.setdefault('implementation_metadata', {})[mode] = metadata
                models[mode] = model
                if mode == 'native':
                    runner = replay.dense.DenseRunner(lambda x, m=model: m(input_ids=x, use_cache=False).logits, inputs[0].clone(), 'native')
                    runner.prepare(); runners['native'] = runner
                emit('capture', mode=mode)
                runner = replay.dense.DenseRunner(lambda x, m=model: replay.scaffold.forward(m, x), inputs[0].clone(), 'graph')
                runner.prepare(); runners[mode+'_graph'] = runner
                free, total = torch.cuda.mem_get_info()
                memory.append({'mode': mode, 'free_bytes': free, 'total_bytes': total,
                               'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(),
                               'capture_seconds': runner.setup_seconds})
                write(dest / 'memory.json', memory)
                assert free >= cfg['minimum_free_bytes_after_capture'], 'Insufficient local GPU headroom'
            result['runtime'] = {**versions, 'gpu': torch.cuda.get_device_name(),
                                 'capability': torch.cuda.get_device_capability(), 'kernel': platform.release()}
            for runner in runners.values():
                for ids in inputs:
                    runner.stage(ids); runner()
            torch.cuda.synchronize()
            emit('timing', inputs=len(inputs), passes=passes)
            samples = replay.dense.paired_probe({k:v for k,v in runners.items() if k.endswith('_graph')},
                                               inputs, passes=passes, seed=cfg['timing_seed'])
            summary = replay.dense.timing_summary(samples, reference='native_graph')
            for mode, item in summary.items():
                values = [s['host_ms'] for s in samples if s['mode'] == mode]
                item['geomean_host_ms'] = math.exp(math.fsum(map(math.log, values))/len(values))
            write(dest / 'timing.json', {'split': split, 'indices': indices.tolist(), 'samples': samples, 'summary': summary})
            valid_start = time.monotonic()
            emit('numerics', blocks=0, target_blocks=blocks)
            def progress(**fields):
                elapsed = time.monotonic()-valid_start
                done = fields['blocks']
                emit('numerics', **fields, blocks_per_second=done/elapsed,
                     remaining_seconds=(blocks-done)*elapsed/done)
            stream = (torch.tensor(data[i*2048:(i+1)*2048].copy(), device='cuda', dtype=torch.long)[None] for i in range(blocks))
            quality = replay.dense.compare_inputs(runners, stream, cfg['numerical_bounds'], progress=progress)
            quality.update(blocks=blocks, split=split, documents=500 if blocks==338 else None,
                           input_tokens=blocks*2048, excluded_tail_tokens=1444 if blocks==338 else None)
            write(dest / 'quality.json', quality)
            result.update(qualification=quality['pass'], loss=quality['loss'],
                          numerical_seconds=time.monotonic()-valid_start,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(), memory=memory,
                          timing_ms={k:v['geomean_host_ms'] for k,v in summary.items()})
            if not all(quality['pass'].values()):
                raise ValueError('Numerical qualification failed; retained failed measurements')
            emit('profiling', loss=quality['loss'])
            from io_utils import module
            profile = module('run053_profile', base / 'profiling.py')
            profile.collect(models, runners, inputs[:2], dest)
            result.update(status='complete')
            emit('complete', loss=quality['loss'], timing_ms=result['timing_ms'],
                 numerical_blocks_per_second=blocks/result['numerical_seconds'])
    except Exception as exc:
        result.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        emit('failed', error=str(exc))
    finally:
        result['elapsed_seconds'] = time.monotonic()-started
        write(dest / 'result.json', result)
        write(dest / 'manifest.json', result)
    return 0 if result['status']=='complete' else 1

if __name__ == '__main__':
    raise SystemExit(main())
