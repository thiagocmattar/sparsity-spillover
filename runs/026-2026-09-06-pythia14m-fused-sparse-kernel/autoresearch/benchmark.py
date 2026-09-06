"""One immutable variant attempt, with paired dense controls and full logits."""
import argparse
import json
import os
import platform
import sys
import time
import traceback
import numpy as np
import torch
import transformers
from common import (HERE, RUN, ROOT, SOURCE, dense, inputs_manifest, module,
                    read_json, write_json, record, verify_record, timing_summary)
from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--idea', required=True)
    parser.add_argument('--implementation', choices=['dense', 'k001', 'k017', 'k018', 'k019'], default='k017')
    parser.add_argument('--mode', choices=['native', 'graph'], default='graph')
    parser.add_argument('--sites', nargs='+', default=['h', 'z'])
    parser.add_argument('--elements', type=int, choices=[2, 4, 8], default=4)
    parser.add_argument('--warps', type=int, choices=[4, 8], default=4)
    parser.add_argument('--controls', nargs='+', default=['native', 'graph'])
    parser.add_argument('--inputs', type=int, default=16)
    parser.add_argument('--passes', type=int, default=5)
    parser.add_argument('--full-validation', action='store_true')
    parser.add_argument('--final-timing', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--hoist-attention-import', action='store_true')
    parser.add_argument('--emulate-precision-casts', action='store_true')
    parser.add_argument('--joint-with-rope', action='store_true')
    args = parser.parse_args()
    if not args.attempt.replace('-', '').isalnum() or not 1 <= args.inputs <= 64:
        parser.error('Invalid attempt/input count')
    if args.passes < 1 or 'native' not in args.controls or len(set(args.controls)) != len(args.controls):
        parser.error('Positive passes and unique native-inclusive controls required')
    dest = HERE / 'artifacts' / args.attempt
    dest.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'attempt': args.attempt, 'idea': args.idea, 'arguments': vars(args),
              'status': 'running', 'timing': {}, 'primary_speedup': None,
              'canonical_R_model': None, 'qualified': False}

    def emit(stage, **fields):
        row = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'stage': stage, 'elapsed_seconds': time.monotonic()-started, **fields}
        write_json(dest/'status.json', row)
        with (dest/'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
            f.flush()
            os.fsync(f.fileno())
        print(json.dumps(row), flush=True)

    try:
        manifest, checkpoint = inputs_manifest()
        result['canonical_logical_products'] = checkpoint['canonical_logical_products']
        result['canonical_R_model'] = checkpoint['canonical_logical_products']['measured']['R_model']
        cfg = read_json(SOURCE/'config.json')
        for name, actual in [('torch', torch.__version__.split('+')[0]),
                             ('transformers', transformers.__version__), ('numpy', np.__version__)]:
            if actual != cfg['runtime'][name]:
                raise RuntimeError(f'Pinned {name} mismatch: {actual}')
        if not torch.cuda.is_available():
            raise RuntimeError('GPU runtime required')
        torch.manual_seed(2503)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        if args.emulate_precision_casts:
            import torch._inductor.config as inductor_config
            inductor_config.emulate_precision_casts = True
        development = np.memmap(verify_record(manifest['development']), dtype=np.int32, mode='r').reshape(64, 2048)
        validation = np.memmap(verify_record(manifest['validation']), dtype=np.int32, mode='r')
        if divmod(len(validation), 2048) != (338, 1444):
            raise ValueError('Validation coverage changed')
        inputs = [torch.tensor(a.copy(), device='cuda', dtype=torch.long)[None] for a in development[:args.inputs]]
        model = load_checkpoint_pythia(transformers.AutoModelForCausalLM,
                    ROOT/checkpoint['checkpoint'], torch=torch).to('cuda', dtype=torch.bfloat16).eval()
        model.set_attn_implementation('sdpa')
        model.config.use_cache = False
        if args.hoist_attention_import:
            import dense_compatible
            with torch.inference_mode():
                before_hoist = model(input_ids=inputs[0], use_cache=False).logits.clone()
                dense_compatible.install(model)
                after_hoist = model(input_ids=inputs[0], use_cache=False).logits
                if not torch.equal(before_hoist, after_hoist):
                    raise ValueError('Import hoist changed eager logits')
                del before_hoist, after_hoist
        sources = [record(p) for p in HERE.rglob('*') if p.is_file() and p.suffix in {'.py', '.cu'} and 'artifacts' not in p.parts]
        write_json(dest/'manifest.json', {'arguments': vars(args), 'checkpoint': checkpoint,
            'development': manifest['development'], 'validation': manifest['validation'],
            'sources': sources, 'config': record(RUN/'config.json'), 'topology': topology_metadata(model),
            'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
            'capability': list(torch.cuda.get_device_capability()), 'bounds': cfg['calibration'],
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'timer': 'synchronized host full forward/full logits; resident rotating input; equal staging excluded'})
        runners, setup = {}, {}
        with torch.inference_mode():
            for mode in args.controls:
                emit('preparing_dense', mode=mode)
                try:
                    control_model = model
                    if mode.startswith('compile_'):
                        # Keep compiler instrumentation off the native control.
                        control_model = load_checkpoint_pythia(transformers.AutoModelForCausalLM,
                            ROOT/checkpoint['checkpoint'], torch=torch).to('cuda', dtype=torch.bfloat16).eval()
                        control_model.set_attn_implementation('sdpa')
                        control_model.config.use_cache = False
                        if args.hoist_attention_import:
                            dense_compatible.install(control_model)
                    runner = dense.DenseRunner(lambda ids, net=control_model: net(input_ids=ids, use_cache=False).logits,
                                               inputs[0].clone(), mode)
                    runner.prepare()
                    runners[mode] = runner
                    setup[mode] = {'supported': True, 'seconds': runner.setup_seconds}
                except Exception as exc:
                    setup[mode] = {'supported': False, 'error': str(exc), 'traceback': traceback.format_exc()}
                    if mode == 'native':
                        raise
                write_json(dest/'setup.json', setup)
            if args.implementation != 'dense':
                emit('preparing_candidate', implementation=args.implementation)
                # Reload: deepcopy would retain the gate objects captured by
                # Python LayerNorm hook closures from the reference model.
                candidate_model = load_checkpoint_pythia(transformers.AutoModelForCausalLM,
                    ROOT/checkpoint['checkpoint'], torch=torch).to('cuda', dtype=torch.bfloat16).eval()
                candidate_model.set_attn_implementation('sdpa')
                candidate_model.config.use_cache = False
                if args.hoist_attention_import:
                    dense_compatible.install(candidate_model)
                if args.implementation == 'k019':
                    candidate = module('run026_k019', HERE/'candidates/k019/candidate.py')
                    adapter = candidate.Adapter(candidate_model, joint=args.joint_with_rope)
                elif args.implementation == 'k018':
                    candidate = module('run026_k018', HERE/'candidates/k018/candidate.py')
                    adapter = candidate.Adapter(candidate_model)
                    adapter.install()
                elif args.implementation == 'k017':
                    candidate = module('run026_k017', HERE/'candidates/k017/candidate.py')
                    adapter = candidate.Adapter(candidate_model, args.sites, args.elements, args.warps)
                    adapter.set_mode(True)
                else:
                    candidate = module('run026_k001', SOURCE/'autoresearch/candidates/k001/candidate.py')
                    adapter = candidate.Adapter(candidate_model, sites=args.sites)
                    adapter.set_mode('k001')
                runner = dense.DenseRunner(lambda ids: candidate_model(input_ids=ids, use_cache=False).logits,
                                           inputs[0].clone(), args.mode)
                runner.prepare()
                runners['candidate'] = runner
                setup['candidate'] = {'supported': True, 'seconds': runner.setup_seconds}
                write_json(dest/'setup.json', setup)
            for runner in runners.values():
                for ids in inputs:
                    runner.stage(ids)
                    runner()
            emit('development_quality')
            quality = dense.compare_inputs(runners, inputs, cfg['calibration'])
            write_json(dest/'development-quality.json', quality)
            result['development_loss'] = quality['loss']
            result['development_correctness'] = quality['pass']
            emit('timing', loss=quality['loss'], correctness=quality['pass'])
            def sink(row):
                with (dest/'timing-samples.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(row)+'\n')
            # Invalid candidates are still measured and retained, never promoted.
            samples = dense.paired_probe(runners, inputs, passes=args.passes, seed=2504, sink=sink)
            summary = timing_summary(samples)
            valid_dense = [name for name in args.controls if name in summary and quality['pass'][name]]
            baseline = min(valid_dense, key=lambda name: summary[name]['median_host_ms'])
            result.update(timing=summary, selected_dense=baseline,
                          candidate_key='candidate' if 'candidate' in runners else baseline)
            paired = timing_summary(samples, reference=baseline)
            result['primary_speedup'] = paired[result['candidate_key']]['paired_geomean_speedup']
            result['qualified'] = quality['pass'][result['candidate_key']]
            result['quality_scope'] = f'development-{args.inputs}'
            if args.profile:
                dense.attention_profile(lambda ids: model(input_ids=ids, use_cache=False).logits,
                                        inputs[0], dest/'dense-profile.json')
                if 'candidate' in runners:
                    dense.attention_profile(lambda ids: candidate_model(input_ids=ids, use_cache=False).logits,
                                            inputs[0], dest/'candidate-profile.json')
            if args.full_validation:
                emit('full_validation', loss=quality['loss'])
                blocks = (torch.tensor(validation[i*2048:(i+1)*2048].copy(), device='cuda', dtype=torch.long)[None]
                          for i in range(338))
                validation_quality = dense.compare_inputs(runners, blocks, cfg['calibration'],
                    progress=lambda **fields: emit('full_validation', **fields))
                validation_quality.update(documents=500, blocks=338, input_tokens=692224, excluded_tail=1444)
                write_json(dest/'full-validation.json', validation_quality)
                result['qualified'] &= validation_quality['pass'][result['candidate_key']] and validation_quality['pass'][baseline]
                result['quality_scope'] = 'complete-validation'
                result['validation_loss'] = validation_quality['loss']
            if args.final_timing:
                if not args.full_validation:
                    raise ValueError('Final timing requires complete validation')
                indices = np.random.default_rng(2504).choice(338, 64, replace=False)
                final_inputs = [torch.tensor(validation[i*2048:(i+1)*2048].copy(), device='cuda', dtype=torch.long)[None] for i in indices]
                for runner in runners.values():
                    for ids in final_inputs:
                        runner.stage(ids)
                        runner()
                torch.cuda.synchronize()
                final_samples = dense.paired_probe(runners, final_inputs, passes=7, seed=2504)
                write_json(dest/'final-timing.json', {'indices': indices.tolist(), 'samples': final_samples,
                    'summary': timing_summary(final_samples), 'matched': timing_summary(final_samples, reference=baseline)})
            result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            result['status'] = 'complete'
            emit('complete', primary_speedup=result['primary_speedup'], timing=summary,
                 qualified=result['qualified'], loss=result.get('validation_loss', quality['loss']))
    except BaseException as exc:
        result.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        emit('failed', error=str(exc))
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic()-started
        write_json(dest/'result.json', result)


if __name__ == '__main__':
    main()
