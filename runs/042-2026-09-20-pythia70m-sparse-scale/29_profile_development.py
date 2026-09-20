"""Profile retained candidates on training inputs without exposing final validation."""
import argparse
from io_utils import RUN, read, verify, record, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--condition', choices=('c00', 'c21'), default='c21')
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import transformers
    import replay
    from profiling import collect
    from sparsity_research.pythia import load_checkpoint_pythia
    cfg = read(RUN / 'config.json')
    manifest = read(RUN / 'provenance/inputs.json')
    checkpoint = next(r for r in manifest['checkpoints'] if r['id'] == args.condition)
    data = np.memmap(verify(manifest['development']), dtype=np.int32, mode='r')
    dest = RUN / 'artifacts/development' / f'profile-{args.candidate}-{args.condition}-{args.tag}'
    dest.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(cfg['runtime_seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    models, runners = {}, {}
    with torch.inference_mode():
        inputs = [torch.tensor(data[i*2048:(i+1)*2048].copy(), device='cuda', dtype=torch.long)[None] for i in range(4)]
        for mode in ('native', 'candidate'):
            model = load_checkpoint_pythia(transformers.AutoModelForCausalLM, RUN/checkpoint['checkpoint'], torch=torch).to('cuda', dtype=torch.bfloat16).eval()
            model.set_attn_implementation('sdpa')
            model.config.use_cache = False
            if mode == 'candidate':
                replay.install(model, args.candidate)
            models[mode] = model
            runner = replay.dense.DenseRunner(lambda ids, m=model: replay.scaffold.forward(m, ids), inputs[0].clone(), 'graph')
            runner.prepare()
            runners[mode + '_graph'] = runner
            for ids in inputs:
                runner.stage(ids)
                runner()
        torch.cuda.synchronize()
        collect(models, runners, inputs, dest)
    write(dest/'identity.json', {'candidate': args.candidate, 'condition': args.condition,
          'split': 'training-development', 'input_blocks': list(range(4)),
          'data': manifest['development'], 'script': record(__file__),
          'candidate_manifest': record(RUN/'candidates'/args.candidate/'manifest.json'),
          'purpose': 'Instrumented development diagnosis only, not final latency or qualification.'})
    print(dest, flush=True)


if __name__ == '__main__':
    main()
