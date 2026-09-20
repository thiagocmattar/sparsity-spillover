"""Training-input full-model correctness checks before validation benchmarking."""
import gc
import torch
import numpy as np
import transformers
import replay
from controls import MODES
from io_utils import RUN, read, write, verify
from sparsity_research.pythia import load_checkpoint_pythia


def main():
    cfg, manifest = read(RUN / 'config.json'), read(RUN / 'provenance/inputs.json')
    torch.manual_seed(cfg['runtime_seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    assert torch.cuda.get_device_name() == cfg['gpu']
    data = np.memmap(verify(manifest['development']), dtype=np.int32, mode='r')
    inputs = [torch.tensor(data[i * 2048:(i + 1) * 2048].copy(), device='cuda', dtype=torch.long)[None]
              for i in range(4)]
    checks = []
    output = {'status': 'running', 'source': manifest['development'], 'blocks': [0, 1, 2, 3], 'checks': checks}
    with torch.inference_mode():
        for checkpoint in manifest['checkpoints']:
            def load():
                model = load_checkpoint_pythia(transformers.AutoModelForCausalLM, RUN / checkpoint['checkpoint'], torch=torch)
                model = model.to('cuda', dtype=torch.bfloat16).eval()
                model.set_attn_implementation('sdpa')
                model.config.use_cache = False
                return model
            native = load()
            refs = [native(input_ids=ids, use_cache=False).logits.clone() for ids in inputs]
            del native
            for mode in MODES:
                model = load()
                replay.install(model, mode)
                for block, (ids, ref) in enumerate(zip(inputs, refs)):
                    out = replay.scaffold.forward(model, ids)
                    delta = out.float() - ref.float()
                    bounds = cfg['numerical_bounds']
                    ratio = float(delta.norm() / ref.float().norm().clamp_min(1e-30))
                    passed = bool(torch.isfinite(out).all()
                                  and (delta.abs() <= bounds['logit_atol'] + bounds['logit_rtol'] * ref.float().abs()).all()
                                  and ratio <= bounds['logit_relative_l2'])
                    checks.append({'condition': checkpoint['id'], 'mode': mode, 'block': block,
                                   'maximum_absolute_error': float(delta.abs().max()), 'relative_l2': ratio,
                                   'bitwise_equal': bool(torch.equal(out, ref)), 'pass': passed})
                    write(RUN / 'artifacts/control-checks.json', output)
                    if not passed: raise RuntimeError(f'Control failed: {checkpoint["id"]}/{mode}/{block}')
                    del out, delta
                del model
                gc.collect()
                torch.cuda.empty_cache()
            del refs
    output['status'] = 'passed'
    write(RUN / 'artifacts/control-checks.json', output)


if __name__ == '__main__': main()
