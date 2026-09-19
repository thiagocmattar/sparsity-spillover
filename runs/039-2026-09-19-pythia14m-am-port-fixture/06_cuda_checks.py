"""Direct synthetic and captured training-operand qualification; no tuning."""
import traceback
import numpy as np
import torch
from io_utils import RUN, read, write, module, verify
import replay
from site_port import Projection, expected_counts


def main():
    torch.manual_seed(2801)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    cfg = read(RUN/'config.json')
    assert torch.cuda.get_device_name() == cfg['gpu']
    rows = []
    output = {'status': 'running', 'cases': rows}
    old_projection = module('run038_old_projection', replay.R28/'candidates/k042/candidate.py').Projection

    def check(linear, x, label):
        ref = torch.nn.functional.linear(x, linear.weight, linear.bias)
        old = old_projection(linear)
        old_value = old(x).clone()
        for skip in [False, True]:
            port = Projection(linear, skip=skip)
            port.count = True
            value = port(x).clone()
            counts = port.stats.sum((0,1)).cpu().tolist()
            expected = expected_counts(x, linear.out_features, port.fast_weights, skip)
            delta = value.float()-ref.float()
            relative = float(delta.norm()/ref.float().norm().clamp_min(1e-30))
            passed = bool(torch.isfinite(value).all() and torch.allclose(value,ref,atol=.125,rtol=.02)
                          and relative <= .02 and counts == expected)
            row = {'label': label, 'output_width': linear.out_features, 'skip': skip,
                   'max_abs': float(delta.abs().max()), 'relative_l2': relative,
                   'bitwise_native': bool(torch.equal(value,ref)),
                   'bitwise_original_projection': bool(torch.equal(value,old_value)),
                   'counters': counts, 'expected': expected, 'pass': passed}
            rows.append(row)
            write(RUN/'artifacts/cuda-controls.json', output)
            if not passed:
                raise RuntimeError(f'Primitive qualification failed: {label}, skip={skip}')

    try:
        with torch.inference_mode():
            for n in [384,512]:
                linear = torch.nn.Linear(128,n,device='cuda',dtype=torch.bfloat16).eval()
                for pattern in ['zero','one','two','mixed','dense','boundary','signed-zero','unsafe-small']:
                    x = torch.zeros((32,128),device='cuda',dtype=torch.bfloat16)
                    if pattern in ['one','two','mixed','unsafe-small']:
                        x[:,0] = .75
                    if pattern in ['two','mixed','unsafe-small']:
                        x[:,17] = -.5
                    if pattern == 'mixed':
                        x[8:,32:48] = .625
                    if pattern == 'dense':
                        x.normal_(0,.25)
                    if pattern == 'boundary':
                        x[0,15:18]=.75;x[7,127]=-.5;x[8,0]=.625;x[15,31:34]=.875
                    if pattern == 'signed-zero':
                        x[::2] = -0.
                    if pattern == 'unsafe-small':
                        x[:,0] = 2.**-60
                    check(linear,x,'synthetic-'+pattern)
            # Real a/m values come from the first eight of the retained seed2500 training blocks.
            from sparsity_research.pythia import load_checkpoint_pythia
            from sparsity_research.capture import ActivationCapture
            import transformers
            manifest = read(RUN/'provenance/inputs.json')
            endpoint = manifest['checkpoints'][0]
            model = load_checkpoint_pythia(transformers.AutoModelForCausalLM,
                RUN/endpoint['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
            model.set_attn_implementation('sdpa')
            development = np.memmap(verify(manifest['development']),dtype=np.int32,mode='r')
            assert len(development) == 64*2048
            with ActivationCapture(model,['a','m'],torch=torch) as capture:
                for block in range(8):
                    ids=torch.tensor(development[block*2048:(block+1)*2048].copy(),device='cuda',dtype=torch.long)[None]
                    model(input_ids=ids,use_cache=False)
                    for i,layer in enumerate(model.gpt_neox.layers):
                        for site,linear in [('a',layer.attention.query_key_value),('m',layer.mlp.dense_h_to_4h)]:
                            x=capture.activations[f'{site}.layer_{i}'].reshape(-1,128)
                            check(linear,x,f'training-{block}-{site}-layer-{i}')
                    capture.clear()
                    print(f'Captured training checks {block+1}/8; cases={len(rows)}',flush=True)
        output['status']='passed'
    except Exception:
        output.update(status='failed',traceback=traceback.format_exc())
        raise
    finally:
        write(RUN/'artifacts/cuda-controls.json',output)
    print(f'Passed {len(rows)} direct output and counter checks',flush=True)


if __name__=='__main__':main()
