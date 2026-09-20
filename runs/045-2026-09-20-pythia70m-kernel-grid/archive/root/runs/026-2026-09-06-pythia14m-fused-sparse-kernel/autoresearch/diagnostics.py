"""Untimed native-BF16 validation diagnostics; never replace canonical R_model."""
import time
import numpy as np
import torch
import transformers
from common import HERE, ROOT, RUN, dense, inputs_manifest, module, record, verify_record, write_json
from sparsity_research.capture import ActivationCapture
from sparsity_research.metrics import ActivationAccumulator, weight_statistics
from sparsity_research.pythia import load_checkpoint_pythia


def main():
    dest = RUN/'runtime/diagnostics'
    dest.mkdir(exist_ok=False)
    manifest, checkpoint = inputs_manifest()
    data = np.memmap(verify_record(manifest['validation']), dtype=np.int32, mode='r')
    assert divmod(len(data), 2048) == (338, 1444)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model = load_checkpoint_pythia(transformers.AutoModelForCausalLM,
        ROOT/checkpoint['checkpoint'], torch=torch).to('cuda', dtype=torch.bfloat16).eval()
    model.set_attn_implementation('sdpa')
    model.config.use_cache = False
    result = {'source': record(__file__), 'checkpoint': checkpoint,
        'coverage': {'documents': 500, 'blocks': 338, 'input_tokens': 692224, 'excluded_tail': 1444},
        'interpretation': 'Untimed canonical native BF16 SDPA; fused internal ports are not captured. Canonical R_model remains the separate source artifact; no eager-attention substitution.',
        'weights': weight_statistics(model)}
    accumulator = ActivationAccumulator((0., .001, .01))
    occupancy = {}
    started = time.monotonic()
    with torch.inference_mode(), ActivationCapture(model, ['a','m','h','z','q_post','k_post','v'], torch=torch) as capture:
        for i in range(338):
            ids = torch.tensor(data[i*2048:(i+1)*2048].copy(), device='cuda', dtype=torch.long)[None]
            model(input_ids=ids, use_cache=False)
            accumulator.update(capture.activations, torch=torch)
            for name, value in capture.activations.items():
                if name.split('.')[0] not in {'h', 'z'}:
                    continue
                width = value.shape[-1]
                histogram = torch.bincount(torch.count_nonzero(value.reshape(-1, width), dim=-1), minlength=width+1)
                occupancy.setdefault(name, torch.zeros_like(histogram)).add_(histogram)
            capture.clear()
            if (i+1) % 32 == 0:
                elapsed = time.monotonic()-started
                print({'stage': 'diagnostics', 'blocks': i+1, 'seconds': elapsed,
                       'blocks_per_second': (i+1)/elapsed, 'etc_seconds': elapsed*(338-i-1)/(i+1),
                       'loss': 'not recomputed; frozen complete-validation loss=5.83130066301001'}, flush=True)
    result.update(per_site_layer=accumulator.rows(), pooled_by_site=accumulator.pooled_by_site(),
        active_features_per_row={key: value.cpu().tolist() for key, value in occupancy.items()},
        elapsed_seconds=time.monotonic()-started)
    write_json(dest/'native-bf16.json', result)
    # All capture hooks have been removed. Profile independently after timing.
    candidate = module('run026_diagnostic_winner', HERE/'candidates/k019/candidate.py')
    adapter = candidate.Adapter(model, joint=True)
    with torch.inference_mode():
        for _ in range(5): model(input_ids=ids, use_cache=False)
        dense.attention_profile(lambda inputs: model(input_ids=inputs, use_cache=False).logits,
                                ids, dest/'winner-profile.json')
    write_json(dest/'resources.json', {'gpu': torch.cuda.get_device_name(),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'free_total_bytes': list(torch.cuda.mem_get_info()),
        'scope': 'diagnostic process including capture and profiler; timed-process peaks are in each result.json'})
    print('Diagnostics complete', flush=True)


if __name__ == '__main__': main()
