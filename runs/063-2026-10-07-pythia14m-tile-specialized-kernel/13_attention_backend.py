"""Local SDPA dispatch and controlled shared-attention substitution diagnostic."""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.nn.attention import SDPBackend, sdpa_kernel

import replay
import topology
from io_utils import RUN, read, verify, write
from sparsity_research.pythia import load_checkpoint_pythia


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    dest = RUN / f'prelaunch/attention-backend-{args.tag}.json'
    if not args.tag.isalnum() or dest.exists():
        parser.error('Fresh simple tag required')
    topology.register()
    cfg, inputs = read(RUN / 'config.json'), read(RUN / 'provenance/inputs.json')
    torch.manual_seed(cfg['runtime_seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    validation = np.memmap(verify(inputs['validation']), dtype=np.int32, mode='r')
    source = inputs['checkpoints'][0]
    groups = {}
    models = []
    with torch.inference_mode():
        ids = torch.tensor(validation[:2048].copy(), device='cuda', dtype=torch.long)[None]
        for mode in ['native', 'generic_sparse', 'tile_sparse']:
            model = load_checkpoint_pythia(transformers.AutoModelForCausalLM,
                                          RUN / source['checkpoint'], torch=torch)
            model = model.to('cuda', dtype=torch.bfloat16).eval()
            model.set_attn_implementation('sdpa')
            model.config.use_cache = False
            if mode != 'native':
                replay.install(model, mode)
            models.append(model)
            if mode == 'native':
                group = replay.dense.DenseRunner(lambda x, m=model: m(input_ids=x, use_cache=False).logits,
                                                ids.clone(), 'native')
                group.prepare()
                groups[mode] = group
            else:
                for layer in model.gpt_neox.layers:
                    layer.attention._run028_attention = NativeAttention()
                group = replay.dense.DenseRunner(lambda x, m=model: replay.scaffold.forward(m, x),
                                                ids.clone(), 'graph')
                group.prepare()
                groups[mode] = group
        stream = (torch.tensor(validation[i*2048:(i+1)*2048].copy(), device='cuda', dtype=torch.long)[None]
                  for i in range(2))
        quality = replay.dense.compare_inputs(groups, stream, cfg['numerical_bounds'])
        # Retain dispatch evidence for the actual first-layer operand layout.
        taps = {}
        attention = models[0].gpt_neox.layers[0].attention
        handles = []
        for site in ['q_post', 'k_post', 'v']:
            def capture(_module, _args, value, site=site):
                taps[site] = value.clone()
            handles.append(getattr(attention, site+'_site').register_forward_hook(capture))
        models[0](input_ids=ids, use_cache=False)
        for handle in handles:
            handle.remove()
        q, k, v = [taps[s] for s in ['q_post', 'k_post', 'v']]
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                              torch.profiler.ProfilerActivity.CUDA]) as profile:
            default = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=attention.scaling)
            torch.cuda.synchronize()
        operators = [item.key for item in profile.key_averages() if 'attention' in item.key]
        cuda_kernels = [item.name for item in profile.events()
                        if item.device_type == torch.autograd.DeviceType.CUDA]
        backends = {}
        for name, backend in [('flash', SDPBackend.FLASH_ATTENTION), ('cudnn', SDPBackend.CUDNN_ATTENTION)]:
            try:
                with sdpa_kernel(backend):
                    result = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=attention.scaling)
                backends[name] = {'equal_default': bool(torch.equal(default, result)),
                                  'max_abs': float((default.float()-result.float()).abs().max())}
            except RuntimeError as exc:
                backends[name] = {'unsupported': str(exc)}
    write(dest, {'gpu': torch.cuda.get_device_name(), 'sm_count': torch.cuda.get_device_properties(0).multi_processor_count,
                 'operators': operators, 'cuda_kernels': cuda_kernels,
                 'backends': backends, 'quality_native_attention': quality,
                 'scope': 'development substitution only; production RTX5090 K035 attention unchanged'})
    print({'operators': operators, 'backends': backends, 'pass': quality['pass']}, flush=True)


class NativeAttention:
    def __call__(self, q, k, v, scale, *, count=False):
        if count:
            raise ValueError('Development substitution has no custom attention counters')
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)


if __name__ == '__main__':
    main()
