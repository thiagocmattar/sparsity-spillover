"""Untimed, equal-input component comparisons on the failing real checkpoint."""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
import transformers

import replay
import topology
from io_utils import RUN, read, record, verify, write
from kernel import gated_value
from sparsity_research.pythia import load_checkpoint_pythia
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb


def difference(expected, actual):
    a, b = expected.float(), actual.float()
    delta = b - a
    return {
        'equal': bool(torch.equal(expected, actual)),
        'changed': int((expected != actual).sum()),
        'elements': expected.numel(),
        'max_abs': float(delta.abs().max()),
        'relative_l2': float(torch.linalg.vector_norm(delta)
                             / torch.linalg.vector_norm(a).clamp_min(1e-12)),
    }


def capture(layer, values):
    handles = []
    for name, module in [('a', layer.input_layernorm),
                         ('m', layer.post_attention_layernorm),
                         ('qkv', layer.attention.query_key_value),
                         ('h_raw', layer.mlp.dense_h_to_4h),
                         ('q', layer.attention.q_post_site),
                         ('k', layer.attention.k_post_site),
                         ('v', layer.attention.v_site)]:
        def hook(_module, _args, output, name=name):
            values[name] = output.detach().clone()
        handles.append(module.register_forward_hook(hook))
    def before_z(_module, args):
        values['z_raw'] = args[0].detach().clone()
    handles.append(layer.attention.z_gate.register_forward_pre_hook(before_z))
    return handles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    dest = RUN / f'prelaunch/trace-{args.tag}.json'
    if not args.tag.isalnum() or dest.exists():
        parser.error('Fresh simple tag required')
    started = time.monotonic()
    topology.register()
    cfg, inputs = read(RUN / 'config.json'), read(RUN / 'provenance/inputs.json')
    torch.manual_seed(cfg['runtime_seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    validation = np.memmap(verify(inputs['validation']), dtype=np.int32, mode='r')
    source = inputs['checkpoints'][0]
    models = []
    for mode in ['native', 'generic_sparse']:
        model = load_checkpoint_pythia(transformers.AutoModelForCausalLM,
                                      RUN / source['checkpoint'], torch=torch)
        model = model.to('cuda', dtype=torch.bfloat16).eval()
        model.set_attn_implementation('sdpa')
        model.config.use_cache = False
        if mode != 'native':
            with torch.inference_mode():
                replay.install(model, mode)
        models.append(model)
    native, candidate = models
    rows = []
    with torch.inference_mode():
        ids = torch.tensor(validation[:2048].copy(), device='cuda', dtype=torch.long)[None]
        hidden = native.gpt_neox.embed_in(ids)
        positions = torch.arange(2048, device='cuda')[None]
        rotary = native.gpt_neox.rotary_emb(hidden, position_ids=positions)
        for index, (nl, cl) in enumerate(zip(native.gpt_neox.layers, candidate.gpt_neox.layers)):
            n, c = {}, {}
            handles = capture(nl, n) + capture(cl, c)
            expected = nl(hidden, attention_mask=None, position_embeddings=rotary)
            actual = cl(hidden, attention_mask=None, position_embeddings=rotary)
            for handle in handles:
                handle.remove()
            propagated = {name: difference(n[name], c[name]) for name in n}
            propagated['layer_output'] = difference(expected, actual)
            # Each comparison below supplies identical operands to both paths.
            pair = cl._run028_norm_pair
            a = pair.start(hidden).clone()
            m = pair.finish(hidden).clone()
            isolated = {'a_norm': difference(n['a'], a), 'm_norm': difference(n['m'], m)}
            for label, linear, operand in [
                ('qkv_projection', cl.attention.query_key_value, n['a']),
                ('w1_projection', cl.mlp.dense_h_to_4h, n['m']),
            ]:
                isolated[label] = difference(F.linear(operand, linear.weight, linear.bias), linear(operand))
            qkv = n['qkv'].view(1, 2048, 4, 96).transpose(1, 2)
            q, k, v = qkv.chunk(3, dim=-1)
            q, k = apply_rotary_pos_emb(q, k, *rotary)
            cq, ck, cv = cl.attention._run026_rope(n['qkv'], *rotary)
            for label, left, right in [('q_rope', q, cq), ('k_rope', k, ck), ('v_layout', v, cv)]:
                isolated[label] = difference(left, right)
            z = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=cl.attention.scaling)
            cz = cl.attention._run028_attention(q, k, v, cl.attention.scaling)
            isolated['attention'] = difference(z, cz)
            joint = cl._run026_joint
            h = gated_value(n['h_raw'], joint.gh, joint.th)
            z = gated_value(n['z_raw'], joint.gz, joint.tz)
            combined = (F.linear(h, joint.w2.weight, joint.w2.bias)
                        + F.linear(z, joint.wo.weight, joint.wo.bias)) + hidden
            isolated['joint_projection_residual'] = difference(
                combined, joint(n['h_raw'], n['z_raw'], hidden))
            row = {'layer': index, 'propagated': propagated, 'isolated': isolated}
            rows.append(row)
            print({'layer': index, 'isolated': {k: v for k, v in isolated.items() if not v['equal']}}, flush=True)
            hidden = expected
    write(dest, {'checkpoint': source['id'], 'gpu': torch.cuda.get_device_name(),
                 'seconds': time.monotonic() - started, 'rows': rows,
                 'script': record(RUN / '12_trace_numerics.py'),
                 'scope': 'one real block; every layer receives native hidden input; untimed development diagnostic'})


if __name__ == '__main__':
    main()
