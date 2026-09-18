"""Evaluation-only clipping around byte-identical final kernels."""
import math

SITES = ('a', 'm', 'h', 'z')


def validate_thresholds(thresholds, layers):
    expected = {f'{site}.layer_{i}' for site in SITES for i in range(layers)}
    if set(thresholds) != expected:
        raise ValueError('Thresholds must cover all four sites in every layer')
    if any(not math.isfinite(float(t)) or float(t) < 0 for t in thresholds.values()):
        raise ValueError('Finite nonnegative cutoffs required')


def clip(value, threshold, *, deferred_relu=False):
    # A Python scalar is compared in the activation dtype, as in Run030.
    mask = value <= threshold if deferred_relu else value.abs() <= threshold
    return value.masked_fill(mask, 0.)


def install(model, thresholds, *, candidate=False):
    from sparsity_research.pythia import expose_attention_sites
    import torch
    layers = model.gpt_neox.layers
    validate_thresholds(thresholds, len(layers))
    expose_attention_sites(model, torch=torch)
    handles = []
    # p=0 is value-identical; preserve original final-kernel control execution.
    if all(float(t) == 0 for t in thresholds.values()):
        return handles
    for i, layer in enumerate(layers):
        modules = {'a': layer.input_layernorm, 'm': layer.post_attention_layernorm,
                   'h': layer.mlp.act, 'z': layer.attention.z_site}
        for site, module in modules.items():
            t = float(thresholds[f'{site}.layer_{i}'])
            deferred = candidate and site == 'h' and isinstance(layer.mlp.act, torch.nn.ReLU)
            # A zero cutoff needs no operator except when materializing deferred ReLU.
            if t == 0 and not deferred:
                continue
            def hook(_module, _args, output, threshold=t, relu=deferred):
                return clip(output, threshold, deferred_relu=relu)
            handles.append(module.register_forward_hook(hook))
    return handles
