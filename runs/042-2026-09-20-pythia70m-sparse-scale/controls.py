"""One-component replacements with the model's original gates and rounding."""
from types import MethodType
import torch

MODES = ("native", "full", "all-skips-off", "hz-skips-off", "native-hz",
         "native-am", "native-attention", "native-norm", "native-rope")


class NativeJoint:
    native = True

    def __init__(self, previous):
        for name in ("w2", "wo", "gh", "gz", "th", "tz"):
            setattr(self, name, getattr(previous, name))

    def __call__(self, h, z, residual):
        if self.gh:
            h = h.masked_fill(h < self.th, 0)
        if self.gz:
            z = z.masked_fill(z < self.tz, 0)
        # Preserve branch BF16 outputs and both separately rounded additions.
        return (torch.nn.functional.linear(h, self.w2.weight, self.w2.bias)
                + torch.nn.functional.linear(z, self.wo.weight, self.wo.bias)) + residual


class NativeAttention:
    native = True

    def __call__(self, q, k, v, scale):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)


class NativeRope:
    def __init__(self, attention):
        self.attention = attention

    def __call__(self, raw, cos, sin):
        from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
        from sparsity_research.pythia import _optional_gate
        attn = self.attention
        q, k, v = raw.view(*raw.shape[:-1], -1, 3 * attn.head_size).transpose(1, 2).chunk(3, dim=-1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return tuple(_optional_gate(attn, site, value)
                     for site, value in (("q_post", q), ("k_post", k), ("v", v)))


def original_norms(model):
    return [(layer.input_layernorm.forward, layer.post_attention_layernorm.forward,
             {name: getattr(layer, name).forward for name in ("a_gate", "m_gate")
              if hasattr(layer, name)}) for layer in model.gpt_neox.layers]


def substitute(model, mode, saved_norms):
    if mode not in MODES:
        raise ValueError(mode)
    for layer, (first, second, gates) in zip(model.gpt_neox.layers, saved_norms):
        if mode == "hz-skips-off":
            layer._run026_joint.skip = False
        elif mode == "native-hz":
            layer._run026_joint = NativeJoint(layer._run026_joint)
        elif mode == "native-am":
            for linear in (layer.attention.query_key_value, layer.mlp.dense_h_to_4h):
                linear.forward = MethodType(torch.nn.Linear.forward, linear)
                linear._run042_native = True
        elif mode == "native-attention":
            layer.attention._run028_attention = NativeAttention()
        elif mode == "native-norm":
            layer.input_layernorm.forward, layer.post_attention_layernorm.forward = first, second
            for name, forward in gates.items():
                getattr(layer, name).forward = forward
        elif mode == "native-rope":
            layer.attention._run026_rope = NativeRope(layer.attention)
