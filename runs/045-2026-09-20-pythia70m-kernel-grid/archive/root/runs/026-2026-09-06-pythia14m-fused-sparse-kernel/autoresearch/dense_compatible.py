"""Canonical attention with its lazy import hoisted outside compiled forward.

Operation-for-operation copy of src/sparsity_research/pythia._attention_forward.
No backend, gate, dtype, mask, layout, or arithmetic is changed.
"""
from types import MethodType
from transformers.models.gpt_neox import modeling_gpt_neox
from sparsity_research.pythia import _optional_gate


def attention_forward(self, hidden_states, attention_mask, layer_past=None,
                      position_embeddings=None, **kwargs):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, 3 * self.head_size)
    qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
    query, key, value = qkv.chunk(3, dim=-1)
    query = _optional_gate(self, 'q_pre', query)
    key = _optional_gate(self, 'k_pre', key)
    query = self.q_pre_site(query)
    key = self.k_pre_site(key)
    cos, sin = position_embeddings
    query, key = modeling_gpt_neox.apply_rotary_pos_emb(query, key, cos, sin)
    query = self.q_post_site(_optional_gate(self, 'q_post', query))
    key = self.k_post_site(_optional_gate(self, 'k_post', key))
    value = self.v_site(_optional_gate(self, 'v', value))
    if layer_past is not None:
        key, value = layer_past.update(key, value, self.layer_idx)
    interface = modeling_gpt_neox.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, modeling_gpt_neox.eager_attention_forward)
    output, weights = interface(self, query, key, value, attention_mask,
        scaling=self.scaling, dropout=0.0 if not self.training else self.attention_dropout, **kwargs)
    output = output.reshape(*input_shape, -1).contiguous()
    output = self.z_site(_optional_gate(self, 'z', output))
    output = self.dense(output)
    return output, weights


def install(model):
    for layer in model.gpt_neox.layers:
        layer.attention.forward = MethodType(attention_forward, layer.attention)
