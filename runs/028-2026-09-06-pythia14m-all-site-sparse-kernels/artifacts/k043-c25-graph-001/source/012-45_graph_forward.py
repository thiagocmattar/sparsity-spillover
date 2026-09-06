"""Explicit fixed uncached, unpadded B1/T2048 full-logit graph scaffold.

Transformers5.12.1 materializes a causal mask during graph capture, changing
SDPA dispatch. This scaffold preserves the eager unmasked is_causal path.
Every graph comparator uses it; an unmodified eager stock anchor is retained.
It does not cache input-dependent embeddings, activations, or logits.
"""
import torch


def forward(model,ids):
    if tuple(ids.shape)!=(1,2048) or model.training or torch.is_grad_enabled():
        raise ValueError('B1 T2048 uncached/unpadded inference only')
    body=model.gpt_neox
    positions=torch.arange(2048,device=ids.device).unsqueeze(0)
    hidden=body.emb_dropout(body.embed_in(ids))
    rotary=body.rotary_emb(hidden,position_ids=positions)
    for layer in body.layers:
        hidden=layer(hidden,attention_mask=None,position_ids=positions,
                     layer_past=None,use_cache=False,position_embeddings=rotary)
    return model.embed_out(body.final_layer_norm(hidden))
