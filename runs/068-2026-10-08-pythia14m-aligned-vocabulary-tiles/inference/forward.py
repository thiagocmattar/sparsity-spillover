"""Fixed uncached/unpadded map retaining the native eager causal SDPA path in capture.

Transformers 5.12's mask builder changes path during CUDA capture: it materializes
a causal mask instead of using is_causal=True, changing SDPA rounding. For this
fixed workload the native eager mask is None. Call the unchanged decoder layers
with that same mask in eager and capture. The original model remains the anchor.
"""
import torch


def forward(model, ids):
    if tuple(ids.shape)!=(1,2048) or model.training or torch.is_grad_enabled() or model.config.use_cache:
        raise ValueError('Uncached/unpadded B1/T2048 inference only')
    body=model.gpt_neox
    hidden=body.emb_dropout(body.embed_in(ids))
    positions=torch.arange(2048,device=ids.device).unsqueeze(0)
    rotary=body.rotary_emb(hidden,position_ids=positions)
    for layer in body.layers:
        hidden=layer(hidden,attention_mask=None,position_ids=positions,layer_past=None,
                     use_cache=False,position_embeddings=rotary)
    hidden=body.final_layer_norm(hidden)
    return model.embed_out(hidden)
