"""Explicit, nonoverlapping execution map; the frozen graph stays the anchor."""
import contextlib
import torch


class Tape:
    def __init__(self, events=False, annotations=False):
        self.events_enabled, self.annotations = events, annotations
        self.names, self.events = [], []

    def start(self):
        self.names, self.events = [], []
        self.mark()

    def mark(self):
        if self.events_enabled:
            event = torch.cuda.Event(enable_timing=True, external=True)
            event.record()
            self.events.append(event)

    def step(self, name, fn, *args, **kwargs):
        context = torch.profiler.record_function('stage::'+name) if self.annotations else contextlib.nullcontext()
        with context:
            value = fn(*args, **kwargs)
        self.names.append(name)
        self.mark()
        return value

    def times(self):
        return [a.elapsed_time(b)*1000 for a,b in zip(self.events,self.events[1:])]


def forward(model, ids, tape, optimized):
    from transformers.models.gpt_neox import modeling_gpt_neox as neox
    from sparsity_research.pythia import _optional_gate
    if tuple(ids.shape)!=(1,2048) or model.training or torch.is_grad_enabled():
        raise ValueError('Uncached B1/T2048 inference only')
    body=model.gpt_neox; tape.start(); s=tape.step
    positions=s('positions',lambda:torch.arange(2048,device=ids.device).unsqueeze(0))
    hidden=s('embedding',lambda:body.emb_dropout(body.embed_in(ids)))
    rotary=s('rotary_tables',lambda:body.rotary_emb(hidden,position_ids=positions))
    for index,layer in enumerate(body.layers):
        p=f'layer{index}.'; attn=layer.attention; shape=hidden.shape[:-1]
        a=s(p+('norm_pair' if optimized else 'norm_attention'),layer.input_layernorm,hidden)
        raw=s(p+'qkv',attn.query_key_value,a)
        if optimized:
            def rope():
                qr,kr,_=raw.view(*shape,-1,3*attn.head_size).transpose(1,2).chunk(3,dim=-1)
                attn.q_pre_site(qr);attn.k_pre_site(kr)
                q,k,v=attn._run026_rope(raw,*rotary)
                return attn.q_post_site(q),attn.k_post_site(k),attn.v_site(v)
            q,k,v=s(p+'rope_qkv_layout',rope)
            context=s(p+'attention',attn._run028_attention,q,k,v,attn.scaling)
            z=s(p+'context_pack',lambda:attn.z_site(_optional_gate(attn,'z',context.transpose(1,2).reshape(*shape,-1).contiguous())))
            m=layer.post_attention_layernorm(hidden)  # already computed by norm_pair; identity retrieval
            h=s(p+'mlp_up',layer.mlp.dense_h_to_4h,m)
            h=layer.mlp.act(h)  # installed identity; original gate is inside joint
            hidden=s(p+'joint_gate_down_residual',layer._run026_joint,h,z,hidden)
        else:
            def rope():
                q,k,v=raw.view(*shape,-1,3*attn.head_size).transpose(1,2).chunk(3,dim=-1)
                q=attn.q_pre_site(_optional_gate(attn,'q_pre',q));k=attn.k_pre_site(_optional_gate(attn,'k_pre',k))
                q,k=neox.apply_rotary_pos_emb(q,k,*rotary)
                return attn.q_post_site(_optional_gate(attn,'q_post',q)),attn.k_post_site(_optional_gate(attn,'k_post',k)),attn.v_site(_optional_gate(attn,'v',v))
            q,k,v=s(p+'rope_qkv_layout',rope)
            interface=neox.ALL_ATTENTION_FUNCTIONS.get_interface(attn.config._attn_implementation,neox.eager_attention_forward)
            context,_=s(p+'attention',interface,attn,q,k,v,None,scaling=attn.scaling,dropout=0.,position_ids=positions,use_cache=False)
            z=s(p+'context_pack',lambda:context.reshape(*shape,-1).contiguous())
            z=s(p+'z_gate',lambda:attn.z_site(_optional_gate(attn,'z',z)))
            zo=s(p+'attention_out',lambda:layer.post_attention_dropout(attn.dense(z)))
            m=s(p+'norm_mlp',layer.post_attention_layernorm,hidden)
            h=s(p+'mlp_up',layer.mlp.dense_h_to_4h,m)
            h=s(p+'h_gate',layer.mlp.act,h)
            ho=s(p+'mlp_down',lambda:layer.post_mlp_dropout(layer.mlp.dense_4h_to_h(h)))
            hidden=s(p+'residual',lambda:ho+zo+hidden)
    hidden=s('final_norm',body.final_layer_norm,hidden)
    return s('lm_head',model.embed_out,hidden)
