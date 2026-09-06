import sys
from pathlib import Path
import pytest
import torch
import transformers
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
sys.path.insert(0,str(Path(__file__).parent))
import adapter
from sparsity_research.pythia import apply_activation_topology


def rope_reference(x,c,s,q,k,v,heads,dimension,tq,tk,tv):
    qr,kr,vr=x.view(x.shape[0],x.shape[1],heads,3*dimension).transpose(1,2).chunk(3,dim=-1)
    qr,kr=apply_rotary_pos_emb(qr,kr,c,s)
    for output,value,threshold in [(q,qr,tq),(k,kr,tk),(v,vr,tv)]:
        output.copy_(value.masked_fill(value.abs()<threshold,0))


def joint_reference(h,z,wh,wz,bh,bz,r,out,th,tz,gh,gz,skip):
    h=h.masked_fill(h<th,0) if gh else h
    z=z.masked_fill(z<tz,0) if gz else z
    a=(h.float()@wh.float()+bh.float()).bfloat16()
    b=(z.float()@wz.float()+bz.float()).bfloat16()
    out.copy_((a+b)+r)


@pytest.mark.parametrize('topology',['A0','A1-H','A4-Z','A7-Z-POST'])
@pytest.mark.parametrize('mode',['fusion_dense','sparse','no_skip'])
def test_compatible_topologies_preserve_model(topology,mode):
    torch.manual_seed(2701)
    cfg=transformers.GPTNeoXConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,vocab_size=97,max_position_embeddings=64,hidden_dropout=0,attention_dropout=0)
    cfg.topology_id=topology
    sites={'A0':[],'A1-H':['h'],'A4-Z':['a','m','h','z'],
           'A7-Z-POST':['a','m','h','z','q_post','k_post','v']}[topology]
    cfg.site_gates={s:{'operator':'relu' if topology=='A1-H' else ('one_sided_threshold' if s in {'a','m','h','z'} else 'symmetric_threshold'),
                       **({} if topology=='A1-H' else {'kappa':.1})} for s in sites} or None
    model=apply_activation_topology(transformers.GPTNeoXForCausalLM(cfg),torch=torch).bfloat16().eval()
    ids=[torch.randint(0,97,(1,16)) for _ in range(3)]
    with torch.inference_mode():
        expected=[model(x,use_cache=False).logits.clone() for x in ids]
        adapter.install(model,mode,rope_backend=rope_reference,joint_backend=joint_reference)
        actual=[model(x,use_cache=False).logits.clone() for x in ids]
    for a,b in zip(actual,expected): torch.testing.assert_close(a,b,rtol=.02,atol=.02)


def test_disabled_gate_preserves_negative_values():
    h=torch.tensor([[-1.,0.,.5]],dtype=torch.bfloat16)
    z=h.clone(); w=torch.ones(3,1,dtype=torch.bfloat16)
    bias=torch.zeros(1,dtype=torch.bfloat16); residual=torch.zeros(1,1,dtype=torch.bfloat16)
    output=torch.empty_like(residual)
    joint_reference(h,z,w,w,bias,bias,residual,output,0.,0.,False,False,True)
    assert output.item()==-1.
