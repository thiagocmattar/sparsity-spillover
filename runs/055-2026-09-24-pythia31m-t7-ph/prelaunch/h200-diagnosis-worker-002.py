
import pathlib,sys,json,os,traceback
import numpy as np
import torch,transformers
r=pathlib.Path('/workspace/sparsity-spillover/runs/055-2026-09-24-pythia31m-t7-ph')
c=pathlib.Path('/workspace/run055-control')
sys.path.insert(0,str(r/'latency'))
import port
from sparsity_research.pythia import load_checkpoint_pythia
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
row,=[v for p in (r/'artifacts').glob('verification-*.json') for v in json.loads(p.read_text()).get('conditions',[]) if v['condition']['id']=='a7-h-ol1-kappa-0' and v['status']=='verified']
checkpoint=r/'artifacts/attempts'/row['attempt']/row['final_checkpoint']['path']
data=np.memmap(r.parents[1]/'data/tokenized/minipile-pythia-14m-full/validation/tokens.int32.bin',dtype=np.int32,mode='r')
def load():
 m=load_checkpoint_pythia(transformers.AutoModelForCausalLM,checkpoint,torch=torch).to('cuda',dtype=torch.bfloat16).eval()
 m.set_attn_implementation('sdpa');m.config.use_cache=False
 return m
def compare(a,b):
 a=a.float();b=b.float();d=(a-b).abs();bound=.25+.02*a.abs()
 return dict(max_abs=float(d.max()),relative_l2=float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(a)),bad_elements=int((d>bound).sum()))
out=dict(scientific_measurement=False,scope='component substitution on failed block15',rows=[])
try:
 with torch.inference_mode():
  ids=torch.tensor(data[15*2048:16*2048].copy(),device='cuda',dtype=torch.long)[None]
  native=load();reference=native(input_ids=ids,use_cache=False).logits.clone()
  del native
  for variant in ['native_flash','native_rope']:
   m=load();old=[]
   for layer in m.gpt_neox.layers:
    old.append(dict(norm1=layer.input_layernorm.forward,norm2=layer.post_attention_layernorm.forward,a=layer.a_gate.forward,m=layer.m_gate.forward,attention=layer.attention.forward,layer=layer.forward,h=layer.mlp.act.forward,z=layer.attention.z_gate.forward,w2=layer.mlp.dense_4h_to_h,wo=layer.attention.dense))
   head=m.embed_out.forward
   port.install(m)
   for layer,saved in zip(m.gpt_neox.layers,old):
    if variant=='native_norm':
     layer.input_layernorm.forward=saved['norm1'];layer.post_attention_layernorm.forward=saved['norm2'];layer.a_gate.forward=saved['a'];layer.m_gate.forward=saved['m']
    if variant=='native_attention':layer.attention.forward=saved['attention']
    if variant=='native_joint':
     layer.forward=saved['layer'];layer.mlp.act.forward=saved['h'];layer.attention.z_gate.forward=saved['z'];layer.mlp.dense_4h_to_h=saved['w2'];layer.attention.dense=saved['wo']
   if variant=='native_head':m.embed_out.forward=head
   if variant=='native_flash':
    from types import MethodType
    adapter=port.module('diag_adapter',port.SOURCE/'hz_adapter.py')
    for layer in m.gpt_neox.layers:layer.attention.forward=MethodType(adapter.rope.attention_forward,layer.attention)
   if variant=='native_rope':
    from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
    def rope(attn):
     def call(raw,cos,sin):
      q,k,v=raw.view(*raw.shape[:-1],-1,3*attn.head_size).transpose(1,2).chunk(3,dim=-1)
      q,k=apply_rotary_pos_emb(q,k,cos,sin)
      return attn.q_post_gate(q),attn.k_post_gate(k),attn.v_gate(v)
     return call
    for layer in m.gpt_neox.layers:layer.attention._run026_rope=rope(layer.attention)
   result=port.replay.scaffold.forward(m,ids).clone()
   row=dict(variant=variant,**compare(reference,result));out['rows'].append(row)
   print(json.dumps(row),flush=True)
   del m,result,old,head
   torch.cuda.empty_cache()
 out['status']='completed'
except BaseException as e:
 out.update(status='failed',error=str(e),traceback=traceback.format_exc());raise
finally:
 (r/'artifacts/preflight-h200-diagnosis-002.json').write_text(json.dumps(out,indent=2))
 (c/'h200-diagnosis-exit-002.json').write_text(json.dumps(out,indent=2))
