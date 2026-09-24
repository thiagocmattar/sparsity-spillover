"""31M shape specialization of Run045 opt073, with its fixed component policy."""
from pathlib import Path
import sys
from types import MethodType

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]/"045-2026-09-20-pythia70m-kernel-grid"
sys.path.insert(0, str(SOURCE))
import replay
from io_utils import module
# Resolve the D256 counting oracles locally even in a complete repository checkout.
sys.path.insert(0, str(HERE))


def install(model):
    c = model.config
    if (c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_hidden_layers) != (256,1024,8,6):
        raise ValueError("Run054 requires the pinned 31M architecture")
    norm = module("run054_norm", HERE/"kernel/norm/candidate.py")
    joint = module("run054_joint", HERE/"kernel/joint/joint.py")
    head = module("run054_head", HERE/"kernel/head/head.py")
    attention = module("run054_attention", HERE/"kernel/attention/candidate.py")
    pairs = [norm.NormPair(layer.input_layernorm, layer.post_attention_layernorm)
             for layer in model.gpt_neox.layers]
    adapter = module("run054_frozen_adapter", SOURCE/"hz_adapter.py")
    adapter.install(model, "sparse")
    forward = module("run054_attention_forward", replay.R28/"candidates/k020/candidate.py").attention_forward
    for layer, pair in zip(model.gpt_neox.layers, pairs):
        layer._run026_joint = joint.Joint(layer._run026_joint)
        layer.attention._run028_attention = attention.Attention(skip=False, shortcut=False)
        layer.attention.forward = MethodType(forward, layer.attention)
        norm.bind_pair(layer, pair)
    linear = model.embed_out
    linear._run054_head = head.Head(linear, 0)
    linear.forward = MethodType(lambda obj, x: obj._run054_head(x), linear)
    return dict(identity="opt073-31m-v1", parent="Run045 opt073", dimensions=dict(d=256,ffn=1024,heads=8,head_dim=32),
        input_projections="native dense, as opt073", normalization="shared-input width256 paired LayerNorm",
        output_projections="M8 N256 K16, <=8-entry scalar rows, parallel inspection and native-order MMA fallback",
        attention="dense Flash M64 N128, four warps, head32, token-major output", head="CUTLASS 128x128x32 full50304",
        new_optimization_search=False, same_policy_all_checkpoints=True)
