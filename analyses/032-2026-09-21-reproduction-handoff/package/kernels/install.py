"""Assemble the final 14M or 70M implementation directly from its components."""

from types import MethodType, SimpleNamespace
import torch
from support import ROOT, module
from sparsity_research.pythia import expose_attention_sites, topology_metadata
from sparsity_research.sites import FixedOneSidedThreshold


def install(model, backend="optimized", control=None, operation_mode=None):
    if backend == "native":
        return {"implementation": "native SDPA"}
    if backend != "optimized":
        raise ValueError(
            "This release contains only the final specialized implementation"
        )
    c = model.config
    if (
        c.hidden_size,
        c.intermediate_size,
        c.num_attention_heads,
        c.num_hidden_layers,
    ) not in {(128, 512, 4, 6), (512, 2048, 8, 6)} or not c.use_parallel_residual:
        raise ValueError("Fixed 14M/70M parallel-residual architectures required")
    small = c.hidden_size == 128
    if operation_mode and (not small or control):
        raise ValueError("Operation ablations require 14M and no additional control")
    if control not in {None, "native-hz", "hz-skips-off"}:
        raise ValueError("Unknown control")
    expose_attention_sites(model, torch=torch)
    folder = ROOT / ("model_14m" if small else "model_70m")
    norm = module("normalization", folder / "normalization.py")
    joint = module("output_projection", folder / "output_projection.py")
    attention = module("attention_kernel", folder / "attention/implementation.py")
    rotary = module("rotary_kernel", ROOT / "rotary/implementation.py")
    block = module("block_forward", ROOT / "block_forward.py")
    attn = module("attention_forward", ROOT / "attention_forward.py")
    projection = (
        module("input_projection", folder / "input_projection.py") if small else None
    )
    pairs = [
        norm.NormPair(
            layer.input_layernorm,
            layer.post_attention_layernorm,
            getattr(layer, "a_gate", None),
            getattr(layer, "m_gate", None),
        )
        for layer in model.gpt_neox.layers
    ]
    for layer, pair in zip(model.gpt_neox.layers, pairs):
        ports = SimpleNamespace(
            w2=layer.mlp.dense_4h_to_h,
            wo=layer.attention.dense,
            gh=isinstance(layer.mlp.act, (FixedOneSidedThreshold, torch.nn.ReLU)),
            gz=isinstance(
                getattr(layer.attention, "z_gate", None), FixedOneSidedThreshold
            ),
            th=getattr(layer.mlp.act, "kappa", 0.0),
            tz=getattr(getattr(layer.attention, "z_gate", None), "kappa", 0.0),
            skip=True,
        )
        layer._sparsity_joint = joint.Joint(ports)
        layer.attention._sparsity_rope = rotary.RopeGate(layer.attention)
        layer.attention._sparsity_attention = attention.Attention(
            skip=small, shortcut=False
        )
        layer.attention.forward = MethodType(attn.attention_forward, layer.attention)
        layer.mlp.dense_4h_to_h = torch.nn.Identity()
        layer.attention.dense = torch.nn.Identity()
        if ports.gh:
            layer.mlp.act.forward = MethodType(block.identity, layer.mlp.act)
        if ports.gz:
            layer.attention.z_gate.forward = MethodType(
                block.identity, layer.attention.z_gate
            )
        layer.forward = MethodType(block.layer_forward, layer)
        if small:
            for linear in (layer.attention.query_key_value, layer.mlp.dense_h_to_4h):
                linear._sparsity_projection = projection.Projection(linear, skip=True)
                linear.forward = MethodType(
                    lambda obj, x: obj._sparsity_projection(x), linear
                )
        norm.bind_pair(layer, pair)
    if not small:
        head = module("vocabulary_projection", folder / "vocabulary.py")
        model.embed_out._sparsity_head = head.Head(model.embed_out, 0)
        model.embed_out.forward = MethodType(
            lambda obj, x: obj._sparsity_head(x), model.embed_out
        )
    metadata = dict(
        implementation="specialized-14m" if small else "specialized-70m",
        topology=topology_metadata(model),
        sparse_sites=["a", "m", "h", "z", "qk", "pv"] if small else ["h", "z"],
    )
    if operation_mode:
        modes = module("operation_modes", ROOT / "ablation/modes.py")
        mask = modes.mask(operation_mode)
        if operation_mode != "frozen":
            switches = module("execution_switches", ROOT / "ablation/implementation.py")
            for layer in model.gpt_neox.layers:
                layer.attention.query_key_value._sparsity_projection.skip = mask["a"]
                layer.mlp.dense_h_to_4h._sparsity_projection.skip = mask["m"]
                layer._sparsity_joint = switches.Joint(
                    layer._sparsity_joint, mask["h"], mask["z"]
                )
                layer.attention._sparsity_attention = switches.Attention(
                    mask["qk"], mask["pv"]
                )
        metadata.update(operation_mask=mask, operation_mode=operation_mode)
    if control == "native-hz":
        controls = module("projection_control", ROOT / "controls.py")
        for layer in model.gpt_neox.layers:
            layer._sparsity_joint = controls.NativeJoint(layer._sparsity_joint)
    elif control == "hz-skips-off":
        for layer in model.gpt_neox.layers:
            layer._sparsity_joint.skip = False
    return metadata
