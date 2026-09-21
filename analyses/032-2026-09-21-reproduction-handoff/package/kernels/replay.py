"""Final implementations only; historical names identify exact retained sources."""

import sys
from support import ROOT, module

R28 = ROOT / "14m"


def install(model, backend="optimized", control=None, operation_mode=None):
    if backend == "native":
        return {"implementation": "native SDPA"}
    if model.config.hidden_size == 128:
        adapter = module("lean_adapter", ROOT / "base/adapter.py")
        adapter.install(model, "sparse")
        m = module("lean_k050", R28 / "candidates/k050/candidate.py")
        metadata = m.install(
            model, shortcut=False, round_p=False, skip=True, projection_skip=True
        )
    elif model.config.hidden_size == 512:
        sys.path.insert(0, str(ROOT / "70m"))
        name = (
            "kernel/candidate.py"
            if backend == "port"
            else "candidates/opt073/candidate.py"
        )
        m = module("lean_70m", ROOT / "70m" / name)
        metadata = m.install(model)
    else:
        raise ValueError("Final specialized kernels cover 14M and 70M only")
    if operation_mode:
        if model.config.hidden_size != 128 or control:
            raise ValueError(
                "Independent operation modes require 14M and no other control"
            )
        modes = module("lean_operation_modes", ROOT / "ablation14m/modes.py")
        mask = modes.mask(operation_mode)
        if operation_mode != "frozen":
            switches = module(
                "lean_site_controls", ROOT / "ablation14m/site_controls.py"
            )
            for layer in model.gpt_neox.layers:
                layer.attention.query_key_value._run028_projection.skip = mask["a"]
                layer.mlp.dense_h_to_4h._run028_projection.skip = mask["m"]
                layer._run026_joint = switches.Joint(
                    layer._run026_joint, mask["h"], mask["z"]
                )
                layer.attention._run028_attention = switches.Attention(
                    mask["qk"], mask["pv"]
                )
        metadata = {
            **metadata,
            "operation_mask": mask,
            "operation_mode": operation_mode,
        }
    if control == "native-hz":
        controls = module("lean_controls", ROOT / "70m/controls.py")
        for layer in model.gpt_neox.layers:
            layer._run026_joint = controls.NativeJoint(layer._run026_joint)
    elif control == "hz-skips-off":
        for layer in model.gpt_neox.layers:
            layer._run026_joint.skip = False
    elif control is not None:
        raise ValueError("Unknown control")
    return metadata
