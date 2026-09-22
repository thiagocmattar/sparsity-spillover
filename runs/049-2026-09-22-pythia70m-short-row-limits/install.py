"""Fixed h/z policies; dense fallback uses gate configuration, never checkpoint ID."""
import replay
from controls import NativeJoint
from short_rows import Joint


def install(model, mode):
    if mode not in ('opt073', 'native_hz', 'limit16', 'limit32', 'limit64'):
        raise ValueError(mode)
    metadata = replay.install(model, 'opt073')
    policy = []
    for i, layer in enumerate(model.gpt_neox.layers):
        previous = layer._run026_joint
        if mode == 'native_hz' or (mode.startswith('limit') and not previous.gh and not previous.gz):
            layer._run026_joint = NativeJoint(previous)
            policy.append({'layer': i, 'h_z': 'native', 'reason': 'explicit dense control' if mode == 'native_hz' else 'both gates inactive'})
        elif mode.startswith('limit'):
            layer._run026_joint = Joint(previous, int(mode.removeprefix('limit')))
            policy.append({'layer': i, 'h_z': mode, 'matrix_fallback': 'unchanged M8/N256/K16'})
        else:
            policy.append({'layer': i, 'h_z': 'frozen opt073'})
    return {**metadata, 'run049_mode': mode, 'effective_hz_policy': policy}
