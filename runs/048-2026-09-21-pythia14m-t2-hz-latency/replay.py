"""Install frozen Run041 K050, replacing only the independent h/z skip switches."""
import frozen_replay as frozen
from controls import MASKS
from io_utils import RUN, module

dense, scaffold, R28 = frozen.dense, frozen.scaffold, frozen.R28


def install(model, mode, catalog):
    row = next(r for r in catalog if r['id'] == 'k050')
    assert row['status'] == 'eligible' and row['settings']['shortcut'] is False
    metadata = frozen.install(model, row)
    before = []
    control = module('run048_hz_controls', RUN/'site_controls.py') if mode != 'frozen' else None
    for layer in model.gpt_neox.layers:
        joint = layer._run026_joint
        before.append({'h_gate': joint.gh, 'z_gate': joint.gz,
                       'h_threshold': joint.th, 'z_threshold': joint.tz})
        assert joint.gh and joint.gz and joint.th == joint.tz == .1
        assert joint.skip
        if control is not None:
            layer._run026_joint = control.Joint(joint, *MASKS[mode])
    return {'frozen_installation': metadata, 'mode': mode, 'gates': before,
            'skip_h_z': list(MASKS.get(mode, (True, True))),
            'only_change': 'Run037 compile-time h/z zero-exploitation controls'}
