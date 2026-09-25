"""Keep every K050 path except the selected h/z mechanism controls frozen."""
import frozen_replay as frozen
from controls import mask
from io_utils import RUN, module

dense, scaffold, R28 = frozen.dense, frozen.scaffold, frozen.R28


def final_row(name, catalog):
    original = next(r for r in catalog if r['id'] == 'k050')
    return {**original, 'id': name, 'mechanism_mask': mask(name)}


def install(model, row):
    metadata = frozen.install(model, row)
    switches = row['mechanism_mask']
    if row['id'] != 'frozen':
        control = module('run056_mechanism_controls', RUN / 'site_controls.py')
        for layer in model.gpt_neox.layers:
            layer._run026_joint = control.Joint(layer._run026_joint, **switches)
    return {**metadata, 'control': row['id'], 'mechanism_mask': switches,
            'scope': 'h/z jointly in six layers; all other K050 paths unchanged'}
