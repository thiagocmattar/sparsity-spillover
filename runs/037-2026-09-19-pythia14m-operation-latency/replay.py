"""Install the frozen K050 composition, then change only execution-path switches."""
import frozen_replay as frozen
from controls import mask
from io_utils import RUN, module

dense, scaffold, R28 = frozen.dense, frozen.scaffold, frozen.R28


def final_row(name, catalog):
    original = next(r for r in catalog if r['id'] == 'k050')
    return {**original, 'id': name, 'operation_mask': mask(name)}


def install(model, row):
    metadata = frozen.install(model, row)
    if row['id'] == 'frozen':
        return {**metadata, 'control': 'untouched frozen K050', 'operation_mask': mask('frozen')}
    switches = row['operation_mask']
    control = module('run037_operation_controls', RUN / 'site_controls.py')
    for layer in model.gpt_neox.layers:
        layer.attention.query_key_value._run028_projection.skip = switches['a']
        layer.mlp.dense_h_to_4h._run028_projection.skip = switches['m']
        layer._run026_joint = control.Joint(layer._run026_joint, switches['h'], switches['z'])
        layer.attention._run028_attention = control.Attention(switches['qk'], switches['pv'])
    return {**metadata, 'control': 'independent sparse-path switches, gates unchanged',
            'operation_mask': switches, 'prefix_enabled': False}
