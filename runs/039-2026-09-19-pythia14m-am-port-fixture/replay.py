"""Install unmodified K050; replace only the explicitly selected input projections."""
import frozen_replay as frozen
from controls import settings

dense, scaffold, R28 = frozen.dense, frozen.scaffold, frozen.R28


def final_row(name, catalog):
    original = next(r for r in catalog if r['id'] == 'k050')
    return {**original, 'id': name, 'port_settings': settings(name)}


def install(model, row):
    metadata = frozen.install(model, row)
    selected = row['port_settings']
    if row['id'] != 'frozen':
        from site_port import Projection
        for layer in model.gpt_neox.layers:
            for site, linear in [('a', layer.attention.query_key_value), ('m', layer.mlp.dense_h_to_4h)]:
                if selected[site]:
                    linear._run028_projection = Projection(linear, skip=selected['skip'])
    return {**metadata, 'control': row['id'], 'port_settings': selected,
            'port': 'M8/K16 load-avoiding hybrid; padded M16 fallback; gates unchanged'}
