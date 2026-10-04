"""Run060's causal corner gate; no change to historical scalar operators."""
from collections.abc import Mapping
import math
import torch

OPERATOR = 'tile_corner_threshold'


class CornerTileThreshold(torch.nn.Module):
    def __init__(self, kappa, tile_tokens=8, tile_features=16):
        super().__init__()
        if isinstance(kappa, bool) or not isinstance(kappa, (int, float)) or not math.isfinite(kappa) or kappa < 0:
            raise ValueError('kappa must be finite and nonnegative')
        for value in (tile_tokens, tile_features):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError('Tile dimensions must be positive integers')
        self.kappa = float(kappa)
        self.tile_tokens, self.tile_features = tile_tokens, tile_features

    def retained_mask(self, value):
        if value.ndim != 3 or not value.is_floating_point() or any(d <= 0 for d in value.shape):
            raise ValueError('Gate requires floating [batch,token,feature]')
        b, t, d = value.shape
        if d % self.tile_features:
            raise ValueError('Feature width must be divisible by tile_features')
        # Partial final token blocks are supported so prefix evaluation is causal.
        # Scientific training/validation uses complete, aligned eight-token blocks.
        live = value.detach()[:, ::self.tile_tokens, ::self.tile_features] > self.kappa
        rows = torch.arange(t, device=value.device) // self.tile_tokens
        return live.index_select(1, rows).unsqueeze(-1).expand(b, t, d // self.tile_features, self.tile_features).reshape_as(value)

    def forward(self, value):
        return value.masked_fill(~self.retained_mask(value), 0.0)

    def metadata(self):
        return {'operator': OPERATOR, 'kappa': self.kappa,
                'tile_tokens': self.tile_tokens, 'tile_features': self.tile_features}


def register():
    """Narrow runtime adapter for the reused model/checkpoint helpers."""
    from sparsity_research import sites, pythia
    if getattr(sites, '_run060_corner_gate_registered', False):
        return
    old_resolve, old_build, old_metadata = sites.resolve_topology_and_gates, sites.build_gate, sites.gate_metadata
    old_apply = pythia.apply_activation_topology

    def resolve(topology_id, site_gate, site_gates=None):
        if not isinstance(site_gates, Mapping) or not any(isinstance(s, Mapping) and s.get('operator') == OPERATOR for s in site_gates.values()):
            return old_resolve(topology_id, site_gate, site_gates)
        topology = sites.resolve_topology(topology_id)
        if topology.active_sites != ('h', 'z') or site_gate is not None or set(site_gates) != {'h', 'z'}:
            raise ValueError('Run060 corner gate requires exactly HZ and per-site gates')
        resolved = {}
        for name, spec in site_gates.items():
            if not isinstance(spec, Mapping) or set(spec) != {'operator', 'kappa', 'tile_tokens', 'tile_features'} or spec['operator'] != OPERATOR:
                raise ValueError('Corner gate requires explicit operator, kappa and both dimensions at h/z')
            resolved[name] = CornerTileThreshold(spec['kappa'], spec['tile_tokens'], spec['tile_features']).metadata()
        return topology, resolved

    def build(spec, *, torch_module=torch):
        if spec.get('operator') == OPERATOR:
            return CornerTileThreshold(spec['kappa'], spec['tile_tokens'], spec['tile_features'])
        return old_build(spec, torch_module=torch_module)

    def metadata(module):
        return module.metadata() if isinstance(module, CornerTileThreshold) else old_metadata(module)

    def apply(model, *, torch):
        result = old_apply(model, torch=torch)
        if any(s.get('operator') == OPERATOR for s in (getattr(model.config, 'site_gates', None) or {}).values()):
            model.config.use_cache = False
            if not getattr(model, '_run060_cache_guard', False):
                def guard(_module, args, kwargs):
                    if kwargs.get('use_cache', False) or kwargs.get('past_key_values') is not None:
                        raise ValueError('Run060 supports uncached sequence-local tiles; cached decoding needs persistent tile state')
                model.register_forward_pre_hook(guard, with_kwargs=True)
                model._run060_cache_guard = True
        return result

    sites.resolve_topology_and_gates = pythia.resolve_topology_and_gates = resolve
    sites.build_gate = pythia.build_gate = build
    sites.gate_metadata = pythia.gate_metadata = metadata
    pythia.apply_activation_topology = apply
    sites._run060_corner_gate_registered = True
