"""Run-local normalized LM-head input gate, capture and pressure contract."""
from dataclasses import replace

import torch
from sparsity_research import pythia
from sparsity_research.capture import ActivationCapture as BlockCapture, CapturedSite
from sparsity_research.pressure import parse_pressure_config as parse_block_pressure
from tile_gate import CornerTileThreshold

HEAD_NAME = 'head_in.final'


def install_head(model, spec):
    if spec is not None:
        if set(spec) != {'operator', 'kappa', 'tile_tokens', 'tile_features'} or spec['operator'] != 'tile_corner_threshold':
            raise ValueError('Explicit corner head-gate specification required')
        gate = CornerTileThreshold(**{k:v for k,v in spec.items() if k != 'operator'})
        if model.config.hidden_size % gate.tile_features:
            raise ValueError('Hidden width must contain complete head feature tiles')
    else:
        gate = torch.nn.Identity()
    previous = getattr(model, '_run066_head_hook', None)
    if previous is not None:
        previous.remove()
    model.head_input_gate = gate
    model.config.head_input_gate = None if spec is None else dict(spec)
    # Applied to the learned affine LN output, immediately before embed_out.
    model._run066_head_hook = model.gpt_neox.final_layer_norm.register_forward_hook(
        lambda _module, _inputs, value: model.head_input_gate(value))
    return model


def build_random_pythia(model_config, **kwargs):
    model = pythia.build_random_pythia(model_config, **kwargs)
    return install_head(model, model_config.get('head_input_gate'))


def load_checkpoint_pythia(auto_model, checkpoint, *, torch):
    model = pythia.load_checkpoint_pythia(auto_model, checkpoint, torch=torch)
    return install_head(model, getattr(model.config, 'head_input_gate', None))


def topology_metadata(model):
    spec = getattr(model.config, 'head_input_gate', None)
    gate = getattr(model, 'head_input_gate', None)
    realized = gate.metadata() if isinstance(gate, CornerTileThreshold) else None
    if realized != spec or (spec is None and not isinstance(gate, torch.nn.Identity)):
        raise ValueError('Realized head gate differs from checkpoint configuration')
    handle = getattr(model, '_run066_head_hook', None)
    if handle is None or handle.id not in model.gpt_neox.final_layer_norm._forward_hooks:
        raise ValueError('Head gate must be installed after final LayerNorm')
    return {**pythia.topology_metadata(model),
            'head_input_gate': spec,
            'head_input_port': 'final_layer_norm output before embed_out'}


def parse_pressure_config(raw):
    sites = tuple(raw.get('sites', ()))
    if 'head_in' not in sites:
        return parse_block_pressure(raw)
    if sites.count('head_in') != 1:
        raise ValueError('Duplicate head pressure site')
    # Validate all normal method/range/site invariants using the existing parser.
    ordinary = tuple(site for site in sites if site != 'head_in')
    parsed = parse_block_pressure({**raw, 'sites': ordinary or ('h',)})
    return replace(parsed, sites=sites)


class ActivationCapture:
    """Standard block capture plus exactly one post-gate head tensor."""
    def __init__(self, model, sites, *, torch):
        if len(sites) != len(set(sites)) or not sites:
            raise ValueError('Nonempty unique capture sites required')
        self.model, self.sites = model, tuple(sites)
        ordinary = [site for site in sites if site != 'head_in']
        self.block = BlockCapture(model, ordinary, torch=torch) if ordinary else None
        self.activations = self.block.activations if self.block else {}
        self.metadata = self.block.metadata if self.block else []
        self.handle = None

    def __enter__(self):
        if self.block:
            self.block.__enter__()
        if 'head_in' in self.sites:
            self.metadata.append(CapturedSite('head_in', HEAD_NAME, 'head_input_gate',
                'post-gate normalized final hidden state', '[B,T,D]', 'embed_out'))
            def hook(_module, _inputs, output):
                self.activations[HEAD_NAME] = output
            self.handle = self.model.head_input_gate.register_forward_hook(hook)
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        if self.block:
            self.block.__exit__(*args)

    def clear(self):
        self.activations.clear()


def pressure_capture_names(model, sites):
    return tuple(sorted(name for site in sites for name in (
        [HEAD_NAME] if site == 'head_in' else
        [f'{site}.layer_{i}' for i in range(len(model.gpt_neox.layers))])))
