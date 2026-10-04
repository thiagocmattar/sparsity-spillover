"""Run-local T7 corner comparison and reused full-pass data primitives."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping
import yaml
from sparsity_research.pressure import parse_pressure_config
from frozen import load_run004_module
from topology import register
from tile_gate import SITES, COMPARISONS
register()
from sparsity_research.sites import resolve_topology_and_gates

RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
DEFAULT_CONFIG = RUN_DIR / 'config.yaml'
EXPECTED_INITIAL_PARAMETER_SHA256 = 'ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57'
EXPECTED_SCHEDULE_SHA256 = 'f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa'
_BASE = load_run004_module('_run062_config_base', 'run_config.py')
_BASE.RUN_DIR, _BASE.REPO_ROOT = RUN_DIR, REPO_ROOT
mapping = _BASE.mapping
repo_path = _BASE.repo_path
load_verified_caches = _BASE.load_verified_caches
build_schedule = _BASE.build_schedule
microbatches_for_step = _BASE.microbatches_for_step
require_cuda = _BASE.require_cuda
seed_everything = _BASE.seed_everything
require_training_cache = _BASE.require_training_cache
require_validation_coverage = _BASE.require_validation_coverage
parameter_sha256 = _BASE.parameter_sha256
inventory_content_sha256 = _BASE.inventory_content_sha256
cache_identity = _BASE.cache_identity
git_identity = _BASE.git_identity
write_json = _BASE.write_json

def load_config(path=DEFAULT_CONFIG):
    value = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    validate_config(value)
    return value

def condition_specs(config):
    rows = []
    c = mapping(config, 'conditions')
    for recipe, sites in c['pressure_recipes'].items():
        for threshold in c['gate_thresholds']:
            token = f'{float(threshold):g}'.replace('.', 'p')
            rows.append({'id':f't7-tile-{recipe}-kappa-{token}',
                'order':len(rows)+1, 'topology_id':'A7-Z-POST',
                'active_sites':list(SITES), 'gate_operator':c['gate_operator'],
                'corner_comparisons':dict(c['corner_comparisons']),
                'tile_tokens':c['tile_tokens'], 'tile_features':c['tile_features'],
                'gate_threshold':float(threshold), 'pressure_recipe':recipe,
                'pressure_method':c['pressure_method'], 'pressure_sites':list(sites),
                'pressure_weight':float(c['pressure_weight']),
                'step_budget':float(c['step_budget']), 'is_control':False})
    return rows

def resolved_condition_config(config, condition):
    resolved = deepcopy(dict(config))
    resolved['condition'] = deepcopy(dict(condition))
    resolved['model'].update(topology_id='A7-Z-POST', site_gate=None, site_gates={
        site:{'operator':condition['gate_operator'],
              'kappa':float(condition['gate_threshold']),
              'tile_tokens':condition['tile_tokens'],
              'tile_features':condition['tile_features'],
              'comparison':condition['corner_comparisons'][site]} for site in SITES})
    resolve_topology_and_gates('A7-Z-POST', None, resolved['model']['site_gates'])
    resolved['activation_pressure'] = {'method':condition['pressure_method'],
        'sites':list(condition['pressure_sites']), 'weight':condition['pressure_weight'],
        'step_budget':condition['step_budget'], 'eps':1e-12}
    parse_pressure_config(resolved['activation_pressure'])
    return resolved

def worker_conditions(config, worker_id):
    assignments = mapping(config['runpod'], 'worker_assignments')
    by_id = {r['id']:r for r in condition_specs(config)}
    return [deepcopy(by_id[i]) for i in assignments[worker_id]]

def validate_config(config):
    if not isinstance(config, Mapping):
        raise ValueError('Config must be a mapping')
    m, c, t, v = (mapping(config, k) for k in ('model','conditions','training','validation'))
    if m.get('initialization') != 'random':
        raise ValueError('Random pretraining required')
    if not isinstance(m.get('revision'), str) or len(m['revision']) != 40:
        raise ValueError('Pinned architecture revision required')
    declared, _ = resolve_topology_and_gates(m.get('topology_id'), m.get('site_gate'), m.get('site_gates'))
    if declared.active_sites != SITES or c.get('corner_comparisons') != COMPARISONS:
        raise ValueError('T7 topology and approved mixed corner comparisons required')
    if any(isinstance(k,bool) or not isinstance(k,(int,float)) for k in c['gate_thresholds']):
        raise ValueError('Thresholds must be numbers, excluding booleans')
    for row in condition_specs(config):
        if row['gate_operator'] != 'tile_corner_threshold' or c['corner_rejection'] != 'less_than_or_equal':
            raise ValueError('Inclusive corner rejection required')
        if not math.isfinite(row['gate_threshold']) or row['gate_threshold'] < 0:
            raise ValueError('Finite nonnegative threshold required')
        if tuple(row['pressure_sites']) != ('h',):
            raise ValueError('Ph requires h-only pressure')
        if row['pressure_method'] != 'orthogonal_l1':
            raise ValueError('OL1 required')
        resolved_condition_config(config,row)
        if config['data']['sequence_length'] % row['tile_tokens']:
            raise ValueError('Scientific sequence must contain complete token tiles')
    ids = [r['id'] for r in condition_specs(config)]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Conditions must be nonempty and unique')
    assigned = [i for group in config['runpod']['worker_assignments'].values() for i in group]
    if sorted(assigned) != sorted(ids):
        raise ValueError('Each condition must be assigned exactly once')
    for key in ('max_steps','micro_batch_size','gradient_accumulation_steps','global_batch_size'):
        if isinstance(t.get(key),bool) or not isinstance(t.get(key),int) or t[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if t['micro_batch_size'] * t['gradient_accumulation_steps'] != t['global_batch_size']:
        raise ValueError('Batch decomposition mismatch')
    for key in ('peak_learning_rate','minimum_learning_rate','weight_decay','gradient_clip_norm','warmup_fraction'):
        if not math.isfinite(float(t[key])) or float(t[key]) < 0:
            raise ValueError(f'Invalid {key}')
    if not 0 <= t['warmup_fraction'] <= 1 or t['minimum_learning_rate'] > t['peak_learning_rate']:
        raise ValueError('Invalid LR schedule')
    if t.get('fp16_ol1_overflow_policy') != 'skip_entire_boundary':
        raise ValueError('Atomic overflow handling required')
    if (v.get('documents'),v.get('complete_sequences'),v.get('input_tokens'),v.get('excluded_tail_tokens')) != (500,338,692224,1444):
        raise ValueError('Full validation required')
    if config['diagnostics'].get('clipping_frontier') is not None:
        raise ValueError('Post-hoc clipping not approved')
    for key in ('gradient_interaction','ol1_adaptive_directions'):
        if not config['diagnostics'].get(key):
            raise ValueError('Training-time gradient diagnostics required')
    checkpoints = config['checkpoints']
    if not checkpoints.get('retain_final') or t['max_steps'] not in checkpoints['model_steps'] or t['max_steps'] not in checkpoints['optimizer_steps']:
        raise ValueError('Final model/recovery retention required')

def run_code_identity():
    # Training/deployment identity stays stable when approved result analyses
    # are appended later. Enumerate only runtime dependencies, not report code.
    names = ('00_setup_remote.sh','01_validate_preflights.py','02_train.py','03_verify.py',
        '04_monitor.py','05_remote_preflight.py','06_build_cache_from_hf.py','07_prepare.py',
        '08_pipeline.sh','09_local_smoke.py','10_remote.py','12_deadline_guard.py',
        '13_local_stop_guard.ps1','15_seal_training.py','16_retrieve_training.py',
        'topology.py','tile_gate.py','frozen.py','initialization.py','optimizer_boundary.py','diagnostics.py','run_config.py','training.py')
    paths = [RUN_DIR/name for name in names]
    paths += [RUN_DIR/'inputs/random-initialization/provenance.json']
    paths += [RUN_DIR.parent/'004-2026-08-29-pythia14m-full-pass-l1n'/n for n in
              ('run_config.py','initialization.py','optimizer_boundary.py','diagnostics.py','06_build_cache_from_hf.py')]
    paths += list((REPO_ROOT/'src/sparsity_research').glob('*.py'))
    files = []
    digest = hashlib.sha256()
    for path in sorted(paths):
        name = path.relative_to(REPO_ROOT).as_posix()
        payload = path.read_bytes().replace(b'\r\n', b'\n')
        sha = hashlib.sha256(payload).hexdigest()
        files.append({'path':name,'bytes':len(payload),'sha256':sha})
        digest.update(f'{name}\0{sha}\n'.encode())
    return {'files':files,'content_sha256':digest.hexdigest(),'text_normalization':'CRLF to LF'}
