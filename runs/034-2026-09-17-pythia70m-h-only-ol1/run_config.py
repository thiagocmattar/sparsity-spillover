"""Run034: matched 70M training, A4/A7 thresholds and pressure only at h."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import yaml
from sparsity_research.artifacts import config_sha256
from sparsity_research.pressure import parse_pressure_config
from _reuse_run017 import load_run017_module

RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
DEFAULT_CONFIG = RUN_DIR / 'config.yaml'
_FROZEN = load_run017_module('_run034_frozen_run017_config', 'run_config.py')
_FROZEN.RUN_DIR = _FROZEN._BASE.RUN_DIR = RUN_DIR
_FROZEN.REPO_ROOT = _FROZEN._BASE.REPO_ROOT = REPO_ROOT
_FROZEN.DEFAULT_CONFIG = _FROZEN._BASE.DEFAULT_CONFIG = DEFAULT_CONFIG
for _name in ('A4_SITES', 'A7_SITES', 'DIAGNOSTIC_SITES', 'ONE_SIDED_SITES',
              'SYMMETRIC_SITES', 'EXPECTED_INITIAL_PARAMETER_SHA256',
              'EXPECTED_SCHEDULE_SHA256', 'EXPECTED_CEILINGS', 'expected_ceiling',
              'site_gates', 'resolved_condition_config', 'build_schedule',
              'mapping', 'repo_path', 'load_verified_caches', 'microbatches_for_step',
              'require_cuda', 'seed_everything', 'require_training_cache',
              'require_validation_coverage', 'parameter_sha256',
              'inventory_content_sha256', 'cache_identity', 'git_identity', 'write_json'):
    globals()[_name] = getattr(_FROZEN, _name)
EXPECTED_THRESHOLDS = (0., .01, .05, .1, .5)
EXPECTED_CONDITION_IDS = tuple(f'{family}-h-ol1-kappa-{k:g}'.replace('.', 'p')
                               for family in ('a4', 'a7') for k in EXPECTED_THRESHOLDS)
EXPECTED_WORKERS = {key: (key,) for key in EXPECTED_CONDITION_IDS}
EXPECTED_PRESSURE_SITES = ('h',)
EXPECTED_PRESSURE_CAPTURE_NAMES = tuple(f'h.layer_{i}' for i in range(6))
EXPECTED_PRESSURE_CAPTURE_NAMES_SHA256 = hashlib.sha256('\n'.join(EXPECTED_PRESSURE_CAPTURE_NAMES).encode()).hexdigest()
EXPECTED_PRESSURE_CAPTURE_TENSOR_COUNT = 6
EXPECTED_MODEL_CHECKPOINTS = (0,1,2,4,8,16,32,64,128,256,512,712)
EXPECTED_OPTIMIZER_CHECKPOINTS = (256,512,712)


def condition_specs(config):
    rows = [r for r in _FROZEN.condition_specs(config) if not r['is_control']]
    for i, row in enumerate(rows, 1):
        row.update(id=row['id'].replace('-ol1-', '-h-ol1-'), order=i,
                   step=row['step'] + '(h)', pressure_sites=['h'],
                   pressure_weight=float(config['conditions']['pressure_weight']),
                   step_budget=float(config['conditions']['step_budget']),
                   label=f"kappa={row['gate_threshold']:g}, OL1(h), lambda=1")
    return rows


def worker_conditions(config, worker_id):
    rows = {r['id']: r for r in condition_specs(config)}
    return [deepcopy(rows[c]) for c in config['runpod']['worker_assignments'][worker_id]]


def load_config(path=DEFAULT_CONFIG):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    validate_config(config)
    return config


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError('Expected a config mapping')
    c, t, ck = config['conditions'], config['training'], config['checkpoints']
    if c['pressure_sites'] != ['h'] or c['pressure_method'] != 'orthogonal_l1':
        raise ValueError('This control requires h-only orthogonal_l1')
    if t['micro_batch_size'] * t['gradient_accumulation_steps'] != t['global_batch_size']:
        raise ValueError('Batch decomposition does not equal global batch')
    if t['max_steps'] <= 0 or config['data']['sequence_length'] <= 1:
        raise ValueError('Training dimensions must be positive')
    if t['max_steps'] not in ck['model_steps'] or t['max_steps'] not in ck['optimizer_steps']:
        raise ValueError('Final model and recovery state are required')
    if not set(ck['optimizer_steps']).issubset(ck['model_steps']):
        raise ValueError('Optimizer checkpoints require a model checkpoint')
    for row in condition_specs(config):
        resolved = resolved_condition_config(config, row)
        parse_pressure_config(resolved['activation_pressure'])
    assigned = [v for values in config['runpod']['worker_assignments'].values() for v in values]
    if sorted(assigned) != sorted(r['id'] for r in condition_specs(config)):
        raise ValueError('Worker assignments must cover each condition exactly once')
    if config['model']['released_weights_loaded'] or config['model']['initialization'] != 'random':
        raise ValueError('Only canonical random pretraining initialization is allowed')


validate_science_config = validate_config


def run_code_identity():
    # Include executed frozen helpers and shared primitives, not artifacts or tests.
    paths = list(RUN_DIR.glob('*.py')) + [RUN_DIR/'config.yaml', RUN_DIR/'architecture_config.json',
                                        RUN_DIR/'prelaunch/initialization/metadata.json']
    for folder in ('004-2026-08-29-pythia14m-full-pass-l1n',
                   '017-2026-09-01-pythia70m-selected-ladder-portable-init',
                   '018-2026-09-01-pythia70m-selected-ladder-canonical-init'):
        paths.extend((RUN_DIR.parent/folder).glob('*.py'))
    paths.extend((REPO_ROOT/'src/sparsity_research').glob('*.py'))
    files = []
    for path in sorted(set(paths)):
        if path.name.startswith('test_') or path.name.startswith(('07_', '08_', '09_', '10_', '11_', '12_')):
            continue
        raw = path.read_bytes()
        files.append({'path': path.relative_to(REPO_ROOT).as_posix(), 'bytes':len(raw),
                      'sha256':hashlib.sha256(raw).hexdigest()})
    digest = hashlib.sha256()
    for row in files:
        digest.update(f"{row['path']}\0{row['sha256']}\n".encode())
    return {'files':files, 'content_sha256':digest.hexdigest()}


def approved_identity():
    return {'config_sha256':config_sha256(load_config()), 'run_code':run_code_identity()}


def git_identity():
    receipt = RUN_DIR / 'prelaunch/source-receipt.json'
    if not receipt.exists():
        return _FROZEN.git_identity()
    value = json.loads(receipt.read_text())
    for row in value['files']:
        path = REPO_ROOT / row['path']
        if path.stat().st_size != row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError(f"Deployed source identity mismatch: {row['path']}")
    return {'git_commit':value['git_commit'], 'git_dirty':None,
            'deployment':'hash_verified_scoped_archive_of_recorded_commit',
            'source_file_count':len(value['files'])}
