"""Per-attempt scientific/transfer checks and complete cohort reconciliation."""
import argparse
import hashlib
import json
from pathlib import Path
from run_config import (RUN_DIR,load_config,condition_specs,resolved_condition_config,
    EXPECTED_INITIAL_PARAMETER_SHA256,EXPECTED_SCHEDULE_SHA256,require_validation_coverage,
    run_code_identity,write_json)
from sparsity_research.artifacts import config_sha256
from sparsity_research.metrics import LOGICAL_OPERATIONS

def verify_attempt(path,config):
    manifest=json.loads((path/'manifest.json').read_text())
    metrics=json.loads((path/'metrics.json').read_text())
    spec=next(r for r in condition_specs(config) if r['id']==metrics['condition']['id'])
    actual_config=json.loads(json.dumps(__import__('yaml').safe_load((path/'config.yaml').read_text())))
    expected=resolved_condition_config(config,spec)
    if config_sha256(actual_config)!=config_sha256(expected) or metrics['condition']!=spec:
        raise ValueError('Condition/config drift')
    if manifest['status']!='completed' or manifest['initial_parameter_sha256']!=EXPECTED_INITIAL_PARAMETER_SHA256 or manifest['training_schedule_hash']!=EXPECTED_SCHEDULE_SHA256:
        raise ValueError('Completion/initialization/schedule mismatch')
    if manifest['run_code']!=run_code_identity():
        raise ValueError('Source identity changed')
    training=metrics['training']
    if training['completed_steps']!=config['training']['max_steps'] or training['optimizer_step_count']!=training['completed_steps'] or training['overflow_steps']:
        raise ValueError('Incomplete or skipped scientific updates')
    for coverage in metrics['validation'].values():
        require_validation_coverage(coverage,config)
    events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
    updates=[r for r in events if r.get('event')=='train']
    if [r['step'] for r in updates]!=list(range(1,config['training']['max_steps']+1)):
        raise ValueError('Missing boundary events')
    names=sorted(f'{site}.layer_{i}' for site in spec['pressure_sites'] for i in range(6))
    for row in updates:
        if row['pressure_capture_names']!=names or row['pressure_capture_tensor_count']!=len(names) or row['optimizer_step_skipped'] or not row['ol1_correction_applied']:
            raise ValueError('Pressure/boundary mismatch')
    activation=json.loads((path/'diagnostics/activation_statistics.json').read_text())
    if len(activation['rows'])!=72 or len(activation['sign_energy_rows'])!=12 or len(activation['occupancy_rows'])!=24 or len(activation['layernorm_affine'])!=12 or len(activation['corner_rows'])!=12:
        raise ValueError('Diagnostic coverage incomplete')
    if any(r['nonfinite'] for r in activation['rows']):raise ValueError('Nonfinite activations')
    for step in config['validation']['steps']:
        require_validation_coverage(metrics['validation'][f'step_{step:06d}'],config)
    for step in config['diagnostics']['steps']:
        require_validation_coverage(metrics['validation'][f'activation_step_{step:06d}'],config)
        statistics=activation if step==config['training']['max_steps'] else json.loads((path/'diagnostics'/f'step_{step:06d}'/'activation_statistics.json').read_text())
        if len(statistics['corner_rows'])!=12 or any(r['certificate_mismatches'] or r['empty']!=r['rejected'] for r in statistics['corner_rows']):
            raise ValueError('Tile corner certificate failed')
        if len(statistics['rows'])!=72 or any(r['nonfinite'] for r in statistics['rows']):
            raise ValueError('Incomplete/nonfinite intermediate statistics')
        for row in statistics['corner_rows']:
            if sum(row['feature_group_tiles'])!=row['tiles'] or sum(row['feature_group_rejected'])!=row['rejected']:
                raise ValueError('Feature-group counts do not reconcile')
    products=json.loads((path/'diagnostics/logical_products.json').read_text())
    if set(products['measured']['per_operation'])!=set(LOGICAL_OPERATIONS):
        raise ValueError('Logical operations incomplete')
    root=path/'checkpoints'
    for step in config['checkpoints']['model_steps']:
        checkpoint=root/f'step_{step:06d}'
        if not (checkpoint/'model.safetensors').exists():raise ValueError('Missing model checkpoint')
        if step in config['checkpoints']['optimizer_steps'] and not (checkpoint/'training_state.pt').exists():
            raise ValueError('Missing recovery state')
    inventory=json.loads((path/'transfer_inventory.json').read_text())
    if inventory['total_bytes']!=sum(r['bytes'] for r in inventory['files']):
        raise ValueError('Inventory byte total differs')
    for row in inventory['files']:
        target=path/row['path']
        if not target.resolve().is_relative_to(path.resolve()):raise ValueError('Unscoped inventory path')
        with target.open('rb') as handle:sha=hashlib.file_digest(handle,'sha256').hexdigest()
        if target.stat().st_size!=row['bytes'] or sha!=row['sha256']:raise ValueError(f'Hash mismatch: {target}')
    return spec['id']

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=Path)
    args=parser.parse_args();config=load_config()
    paths=[args.attempt] if args.attempt else sorted((RUN_DIR/'artifacts/attempts').glob('*'))
    ids=[verify_attempt(path,config) for path in paths if path.is_dir()]
    if not args.attempt and sorted(ids)!=sorted(r['id'] for r in condition_specs(config)):
        raise ValueError('Cohort incomplete or duplicate')
    if not args.attempt:
        write_json(RUN_DIR/'artifacts/verification.json',{'status':'verified','condition_count':len(ids),'conditions':ids})
    print(json.dumps({'verified_conditions':ids}))
