"""Re-evaluate immutable Run060 endpoints with the same head diagnostics."""
import argparse
import gc
import hashlib
import json
from time import perf_counter
from head_site import load_checkpoint_pythia, topology_metadata
from run_config import RUN_DIR, load_config, load_verified_caches, write_json, parameter_sha256, run_code_identity


def verify_inputs(root=RUN_DIR):
    inventory = json.loads((root/'inputs/controls/inventory.json').read_text())
    for row in inventory['files']:
        path = root/row['path']
        if not path.resolve().is_relative_to((root/'inputs/controls').resolve()):
            raise ValueError('Unscoped control source')
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if path.stat().st_size != row['bytes'] or digest != row['sha256']:
            raise ValueError('Retained control input hash mismatch')
    return inventory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    from diagnostics import activation_diagnostic_validation, logical_product_validation
    from training import timed_validation
    from sparsity_research.metrics import weight_statistics
    config = load_config()
    inventory = verify_inputs()
    source = next(row for row in inventory['controls'] if row['id'] == args.control)
    target = RUN_DIR/'artifacts/controls'/source['id']
    target.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    write_json(target/'status.json', {'status':'running','source':source})
    _, tokens, _, metadata, cache_seconds = load_verified_caches(config,np=np)
    model = load_checkpoint_pythia(AutoModelForCausalLM, RUN_DIR/source['checkpoint'],torch=torch).cuda().eval()
    if model.config.head_input_gate is not None:
        raise ValueError('Control checkpoint must not have a head gate')
    if parameter_sha256(model) != source['parameter_sha256']:
        raise ValueError('Control parameters differ from retained endpoint')
    validation, loss_seconds = timed_validation(model=model,tokens=tokens,config=config,torch=torch,np=np)
    start = perf_counter()
    coverage, activation = activation_diagnostic_validation(model=model,tokens=tokens,config=config,torch=torch,np=np)
    activation_seconds = perf_counter()-start
    start = perf_counter()
    logical_coverage, measured, maximum = logical_product_validation(model=model,tokens=tokens,config=config,torch=torch,np=np,activation_statistics=activation)
    logical_seconds = perf_counter()-start
    write_json(target/'activation_statistics.json',activation)
    write_json(target/'logical_products.json',{'coverage':logical_coverage,'measured':measured,'maximum':maximum})
    write_json(target/'weight_statistics.json',{'inclusion':'all named parameters, including bias and normalization','rows':weight_statistics(model)})
    write_json(target/'metrics.json',{'status':'completed','source':source,'validation':validation,
        'activation_coverage':coverage,'logical_coverage':logical_coverage,'validation_metadata':metadata,
        'topology':topology_metadata(model),'run_code':run_code_identity(),
        'seconds':{'cache_verification':cache_seconds,'loss':loss_seconds,'activation':activation_seconds,
            'logical':logical_seconds,'total':perf_counter()-started}})
    write_json(target/'status.json', {'status':'completed','source':source})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({'control':source['id'],'status':'completed','loss':validation['loss']}),flush=True)


if __name__=='__main__':
    main()
