"""Run the missing complete checkpoint sweeps; all inputs and outputs are durable."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'src'))
from clipping import _calibrate, _evaluate_point, _thresholds_for_target, nondominated


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    tmp.replace(path)


def log(path, row):
    row = {'time':datetime.now(timezone.utc).isoformat(), **row}
    with path.open('a') as f:
        f.write(json.dumps(row)+'\n')
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--keys', nargs='*')
    parser.add_argument('--targets', nargs='*', type=float)
    args = parser.parse_args()
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = json.loads((HERE/'input-manifest.json').read_text())
    targets = args.targets if args.targets is not None else data['targets']
    if not targets or any(t not in data['targets'] for t in targets):
        raise ValueError('Targets must come from the declared grid')
    rows = [r for r in data['checkpoints'] if r['evaluate'] and
            (args.keys is None or r['key'] in args.keys)]
    if not rows or (args.keys is not None and set(args.keys)!={r['key'] for r in rows}):
        raise ValueError('Unrecognized checkpoint selection')
    tokens = {}
    for split in ('train','validation'):
        path = HERE/'inputs'/(split+'.bin')
        assert sha(path)==data['caches'][split]['transferred_sha256']
        tokens[split] = np.memmap(path, dtype=np.int32, mode='r')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    events = out/'events.jsonl'
    code_paths = [HERE/'02_evaluate.py', HERE/'clipping.py', *sorted((ROOT/'src/sparsity_research').glob('*.py'))]
    identity = {'input_manifest_sha256':sha(HERE/'input-manifest.json'),
                'source_preflight':json.loads((HERE/'preflight.json').read_text()),
                'code':{p.relative_to(ROOT).as_posix():sha(p) for p in code_paths},
                'torch':torch.__version__, 'transformers':transformers.__version__,
                'python':sys.version,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
                'precision':'FP32 parameters, FP16 autocast, eager attention',
                'near_zero_thresholds':[0,0.001,0.01]}
    protocol = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = {'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
                'identity':identity,'protocol_sha256':protocol,'keys':[r['key'] for r in rows],
                'targets':targets,'expected_points':len(rows)*len(targets),
                'pid':os.getpid(),'command':sys.argv}
    write(out/'manifest.json',manifest)
    started = time.perf_counter()
    done = 0
    try:
        for source in rows:
            folder = out/source['key']
            folder.mkdir(exist_ok=True)
            checkpoint = args.checkpoint_root/source['key']
            for f in source['files']:
                p = checkpoint/f['name']
                assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'], str(p)
            log(events, {'event':'checkpoint_start','key':source['key']})
            model = load_checkpoint_pythia(AutoModelForCausalLM, checkpoint, torch=torch)
            assert topology_metadata(model)==source['topology']
            model.config.use_cache = False
            model.config._attn_implementation = 'eager'
            model.to(device='cuda', dtype=torch.float32)
            model.eval()
            torch.cuda.reset_peak_memory_stats()
            calpath = folder/'calibration.json'
            if calpath.exists():
                artifact = json.loads(calpath.read_text())
                assert artifact['protocol_sha256']==protocol
                calibration = artifact['calibration']
            else:
                calibration = _calibrate(model,tokens['train'],targets=tuple(data['targets']),
                                         blocks=10,block_size=2048,device=torch.device('cuda'),torch=torch,np=np)
                write(calpath, {'protocol_sha256':protocol,'source':source,'calibration':calibration})
            log(events, {'event':'calibration_complete','key':source['key'],
                         'seconds':calibration['wall_seconds']})
            for target in targets:
                result_path = folder/f'p{round(target*10)}.json'
                if result_path.exists():
                    point = json.loads(result_path.read_text())
                    assert point['protocol_sha256']==protocol
                else:
                    point = _evaluate_point(model,tokens['validation'],_thresholds_for_target(calibration,target),
                                            block_size=2048,batch_size=1,device=torch.device('cuda'),torch=torch,np=np)
                    if target==0 and abs(point['validation']['loss']-source['source_loss'])>5e-4:
                        write(folder/'failed-p0.json',point)
                        raise ValueError(f'p=0 loss mismatch: {source["key"]}')
                    assert all(r['nonfinite']==0 for r in point['activation_rows'])
                    point.update({'protocol_sha256':protocol,'source_id':source['id'],
                                  'checkpoint_content_sha256':source['checkpoint_content_sha256'],
                                  'target_sparsity':target,'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
                    write(result_path,point)
                done += 1
                elapsed = time.perf_counter()-started
                log(events, {'event':'point_complete','key':source['key'],'p':target,
                             'loss':point['validation']['loss'],'R_model':point['logical_products']['R_model'],
                             'seconds':point['evaluation_seconds'],'tokens_per_second':point['input_tokens_per_second'],
                             'complete':done,'expected':manifest['expected_points'],
                             'elapsed_seconds':elapsed,'estimated_remaining_seconds':elapsed/done*(manifest['expected_points']-done)})
            if all((folder/f'p{i}.json').exists() for i in range(10)):
                points = [json.loads((folder/f'p{i}.json').read_text()) for i in range(10)]
                baseline = points[0]['validation']['loss']
                for point, flag in zip(points,nondominated(points)):
                    point['loss_delta_from_zero_threshold'] = point['validation']['loss']-baseline
                    point['nondominated_within_checkpoint'] = flag
                write(folder/'sweep.json',{'status':'complete_verified','source':source,'points':points})
            del model
            gc.collect()
            torch.cuda.empty_cache()
        manifest.update(status='completed',completed_points=done,elapsed_seconds=time.perf_counter()-started)
    except BaseException as e:
        manifest.update(status='failed',completed_points=done,error=repr(e))
        raise
    finally:
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write(out/'manifest.json',manifest)


if __name__=='__main__':
    main()
