"""Evaluate signed activation histograms without updating weights or gates."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import time

from density import GRID, GROUPS, SITES, SignedHistogram, pool, read_json, sha, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'src'))


def log(path, row):
    row = {'utc':datetime.now(timezone.utc).isoformat(), **row}
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(row),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--keys',nargs='+')
    parser.add_argument('--max-blocks',type=int)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args = parser.parse_args()
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata
    from sparsity_research.capture import ActivationCapture
    from sparsity_research.metrics import ActivationAccumulator
    from sparsity_research.evaluation import evaluate_complete_blocks
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if device.type=='cpu' and args.max_blocks is None:
        raise ValueError('CPU is restricted to smoke evaluations')
    torch.backends.cuda.matmul.allow_tf32 = False
    data = read_json(HERE/'input-manifest.json')
    sources = [r for r in data['checkpoints'] if args.keys is None or r['key'] in args.keys]
    if not sources or args.keys is not None and set(args.keys)!={r['key'] for r in sources}:
        raise ValueError('Unknown checkpoint selection')
    if args.max_blocks is not None and not 1<=args.max_blocks<338:
        raise ValueError('Smoke scope must be between 1 and 337 blocks')
    tokens_path = HERE/'inputs/validation.bin'
    assert sha(tokens_path)==data['validation']['sha256']
    all_tokens = np.memmap(tokens_path,dtype=np.int32,mode='r')
    tokens = all_tokens if args.max_blocks is None else all_tokens[:args.max_blocks*2048]
    blocks = len(tokens)//2048
    out = args.output
    out.mkdir(parents=True,exist_ok=False)
    events = out/'events.jsonl'
    code = [*sorted(HERE.glob('*.py')),*sorted((ROOT/'src/sparsity_research').glob('*.py'))]
    manifest = {'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
                'input_manifest_sha256':sha(HERE/'input-manifest.json'),
                'code':{p.relative_to(ROOT).as_posix():sha(p) for p in code},
                'command':sys.argv,'pid':os.getpid(),'keys':[r['key'] for r in sources],
                'expected_blocks':blocks*len(sources),'blocks_per_checkpoint':blocks,
                'smoke':args.max_blocks is not None,'python':sys.version,
                'torch':torch.__version__,'transformers':transformers.__version__,
                'device':args.device,'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,
                'precision':'FP32 parameters; FP16 autocast' if device.type=='cuda' else 'CPU FP32 smoke'}
    write_json(out/'manifest.json',manifest)
    started = time.perf_counter()
    done = 0
    try:
        for source in sources:
            checkpoint = HERE/'inputs/checkpoints'/source['key']
            for entry in source['files']:
                path = checkpoint/entry['name']
                assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
            model = load_checkpoint_pythia(AutoModelForCausalLM,checkpoint,torch=torch)
            assert topology_metadata(model)==source['topology']
            model.config.use_cache = False
            model.config._attn_implementation = 'eager'
            model.to(device=device,dtype=torch.float32).eval()
            if device.type=='cuda':
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            moments = ActivationAccumulator(thresholds=(0.,.001,.01))
            hist = SignedHistogram(torch=torch,device=device)
            current_blocks, loss_sum = 0, 0.
            checkpoint_start = time.perf_counter()
            log(events,{'event':'checkpoint_start','key':source['key'],'completed_blocks':done})
            with ActivationCapture(model,list(SITES),torch=torch) as capture:
                def observe(output, sequences):
                    nonlocal current_blocks, loss_sum
                    moments.update(capture.activations,torch=torch)
                    hist.update(capture.activations)
                    capture.clear()
                    current_blocks += sequences
                    loss_sum += float(output.loss.detach().cpu())*sequences
                    if current_blocks%20==0 or current_blocks==blocks:
                        elapsed = time.perf_counter()-checkpoint_start
                        completed = done+current_blocks
                        log(events,{'event':'progress','key':source['key'],'checkpoint_blocks':current_blocks,
                                    'completed_blocks':completed,'expected_blocks':manifest['expected_blocks'],
                                    'loss':loss_sum/current_blocks,'tokens_per_second':2048*current_blocks/elapsed,
                                    'estimated_remaining_seconds':(manifest['expected_blocks']-completed)*elapsed/current_blocks})
                coverage = evaluate_complete_blocks(model=model,tokens=tokens,block_size=2048,batch_size=1,
                    device=device,torch=torch,np=np,autocast_dtype=torch.float16 if device.type=='cuda' else None,
                    after_batch=observe)
                hooks = [asdict(site) for site in capture.metadata]
            seconds = time.perf_counter()-checkpoint_start
            rows = hist.rows(moments.rows())
            assert len(rows)==len(SITES)*model.config.num_hidden_layers
            for row in rows:
                width = model.config.intermediate_size if row['name'].split('.layer_')[0]=='h' else model.config.hidden_size
                assert row['total']==blocks*2048*width
            sites = {site:pool(rows,site,(site,)) for site in SITES}
            groups = {name:pool(rows,name,aliases) for name,aliases in GROUPS.items()}
            coverage['complete_block_coverage'] = args.max_blocks is None
            coverage['full_source_tokens'] = len(all_tokens)
            coverage['full_source_excluded_tail_tokens'] = len(all_tokens)%2048
            artifact = {'source':source,'coverage':coverage,'grid':GRID,'rows':rows,
                        'sites':sites,'groups':groups,'hooks':hooks,'evaluation_seconds':seconds,
                        'input_manifest_sha256':manifest['input_manifest_sha256'],
                        'peak_reserved_bytes':torch.cuda.max_memory_reserved() if device.type=='cuda' else None}
            if args.max_blocks is None:
                artifact['reference_comparison'] = {
                    'loss_delta':coverage['loss']-source['source_loss'],
                    'sites':{site:{'exact_zero_fraction_delta':sites[site]['exact_zero_count']/sites[site]['total']-
                                  source['reference_sites'][site]['exact_zero_count']/source['reference_sites'][site]['total'],
                                  'rms_delta':sites[site]['rms']-source['reference_sites'][site]['rms']} for site in SITES}}
            write_json(out/(source['key']+'.json.gz'),artifact)
            if args.max_blocks is None and abs(coverage['loss']-source['source_loss'])>5e-4:
                raise ValueError(f'Full-validation loss mismatch: {source["key"]}')
            done += blocks
            log(events,{'event':'checkpoint_complete','key':source['key'],'completed_blocks':done,
                        'loss':coverage['loss'],'tokens_per_second':2048*blocks/seconds,
                        'seconds':seconds,'estimated_remaining_seconds':(manifest['expected_blocks']-done)*seconds/blocks})
            del model, capture, moments, hist
            gc.collect()
            if device.type=='cuda': torch.cuda.empty_cache()
        manifest['status'] = 'completed'
    except BaseException as error:
        manifest.update(status='failed',error=repr(error))
        raise
    finally:
        manifest.update(completed_blocks=done,elapsed_seconds=time.perf_counter()-started,
                        finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(out/'manifest.json',manifest)


if __name__=='__main__':
    main()
