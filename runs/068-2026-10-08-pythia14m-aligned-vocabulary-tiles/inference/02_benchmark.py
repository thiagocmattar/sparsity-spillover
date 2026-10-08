"""Full native-reference qualification and paired head/full-forward measurements."""
import argparse
import json
import platform
import time
import traceback
import numpy as np
import torch
import transformers
from io_utils import RUN, REPO, read, write, record, verify, load_model, configure_cuda, source_identity
from head import install, extension
from measurement import Graph, paired_timing, summarize, compare, loss_sum
import diagnostics
from forward import forward


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--replicate',type=int,required=True)
    parser.add_argument('--attempt',required=True)
    parser.add_argument('--development',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();cfg=read(RUN/'config.json')
    inputs=read(RUN/'provenance'/('inputs.json' if args.development else 'trained-inputs.json'))
    if not args.attempt.replace('-','').replace('_','').isalnum() or not 1<=args.replicate<=cfg['process_replicates']:
        parser.error('Fresh simple attempt and declared replicate required')
    source=next(c for c in inputs['checkpoints'] if c['id']==args.model)
    if not args.development and (inputs.get('development') or source.get('development_model')):
        raise ValueError('Development models cannot be accepted scientific endpoints')
    dest=RUN/'artifacts/attempts'/args.attempt;dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();result={'status':'running','arguments':vars(args),'config':record(RUN/'config.json'),
                                   'source':source,'runtime':{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
                                   'transformers':transformers.__version__,'cuda':torch.version.cuda,'python':platform.python_version(),
                                   'numpy':np.__version__,'gpu_uuid':str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable')),
                                   'gpu_properties':str(torch.cuda.get_device_properties(0)),
                                   'tf32':False,'fp16_reduced_precision_reduction':False}}
    def emit(stage,**fields):
        row={'stage':stage,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'elapsed_seconds':time.monotonic()-started,**fields}
        write(dest/'status.json',row)
        with (dest/'events.jsonl').open('a',encoding='utf-8',newline='\n') as handle:handle.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)
    def save_samples(name,graphs,values):
        with (dest/f'samples-{name}.jsonl').open('x',encoding='utf-8',newline='\n') as handle:
            def sink(row):handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush()
            samples=paired_timing(graphs,values,passes=2 if args.smoke else cfg['timing_passes'],
                                 seed=cfg['timing_seed']+args.replicate-1,sink=sink)
        return {'summary':summarize(samples),'boundary':name}
    try:
        configure_cuda(cfg['runtime_seed'])
        result['source_checkout']=source_identity(development=args.development)
        for key,actual in [('torch',torch.__version__.split('+')[0]),('transformers',transformers.__version__),('numpy',np.__version__),('cuda',torch.version.cuda)]:
            if actual!=cfg['runtime'][key]:raise ValueError('Runtime pin mismatch: '+key)
        if not args.development and torch.cuda.get_device_name()!=cfg['gpu']:raise ValueError('RTX5090 required for accepted cohort')
        for row in source['files']+source['provenance']+inputs['runtime_dependencies']:verify(row)
        validation=np.memmap(verify(inputs['validation']),dtype=np.int32,mode='r')
        if divmod(len(validation),2048)!=(338,1444):raise ValueError('Full validation coverage mismatch')
        extension();emit('load')
        checkpoint=REPO/source['checkpoint']
        models={};graphs={};implementation={}
        indices=np.random.default_rng(cfg['timing_seed']).choice(338,4 if args.smoke else cfg['timing_inputs'],replace=False)
        ids_list=[torch.tensor(validation[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in indices]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,cache_enabled=True):
            native=load_model(checkpoint)
            for mode in cfg['modes']:
                model=load_model(checkpoint)
                implementation[mode]=install(model,mode,cfg)
                models[mode]=model
                emit('capture',mode=mode)
                graphs[mode]=Graph(lambda ids,m=model:forward(m,ids),[ids_list[0]])
            result['implementation']=implementation
            result['setup_seconds']={mode:g.setup_seconds for mode,g in graphs.items()}
            result['source_files']=[record(RUN/n) for n in ('head.py','head.cu','forward.py','measurement.py','diagnostics.py','02_benchmark.py')]
            write(dest/'manifest.json',result)
            quality={mode:[] for mode in graphs};sums={'native':0.,**{mode:0. for mode in graphs}};predictions=0;head_rows=[]
            # Untimed hooks provide native operands, never instrumentation in timed graphs.
            capture={}
            def pre_hook(_m,values):capture['pre']=values[0].detach().clone()
            def post_hook(_m,_values,value):capture['post']=value.detach().clone()
            before=native.head_input_gate.register_forward_pre_hook(pre_hook)
            after=native.head_input_gate.register_forward_hook(post_hook)
            blocks=8 if args.smoke else cfg['validation']['blocks'];validation_started=time.monotonic()
            for block in range(blocks):
                ids=torch.tensor(validation[block*2048:(block+1)*2048].copy(),device='cuda',dtype=torch.long)[None]
                reference=native(input_ids=ids,use_cache=False).logits
                value,count=loss_sum(reference,ids);sums['native']+=value;predictions+=count
                head_rows.append(diagnostics.counts(capture['post'],source['kappa'],cfg['kernel_tile_tokens'],cfg['kernel_tile_vocabulary']))
                for mode,graph in graphs.items():
                    graph.stage([ids]);output=graph()
                    check=compare(reference,output,cfg['numerical_bounds']);check['block']=block
                    quality[mode].append(check)
                    if not check['pass']:
                        write(dest/'numerical-failure.json',{'mode':mode,'check':check})
                        raise ValueError('Full logit qualification failed')
                    value,_=loss_sum(output,ids);sums[mode]+=value
                    if mode!='dense' and args.replicate==1:
                        head=models[mode].embed_out;head.project(count=True)
                        actual=head.stats.sum((0,1)).cpu().tolist()
                        expected=head_rows[-1]
                        expected=[expected['issued_mma'],expected['skipped_mma']] if head.skip else [expected['potential_mma'],0]
                        if actual!=expected:raise ValueError('Real-operand work counters differ')
                if (block+1)%16==0 or block+1==blocks:
                    elapsed=time.monotonic()-validation_started
                    emit('validation',blocks=block+1,target=blocks,loss={mode:total/predictions for mode,total in sums.items()},
                         tokens_per_second=(block+1)*2048/elapsed,remaining_seconds=(blocks-block-1)*elapsed/(block+1))
            before.remove();after.remove()
            losses={mode:total/predictions for mode,total in sums.items()}
            if any(abs(losses[mode]-losses['native'])>cfg['numerical_bounds']['validation_loss_atol'] for mode in graphs):raise ValueError('Pooled loss qualification failed')
            coverage={'documents':500 if blocks==338 else None,'blocks':blocks,'input_tokens':blocks*2048,
                      'predictions':predictions,'excluded_tail_tokens':1444 if blocks==338 else None}
            write(dest/'quality.json',{'rows':quality,'loss':losses,'coverage':coverage,'bounds':cfg['numerical_bounds'],
                                      'native_reference':'unmodified FP32 parameters, FP16 autocast, SDPA B1/T2048; Run066 B32 loss is provenance only'})
            write(dest/'head-diagnostics.json',{'counts':diagnostics.pool(head_rows),'per_block':head_rows,'coverage':coverage,
                                               'native_dtype':'float32','gate_tile':[16,16],
                                               'consumer_tile':[cfg['kernel_tile_tokens'],16],
                                               'mma_counters':'includes padded M/N instructions; separate from logical scalar-product counts'})
            emit('timing',boundary='full-forward',loss=losses)
            result['timings']={'full_forward':save_samples('full-forward',graphs,[[ids] for ids in ids_list])}
            # Capture the same checkpoint's native final-LN inputs for every timed sequence.
            head_inputs=[]
            handle=native.head_input_gate.register_forward_pre_hook(pre_hook)
            for ids in ids_list:
                native(input_ids=ids,use_cache=False)
                head_inputs.append(capture['pre'].clone())
            handle.remove()
            combined={}
            for mode,model in models.items():
                combined[mode]=Graph(lambda value,m=model:m.embed_out(m.head_input_gate(value)),[head_inputs[0]])
            head_quality={'gate_cast_projection':{},'projection_only':{}}
            for mode,graph in combined.items():
                checks=[]
                for value in head_inputs:
                    reference=native.embed_out(native.head_input_gate(value))
                    graph.stage([value]);check=compare(reference,graph(),cfg['numerical_bounds']);checks.append(check)
                    if not check['pass']:raise ValueError('Gate/cast/projection graph qualification failed')
                head_quality['gate_cast_projection'][mode]=checks
            emit('timing',boundary='gate-cast-projection',loss=losses)
            result['timings']['gate_cast_projection']=save_samples('gate-cast-projection',combined,[[x] for x in head_inputs])
            del combined
            # Projection-only excludes preparation and is explicitly diagnostic, not the primary speed claim.
            raw=[native.head_input_gate(x).half().reshape(2048,128) for x in head_inputs]
            supports=[]
            for x in head_inputs:
                live=x[:,::16,::16].reshape(128,8)>source['kappa']
                supports.append((live.int()*(1<<torch.arange(8,device='cuda'))).sum(-1).int())
            head_graphs={}
            for mode,model in models.items():
                weight=model.embed_out.packed_weight
                if mode=='dense':
                    head_graphs[mode]=Graph(lambda x,s,w=weight:torch.nn.functional.linear(x,w),[raw[0],supports[0]])
                else:
                    head=model.embed_out
                    def project(x,s,h=head):
                        extension().project(x,s,h.mma_weight,h.out,h.stats,h.skip,False,h.tile_tokens,h.tile_vocabulary)
                        return h.out
                    head_graphs[mode]=Graph(project,[raw[0],supports[0]])
            emit('timing',boundary='projection-only',loss=losses)
            for mode,graph in head_graphs.items():
                checks=[]
                for x,s in zip(raw,supports):
                    reference=torch.nn.functional.linear(x,native.embed_out.weight.half())
                    graph.stage([x,s]);check=compare(reference,graph(),cfg['numerical_bounds']);checks.append(check)
                    if not check['pass']:raise ValueError('Projection-only graph qualification failed')
                head_quality['projection_only'][mode]=checks
            write(dest/'head-graph-quality.json',{'rows':head_quality,'input_indices':indices.tolist(),
                                                'bounds':cfg['numerical_bounds'],'scope':'each component graph on all timed real operands; full forwards separately cover all 338 blocks'})
            result['timings']['projection_only']=save_samples('projection-only',head_graphs,list(zip(raw,supports)))
            result.update(status='complete',qualified=True,loss=losses,coverage=coverage,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          evidence_label='development' if args.development or args.smoke else 'valid')
            emit('complete',loss=losses,peak_allocated_bytes=result['peak_allocated_bytes'])
    except Exception as exc:
        result.update(status='failed',error=str(exc),traceback=traceback.format_exc());emit('failed',error=str(exc))
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        write(dest/'manifest.json',result);write(dest/'result.json',result)
    return 0 if result['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
