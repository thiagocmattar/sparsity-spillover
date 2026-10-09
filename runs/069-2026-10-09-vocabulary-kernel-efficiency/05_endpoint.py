"""Fresh final processes: complete native-reference qualification and three boundaries."""
import argparse
import json
import time
import numpy as np
import torch
from common import RUN, REPO, config, sources, read, verify, write, runtime, load_model, manifest_identity, check_runtime, emit, support
from kernel import make, dense
from measurement import Graph, compare, loss_sum, paired_timing, summarize
from forward import forward


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--replicate',type=int,required=True)
    p.add_argument('--development',action='store_true');p.add_argument('--blocks',type=int)
    args=p.parse_args();cfg=config();runtime(cfg['runtime_seed']);identity=manifest_identity(args.development)
    actual_runtime=check_runtime(args.development)
    if not 1<=args.replicate<=cfg['process_replicates']:p.error('Declared replicate required')
    if args.blocks and not args.development:p.error('Full 338 blocks required for final qualification')
    capture_root=RUN/('prelaunch/development-capture' if args.development else 'artifacts/capture')
    screen_root=RUN/('prelaunch/development-screen' if args.development else 'artifacts/screen')
    snapshot=next(r for r in read(capture_root/'manifest.json')['files'] if r['path'].endswith(args.model+'.pt'))
    data=torch.load(verify(snapshot),map_location='cpu',weights_only=True)
    screening=read(screen_root/args.model/'result.json')
    variants={v['name']:v for v in cfg['variants']}
    chosen=list(dict.fromkeys(['legacy_n256']+screening['shortlist']))
    dest=RUN/('prelaunch/development-endpoints' if args.development else 'artifacts/endpoints')/f'{args.model}-r{args.replicate}'
    dest.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    inputs=sources();source=next(s for s in inputs['checkpoints'] if s['id']==args.model)
    tokens=np.memmap(verify(inputs['validation']),dtype=np.int32,mode='r')
    result={'status':'running','identity':identity,'runtime':actual_runtime,'source':source,'snapshot':snapshot,
            'replicate':args.replicate,'shortlist':chosen,'qualified':False}
    write(dest/'result.json',result)
    def timed(name,graphs,values):
        emit(dest,'timing',boundary=name,elapsed_seconds=time.monotonic()-started)
        with (dest/f'samples-{name}.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            def sink(r):stream.write(json.dumps(r)+'\n');stream.flush()
            return summarize(paired_timing(graphs,values,passes=2 if args.development else cfg['timing_passes'],
                          seed=cfg['timing_seed']+args.replicate-1,sink=sink))
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            anchor=load_model(REPO/source['checkpoint']);models={};graphs={};bounds=cfg['numerical_bounds']
            ids_list=[ids.cuda() for ids in data['ids']]
            modes={'native':None}
            for name in chosen:
                for skip in (False,True):modes[name+('/sparse' if skip else '/no_skip')]=(variants[name],skip)
            for mode,spec in modes.items():
                model=load_model(REPO/source['checkpoint'])
                model.embed_out=dense(model.embed_out) if spec is None else make(model.embed_out,model.config.head_input_gate,*spec)
                # make's last argument is explicitly positional and is skip.
                models[mode]=model
                graphs[mode]=Graph(lambda ids,m=model:forward(m,ids),[ids_list[0]])
                emit(dest,'graph',mode=mode,elapsed_seconds=time.monotonic()-started)
            quality={mode:[] for mode in modes};sums={'anchor':0.,**{mode:0. for mode in modes}}
            predictions=0;blocks=args.blocks or cfg['validation']['blocks'];failed=set()
            val_start=time.monotonic()
            for block in range(blocks):
                ids=torch.tensor(tokens[block*2048:(block+1)*2048].copy(),device='cuda',dtype=torch.long)[None]
                reference=anchor(input_ids=ids,use_cache=False).logits
                total,count=loss_sum(reference,ids);sums['anchor']+=total;predictions+=count
                for mode,graph in graphs.items():
                    graph.stage([ids]);actual=graph();check=compare(reference,actual,bounds)
                    quality[mode].append(check)
                    if not check['pass']:failed.add(mode)
                    sums[mode]+=loss_sum(actual,ids)[0]
                if (block+1)%16==0 or block+1==blocks:
                    seconds=time.monotonic()-val_start
                    emit(dest,'validation',block=block+1,blocks=blocks,loss=sums['anchor']/predictions,
                         blocks_per_second=(block+1)/seconds,etc_seconds=(blocks-block-1)*seconds/(block+1),
                         elapsed_seconds=time.monotonic()-started)
            losses={mode:value/predictions for mode,value in sums.items()}
            for mode in graphs:
                if abs(losses[mode]-losses['anchor'])>bounds['validation_loss_atol']:failed.add(mode)
            write(dest/'quality.json',{'blocks':blocks,'predictions':predictions,'excluded_tail_tokens':1444,
                  'loss':losses,'checks':quality,'failed_modes':sorted(failed),'bounds':bounds})
            if 'native' in failed or any(mode.startswith('legacy_n256') for mode in failed):
                raise ValueError('Reference/current implementation failed unchanged qualification')
            for mode in failed:
                del graphs[mode],models[mode]
            result['excluded_modes']=sorted(failed);result['loss']=losses
            result['timings']={'full_forward':timed('full-forward',graphs,[[x] for x in ids_list])}
            del graphs,anchor
            torch.cuda.empty_cache()
            pre=[x.cuda() for x in data['pre']]
            combined={mode:Graph(lambda x,m=model:m.embed_out(m.head_input_gate(x)),[pre[0]]) for mode,model in models.items()}
            component_checks={'gate_cast_projection':{},'projection_only':{}}
            for mode,graph in combined.items():
                checks=[]
                for value in pre:
                    reference=models['native'].embed_out(models['native'].head_input_gate(value))
                    graph.stage([value]);checks.append(compare(reference,graph(),bounds))
                if not all(c['pass'] for c in checks):raise ValueError('Head-path graph qualification failed: '+mode)
                component_checks['gate_cast_projection'][mode]=checks
            result['timings']['gate_cast_projection']=timed('gate-cast-projection',combined,[[x] for x in pre])
            del combined
            raw=[models['native'].head_input_gate(x).half().reshape(2048,128) for x in pre]
            masks=[support(x,source['kappa']) for x in pre];projection={}
            for mode,model in models.items():
                h=model.embed_out
                if mode=='native':operation=lambda x,s:torch.nn.functional.linear(x,models['native'].embed_out.packed_weight)
                else:
                    def operation(x,s,head=h):head.x_half=x;head.support=s;return head.project()
                projection[mode]=Graph(operation,[raw[0],masks[0]])
                checks=[]
                for x,s in zip(raw,masks):
                    projection[mode].stage([x,s]);checks.append(compare(torch.nn.functional.linear(x,models['native'].embed_out.packed_weight),projection[mode](),bounds))
                if not all(c['pass'] for c in checks):raise ValueError('Projection graph qualification failed: '+mode)
                component_checks['projection_only'][mode]=checks
            write(dest/'component-quality.json',component_checks)
            result['timings']['projection_only']=timed('projection-only',projection,list(zip(raw,masks)))
            result.update(status='complete',qualified=True,blocks=blocks,predictions=predictions,
                          evidence='development' if args.development else 'valid',
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=time.monotonic()-started)
            emit(dest,'complete',loss=losses,elapsed_seconds=result['elapsed_seconds'])
    except Exception as exc:
        import traceback
        result.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        emit(dest,'failed',error=str(exc));raise
    finally:write(dest/'result.json',result)


if __name__=='__main__':main()
