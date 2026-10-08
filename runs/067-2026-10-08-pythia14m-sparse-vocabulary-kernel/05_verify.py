"""Check immutable upload bytes and the complete, qualified paired measurement cohort."""
import argparse
from io_utils import RUN, read, verify, write, record


def verify_result(path, config, model_id, replicate):
    result=read(path/'result.json');quality=read(path/'quality.json')
    expected=config['validation']
    coverage={'documents':expected['documents'],'blocks':expected['blocks'],'input_tokens':expected['input_tokens'],
              'predictions':expected['predictions'],'excluded_tail_tokens':expected['excluded_tail_tokens']}
    if (result['status']!='complete' or result.get('qualified') is not True or result['evidence_label']!='valid'
            or result['arguments']['model']!=model_id or result['arguments']['replicate']!=replicate
            or result['runtime']['gpu']!=config['gpu'] or result['coverage']!=coverage or quality['coverage']!=coverage):
        raise ValueError('Unqualified or incomplete accepted cohort member: '+path.name)
    if quality['bounds']!=config['numerical_bounds'] or result['config']!=record(RUN/'config.json'):
        raise ValueError('Changed numerical bounds or configuration')
    for row in result['source_files']:verify(row)
    for mode in config['modes']:
        rows=quality['rows'][mode]
        if len(rows)!=338 or [r['block'] for r in rows]!=list(range(338)) or not all(r['pass'] for r in rows):
            raise ValueError('Incomplete full-logit coverage')
        if abs(quality['loss'][mode]-quality['loss']['native'])>config['numerical_bounds']['validation_loss_atol']:
            raise ValueError('Pooled loss criterion failed')
    for boundary in ('full_forward','gate_cast_projection','projection_only'):
        summary=result['timings'][boundary]['summary']
        if set(summary)!=set(config['modes']) or any(v['count']!=config['timing_inputs']*config['timing_passes'] for v in summary.values()):
            raise ValueError('Incomplete paired timing coverage')
    head=read(path/'head-graph-quality.json')
    if head['bounds']!=config['numerical_bounds'] or len(head['input_indices'])!=config['timing_inputs']:
        raise ValueError('Incomplete real-operand head qualification')
    for graphs in head['rows'].values():
        if set(graphs)!=set(config['modes']) or any(len(rows)!=config['timing_inputs'] or not all(r['pass'] for r in rows) for rows in graphs.values()):
            raise ValueError('Failed standalone head graph qualification')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inputs-only',action='store_true');parser.add_argument('--tag');args=parser.parse_args()
    rows=read(RUN/'provenance/upload-inventory.json')
    for row in rows:verify(row)
    if args.inputs_only:
        print({'status':'passed','verified_upload_files':len(rows)},flush=True);return
    if not args.tag or not args.tag.isalnum():raise ValueError('Simple cohort tag required')
    config=read(RUN/'config.json');inputs=read(RUN/'provenance/inputs.json');results=[]
    for replicate in range(1,config['process_replicates']+1):
        for model in inputs['checkpoints']:
            path=RUN/'artifacts/attempts'/f'full-{model["id"]}-{replicate}-{args.tag}'
            result=verify_result(path,config,model['id'],replicate)
            if result['source']!=model:raise ValueError('Checkpoint identity changed')
            results.append(record(path/'result.json'))
    write(RUN/f'artifacts/verification-{args.tag}.json',{'status':'passed','members':results,'count':len(results)})
    print({'status':'passed','cohort_processes':len(results)},flush=True)


if __name__=='__main__':main()
