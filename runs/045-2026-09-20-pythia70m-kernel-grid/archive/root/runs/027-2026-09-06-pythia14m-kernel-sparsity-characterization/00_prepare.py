"""Pin the 35 existing trained endpoints; no model execution or new training."""
from run027_common import ROOT, RUN, R25, R26, read_json, write_json, record, verify_record


def main():
    analysis = ROOT/'analyses/013-2026-09-04-matched-intervention-manuscript/figure_data.json'
    evidence = read_json(analysis)
    selected = [r for r in evidence['trained'] if r['scale']=='14M']
    if len(selected)!=35: raise ValueError('Expected the complete 35-endpoint study')
    rows = []
    for index, row in enumerate(selected,1):
        attempt = ROOT/row['source']
        manifest = read_json(attempt/'manifest.json')
        if manifest['status']!='completed' or manifest['completed_steps']!=712:
            raise ValueError('Incomplete source endpoint')
        checkpoint = attempt/manifest['checkpoints']['final']['path']
        logical = read_json(attempt/'diagnostics/logical_products.json')
        count = logical['measured']
        cov = logical['coverage']
        if (cov['sequences'],cov['input_tokens'],cov['excluded_tail_tokens'])!=(338,692224,1444):
            raise ValueError('Incomplete source logical coverage')
        if count['block_zero_product_count']/count['model_product_count']!=row['R_model']:
            raise ValueError('Canonical count discrepancy')
        for filename in ['diagnostics/logical_products.json','diagnostics/activation_statistics.json']:
            path=attempt/filename
            if record(path)['sha256']!=evidence['sources'][path.relative_to(ROOT).as_posix()]:
                raise ValueError('Analysis source identity changed')
        files=[record(checkpoint/name) for name in ['config.json','model.safetensors','checkpoint_metadata.json','generation_config.json']]
        rows.append({'id':f'c{index:02d}', 'family':row['family'],'dose':row['dose'],
            'historical':row['family']=='A4+OL1@h','source':row['source'],
            'checkpoint':checkpoint.relative_to(ROOT).as_posix(),
            'topology':read_json(checkpoint/'config.json'), 'source_condition':manifest['condition'],
            'files':files,'provenance':[record(attempt/f) for f in ['manifest.json','config.yaml','diagnostics/logical_products.json']],
            'canonical_logical_products':logical})
    old=read_json(R25/'prelaunch/input_manifest.json')
    for name in ['development','validation','validation_metadata']: verify_record(old[name])
    write_json(RUN/'prelaunch/inputs.json',{'cohort_source':record(analysis),'checkpoints':rows,
        'inputs':{name:old[name] for name in ['development','validation','validation_metadata']},
        'kernel_sources':[record(R26/f'autoresearch/candidates/{k}/{f}') for k in ['k018','k019'] for f in ['candidate.py','kernel.cu']],
        'config':record(RUN/'config.json')})
    print(f'Pinned {len(rows)} checkpoints; {sum(f["bytes"] for r in rows for f in r["files"]):,} model bytes')


if __name__=='__main__': main()
