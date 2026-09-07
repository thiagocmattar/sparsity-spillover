"""Explicit historical configuration list; no search or performance selection."""
from io_utils import RUN, archived_run, record, write


def configurations():
    rows = [{'id':'p0','candidate':'p0','original_iteration':0,'family':'sakana-adapted',
             'settings':{'sites':'active'},'status':'eligible',
             'note':'Existing signed-exact Sakana-derived Pythia adapter, topology-active linears.'}]
    excluded = {2:'Standalone attention primitive, no historical14M full-model composition.',
                7:'70M launch-geometry search.',8:'70M layer-mask search.',9:'Frozen70M policy.',
                10:'Frozen410M layer policy.',14:'70M topology search.',15:'70M site/layer search.',16:'Frozen70M policy.'}
    paper_iteration = 0
    for n in range(1,51):
        name = f'k{n:03d}'
        family = 25 if n<=16 else 26 if n<=19 else 28
        folder = archived_run(family)/('autoresearch/candidates' if family<28 else 'candidates')/name
        base = {'candidate':name,'original_iteration':n,'family':family,
                'source_directory':folder.relative_to(RUN).as_posix(),
                'source_files':[record(p) for p in sorted(folder.iterdir(),key=lambda p:p.name.lower()) if p.is_file()],
                'status':'excluded' if n in excluded else 'eligible', 'note':excluded.get(n,'')}
        if n not in excluded:
            paper_iteration += 1
        base['paper_iteration'] = paper_iteration if n not in excluded else None
        settings = {}
        if n<=16:
            settings['sites'] = 'active'
        if n==4:
            settings['strategy'] = 'hybrid'
        if n in [3,4,5]:
            base['note'] = 'Existing14M-capable implementation; first common-protocol full-model evaluation is retrospective.'
        if n==17:
            settings = {'sites':['h','z'],'elements':4,'warps':4}
        if n==19:
            base['note'] = 'Executed K019+K018 compatibility composition from Run027; original Run026 sources retained.'
        if n>=20:
            settings = {'shortcut':n<=35,'round_p':False}
        if n==44:
            base['note'] = 'Preserves actually executed K033-base composition, not the intended K036 attention-only hypothesis.'
        masks = ([f'{kind}{count}' for kind in ['prefix','suffix'] for count in range(1,6)]+['even','odd']) if n==11 else [None]
        for mask in masks:
            row = {**base,'id':name if mask is None else f'{name}-{mask}', 'settings':dict(settings)}
            if mask is not None:
                row['settings']['mask_id'] = mask
            rows.append(row)
    return rows


def main():
    rows = configurations()
    write(RUN/'provenance/candidates.json', {'configurations':rows,
          'selection':'All eligible historical14M sources; no runtime-based selection or retuning.',
          'paper_axis':'Eligible proposal ordinal; twelve K011 masks share one proposal position.',
          'unsupported_policy':'Retain exclusion or runtime failure, never replace by an invented timing.',
          'source':record(__file__)})
    print({'configurations':len(rows),'eligible':sum(r['status']=='eligible' for r in rows),
           'eligible_proposals':len({r['candidate'] for r in rows if r['status']=='eligible' and r['candidate']!='p0'})})


if __name__=='__main__':
    main()
