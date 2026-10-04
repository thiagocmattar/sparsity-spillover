"""Append qualified T7 tile endpoints; preserve previous frontier coordinates."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator,NullLocator,ScalarFormatter

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
RUN=ROOT/'runs/062-2026-10-04-pythia14m-t7-corner-tile-ph/latency'
COLORS={'hz':'#173FAD','7':'#C27539'}
T2='#008A79'; T7='#8C62A8'; MARKERS={'14M':'o','31M':'^','70M':'s'}
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def identity(path):
    with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=digest)
def dominates(a,b):
    return a['loss']<=b['loss'] and a['latency_ms']<=b['latency_ms'] and (a['loss']<b['loss'] or a['latency_ms']<b['latency_ms'])
def frontier(points): return sorted([p for p in points if not any(dominates(q,p) for q in points)],key=lambda p:p['loss'])
def collect():
    previous_path=HERE/'data/quality-latency-frontier.json'; previous=read(previous_path)
    verification=read(HERE/'results/frontier-verification.json')
    assert verification['status']=='verified' and identity(previous_path)==verification['data']
    quality_path=HERE/'results/t7-training.json'; quality=read(quality_path)
    for row in quality['sources']: assert identity(ROOT/row['path'])==row
    timing_path=RUN/'results/results.json'; timing=read(timing_path); assert timing['status']=='verified'
    sources=[identity(previous_path),identity(quality_path),identity(timing_path),identity(Path(__file__))]
    new=[]; controls=[]
    for pair in timing['pairs']:
        assert pair['qualified'] and pair['paired_samples_per_comparison']==1344 and len(pair['processes'])==3
        for row in pair['source_files']:
            path=RUN/row['path']; actual=identity(path)
            assert actual['bytes']==row['bytes'] and actual['sha256']==row['sha256']; sources.append(actual)
        for family in ['scalar','tile']:
            q=next(r for r in quality['rows'] if r['family']==family and r['kappa']==pair['kappa'])
            processes=[]
            for process in pair['processes']:
                path=RUN/'artifacts/attempts'/process['attempt']; result=read(path/'result.json')
                assert result['qualified'] and result['status']=='complete'
                checkpoint=result['checkpoints'][family]
                assert checkpoint['dose']==q['kappa'] and checkpoint['original_checkpoint'].startswith(q['attempt']+'/checkpoints/')
                assert checkpoint['final_checkpoint_content_sha256']==q['final_checkpoint_content_sha256']
                values=[s['host_ms'] for s in read(path/'timing.json')['samples'] if s['mode']==family+'_sparse']
                assert len(values)==448; processes.append(math.exp(statistics.mean(math.log(x) for x in values)))
            latency=pair['host_geomean_ms'][family+'_sparse']
            assert math.isclose(latency,math.exp(statistics.mean(math.log(x) for x in processes)),rel_tol=1e-12)
            p=dict(model='14M',scope='7',pressure='h',kappa=q['kappa'],loss=q['loss'],latency_ms=latency,
                process_latency_ms=processes,backend='Run062 sparse',gate=family,condition=q['condition'],
                source_attempt=q['attempt'],final_checkpoint_content_sha256=q['final_checkpoint_content_sha256'])
            (new if family=='tile' else controls).append(p)
    assert len(new)==len(controls)==2
    historical=previous['historical_points']; t2=previous['new_tile_points']; points=historical+t2+new
    edge=frontier(points)
    return dict(historical_points=historical,t2_tile_points=t2,t7_tile_points=new,matched_t7_scalar_references=controls,
        measured_frontier=edge,t7_dominance=[dict(condition=p['condition'],nondominated=p in edge,
          previous_dominators=[q for q in historical+t2 if dominates(q,p)]) for p in new],
        paired_speedup=[dict(kappa=p['kappa'],tile_over_scalar_sparse=p['speedup']['tile_over_scalar_sparse']) for p in timing['pairs']],
        source_files=sources,coverage=quality['coverage'],frontier_cohort_points=len(points),auxiliary_scalar_references=2,
        limits='Original orange T7/Ph and prior T2 coordinates are from separate sessions. Run062 adds gate passes included in timing. Same-session controls establish paired runtime ratios; historical Pareto membership is descriptive. One seed; whiskers are process ranges, not confidence intervals.')

def draw(ax,data,scales,detail=False):
    old=[p for p in data['historical_points'] if p['model'] in scales]
    for model in scales:
        for scope in ['hz','7']:
            points=sorted([p for p in old if p['model']==model and p['scope']==scope],key=lambda p:p['kappa'])
            ax.plot([p['loss'] for p in points],[p['latency_ms'] for p in points],color=COLORS[scope],
              ls='--' if scope=='hz' else '-',marker=MARKERS[model],ms=4.5,lw=1.3,mec='white',mew=.5)
        for p in [p for p in old if p['model']==model and p['scope']=='0']:
            kernel=p['backend']=='kernel'
            ax.scatter(p['loss'],p['latency_ms'],marker=MARKERS[model],s=48 if kernel else 22,
              facecolor='white' if kernel else '#52565C',edgecolor='#52565C',zorder=6)
    edge=frontier(old+data['t2_tile_points']+data['t7_tile_points'])
    ax.step([p['loss'] for p in edge],[p['latency_ms'] for p in edge],where='post',color='#656C73',ls=':',lw=1,zorder=2)
    for label,key,color,marker in [('T2','t2_tile_points',T2,'D'),('T7','t7_tile_points',T7,'P')]:
        for p in data[key]:
            ax.errorbar(p['loss'],p['latency_ms'],yerr=[[p['latency_ms']-min(p['process_latency_ms'])],
              [max(p['process_latency_ms'])-p['latency_ms']]],fmt=marker,color=color,ms=8,mec='white',mew=.6,capsize=3,zorder=8)
            if detail:
                offset=(9,10 if p['kappa']==0 else -18)
                ax.annotate(f"{label} tile k={p['kappa']:g}",(p['loss'],p['latency_ms']),xytext=offset,
                  textcoords='offset points',fontsize=8,color=color,bbox=dict(facecolor='white',edgecolor='none',pad=.4))
    if detail:
        for p in data['matched_t7_scalar_references']:
            ax.scatter(p['loss'],p['latency_ms'],marker='P',s=50,facecolor='white',edgecolor=T7,zorder=7)
    all_points=old+data['t2_tile_points']+data['t7_tile_points']+(data['matched_t7_scalar_references'] if detail else [])
    x=[p['loss'] for p in all_points]; y=[p['latency_ms'] for p in all_points]
    xm=max(.08,.10*(max(x)-min(x))); ym=max(.025,.12*(max(y)-min(y)))
    ax.set(xlabel='Validation loss (canonical FP16; lower is better)',ylabel='Full-model latency (ms; lower is better)',
      xlim=(min(x)-xm,max(x)+xm),ylim=(max(.01,min(y)-ym),max(y)+ym))
    ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)

def main():
    data=collect(); preserved={p.name:identity(p)['sha256'] for p in (HERE/'figures').glob('*.pdf') if not p.name.startswith('07-')}
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.7)); fig.subplots_adjust(left=.065,right=.985,bottom=.25,top=.88,wspace=.29)
    draw(axes[0],data,['14M','31M','70M']); draw(axes[1],data,['14M'],detail=True)
    axes[0].set(xscale='log',yscale='log')
    for axis,bounds in [(axes[0].xaxis,axes[0].get_xlim()),(axes[0].yaxis,axes[0].get_ylim())]:
        axis.set_ticks([v for v in MaxNLocator(nbins=6).tick_values(*bounds) if bounds[0]<=v<=bounds[1]])
        axis.set_major_formatter(ScalarFormatter()); axis.set_minor_locator(NullLocator())
    axes[0].set_title('(a) Previous manuscript cohort + four tile endpoints',loc='left',fontsize=10)
    axes[1].set_title('(b) 14M detail and matched T7 scalar references',loc='left',fontsize=10)
    fig.suptitle('T7/Ph corner tiles against the orange quality-latency frontier',fontsize=13)
    handles=[Line2D([],[],color=COLORS['hz'],ls='--',label='Previous scalar T2/Ph'),
      Line2D([],[],color=COLORS['7'],label='Previous scalar T7/Ph (orange)'),
      Line2D([],[],color=T2,marker='D',ls='none',label='T2/Ph tiles (Run061)'),
      Line2D([],[],color=T7,marker='P',ls='none',label='T7/Ph tiles (Run062)'),
      Line2D([],[],color=T7,marker='P',mfc='white',ls='none',label='Matched T7 scalar (reference)'),
      Line2D([],[],color='#656C73',ls=':',label='Observed Pareto frontier')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.09),ncol=3,frameon=False,fontsize=8.5)
    fig.text(.52,.035,'40 frontier points: 36 previous + 2 T2 tiles + 2 T7 tiles. Hollow references excluded.\n'
      'Process-range whiskers; one seed. Historical sessions/kernel variants differ; no coordinate normalization.',ha='center',fontsize=8,color='#555B61')
    output=HERE/'figures/07-t7-quality-latency-frontier.pdf'; fig.savefig(output,metadata={'CreationDate':None,'ModDate':None}); plt.close(fig)
    assert all(identity(HERE/'figures'/n)['sha256']==h for n,h in preserved.items())
    data['output']=identity(output); data['preserved_figures']=preserved
    (HERE/'data/t7-quality-latency-frontier.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    lines=['# T7 corner endpoints against the orange frontier','',
      '| Endpoint | kappa | Canonical FP16 loss | Sparse latency (ms) | Process range (ms) | Nondominated? |','|---|---:|---:|---:|---|---|']
    for p,d in zip(data['t7_tile_points'],data['t7_dominance'],strict=True):
        lines.append(f"| T7/Ph tile | {p['kappa']:g} | {p['loss']:.6f} | {p['latency_ms']:.6f} | {min(p['process_latency_ms']):.6f}–{max(p['process_latency_ms']):.6f} | {'yes' if d['nondominated'] else 'no'} |")
    lines+=['','| kappa | Same-session scalar/tile sparse latency ratio |','|---:|---:|']
    lines += [f"| {p['kappa']:g} | {p['tile_over_scalar_sparse']:.6f}x |" for p in data['paired_speedup']]
    lines+=['','Ratios >1 mean tiles are faster.',data['limits']]
    (HERE/'results/t7-quality-latency-frontier.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'frontier_members':len(data['measured_frontier']),'t7_dominance':data['t7_dominance'],'paired_speedup':data['paired_speedup']},indent=2))

if __name__=='__main__': main()
