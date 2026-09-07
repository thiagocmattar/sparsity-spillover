"""Three single-panel, vector-only scientific figures; no rescaling of history."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from io_utils import RUN, read, write, record, verify

BLUE='#176b91';GRAY='#6d7378';RED='#bc413c'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,
    'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,
    'axes.linewidth':.65,'savefig.dpi':300})


def axes():
    fig,ax=plt.subplots(figsize=(4.8,3.15),layout='constrained')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(True,which='major',color='#dadde0',linewidth=.5,zorder=0)
    ax.set_axisbelow(True);ax.tick_params(direction='out',length=3,width=.65)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:.1f}'))
    ax.set_ylabel('Full-model speedup (×)')
    return fig,ax


def progress_figure(data):
    fig,ax=axes();curve=data['progress']
    x=[r['iteration'] for r in curve];y=[r['speedup'] for r in curve]
    ax.step(x,y,where='post',color=BLUE,lw=1.7,zorder=3)
    changes=[0]+[i for i in range(1,len(y)) if y[i]>y[i-1]]
    ax.scatter([x[i] for i in changes],[y[i] for i in changes],s=16,color=BLUE,zorder=4)
    ax.axhline(1,color=GRAY,lw=.7,ls=(0,(3,3)),zorder=1)
    high=max(y);span=max(high-1,.2)
    ax.set_ylim(1-.025*span,high+.14*span)
    ax.set_xlim(-.7,x[-1]+1.)
    ax.set_xticks([1,10,20,30,x[-1]])
    ax.set_xlabel('Kernel proposal iteration')
    ax.text(.02,.055,'Native dense = 1×',transform=ax.transAxes,fontsize=8,color=GRAY)
    ax.annotate(f'Best found: {high:.2f}×',xy=(x[-1],high),xytext=(-4,9),
                textcoords='offset points',ha='right',color=BLUE,fontsize=9)
    return fig,ax


def individual_figure(data,catalog):
    fig,ax=axes()
    mapping={r['id']:r.get('paper_iteration',0) for r in catalog if r['status']=='eligible'}
    rows=[p for p in data['points'] if p['phase']=='history' and 'speedup' in p]
    for qualified in [True,False]:
        selected=[p for p in rows if p['qualified']==qualified]
        ax.scatter([mapping[p['candidate']] for p in selected],[p['speedup'] for p in selected],
                   s=15 if qualified else 24,color=GRAY if qualified else RED,
                   marker='o' if qualified else 'x',alpha=.5 if qualified else .85,
                   linewidths=0 if qualified else .8,zorder=2 if qualified else 4)
    curve=data['progress']
    ax.step([r['iteration'] for r in curve],[r['speedup'] for r in curve],where='post',color=BLUE,lw=1.6,zorder=3)
    ax.axhline(1,color=GRAY,lw=.7,ls=(0,(3,3)))
    ys=[p['speedup'] for p in rows]+[1.]
    low,high=min(ys),max(ys);span=max(high-low,.2)
    ax.set_ylim(max(0,low-.08*span),high+.14*span)
    ax.set_xlim(-1.,curve[-1]['iteration']+1.)
    ax.set_xticks([0,10,20,30,curve[-1]['iteration']],["P0","10","20","30",str(curve[-1]['iteration'])])
    ax.set_xlabel('Kernel proposal iteration')
    ax.text(.02,.98,'Blue: best qualified, fixed checkpoint\nGray: qualified; ×: failed numerical gate',
            va='top',transform=ax.transAxes,fontsize=7.5)
    return fig,ax


def rmodel_figure(data):
    fig,ax=axes()
    rows=[p for p in data['points'] if p['phase']=='final' and p['candidate']=='k050' and 'speedup' in p]
    for qualified in [True,False]:
        selected=[p for p in rows if p['qualified']==qualified]
        ax.scatter([p['R_model'] for p in selected],[p['speedup'] for p in selected],s=24,
                   color=BLUE if qualified else RED,marker='o' if qualified else 'x',
                   alpha=.85,zorder=3,linewidths=0 if qualified else .9)
    fit=data['k050_regression']
    if fit:
        qualified=[p for p in rows if p['qualified']]
        x=[min(p['R_model'] for p in qualified),max(p['R_model'] for p in qualified)]
        ax.plot(x,[fit['intercept']+fit['slope']*v for v in x],color=BLUE,lw=1.3,zorder=2)
        ax.text(.04,.96,rf'$\widehat{{S}} = {fit["intercept"]:.3f} + {fit["slope"]:.3f}\,R_{{\mathrm{{model}}}}$'+'\n'+
                rf'$R^2 = {fit["r_squared"]:.3f}$',transform=ax.transAxes,va='top',fontsize=10)
    ys=[p['speedup'] for p in rows];span=max(max(ys)-min(ys),.2)
    ax.set_ylim(min(ys)-.09*span,max(ys)+.13*span)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel(r'Canonical $R_{\mathrm{model}}$ (fraction)')
    return fig,ax


def main():
    path=RUN/'results/matched-retrospective-001.json';data=read(path)
    for row in data['sources']:verify(row)
    catalog=read(RUN/'provenance/candidates.json')['configurations']
    folder=RUN/'figures';folder.mkdir(exist_ok=True)
    outputs=[]
    for name,builder in [('01-matched-autoresearch-progress',lambda:progress_figure(data)),
                         ('02-matched-individual-candidates',lambda:individual_figure(data,catalog)),
                         ('03-matched-rmodel-speedup',lambda:rmodel_figure(data))]:
        dest=folder/(name+'.pdf')
        if dest.exists():raise ValueError('Publication file exists; use a new revision, never overwrite executed output')
        fig,ax=builder();fig.savefig(dest,metadata={'Creator':'Run029 / Matplotlib','Title':name})
        plt.close(fig);outputs.append(record(dest))
    write(RUN/'results/figures-001.json',{'source':record(path),'script':record(__file__),
          'figures':outputs,'format':'Single axes; vector PDF; embedded TrueType; linear unnormalized speedup; major grid.'})
    print(outputs)


if __name__=='__main__':main()
