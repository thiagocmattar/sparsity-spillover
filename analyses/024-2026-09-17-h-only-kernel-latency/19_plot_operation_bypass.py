"""All retained settings: actual operand zeros versus matrix-instruction bypass."""
import hashlib
import json
from importlib import import_module
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'data/operation-bypass.json'
OUTPUT = HERE/'figures/10-operation-bypass.pdf'
OPS = ['qkv_projection', 'mlp_w1', 'mlp_w2', 'attention_output_projection',
       'qk_scores', 'probability_value']
LABELS = ['QKV projection ($a$)', 'FFN-up ($m$)', 'FFN-down ($h$)',
          'Attention output ($z$)', 'QK scores', 'PV product']
STYLE_SOURCE = HERE/'14_plot_14m_quality_latency.py'
REFERENCE = HERE/'figures/08-14m-70m-quality-sparsity-latency.pdf'
STYLES = list(import_module(STYLE_SOURCE.stem).STYLES) + [
    ('4', 'none', r'$T_4/P_0$', '#82A8C4', (0, (5, 2, 1, 2))),
    ('7', 'none', r'$T_7/P_0$', '#D5A47D', (0, (5, 2, 1, 2))),
    ('1', 'h', r'$T_1/P_h$', '#455E37', (0, (3, 2))),
    ('1', 'L1', r'$T_1$/L1', '#9B8270', (0, (5, 2, 1, 2))),
]
TITLE = 'Scalar Sparsity and Matrix-Instruction Bypass'


def main():
    data = json.loads(SOURCE.read_text(encoding='utf-8'))
    rows = data['settings']
    assert len(rows) == 102
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.titlesize':11.5,
        'axes.labelsize':11, 'xtick.labelsize':10, 'ytick.labelsize':10,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.65,
        'pdf.fonttype':42, 'mathtext.fontset':'dejavusans'})
    fig, axes = plt.subplots(4, 3, figsize=(12.8, 11.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.07, right=.985, top=.93, bottom=.13, hspace=.64, wspace=.25)
    fig.suptitle(TITLE, fontsize=14, y=.985)
    panels = []
    series = []
    for i, ax in enumerate(axes.flat):
        size = '14M' if i < 6 else '70M'
        op, label = OPS[i % 6], LABELS[i % 6]
        size_rows = [r for r in rows if r['model'] == size]
        plotted = []
        for scope, pressure, recipe, color, ls in STYLES:
            group = sorted([r for r in size_rows if r['kind']=='trained'
                            and (r['scope'],r['pressure'])==(scope,pressure)],
                           key=lambda r: r['kappa'] if r['kappa'] is not None
                           else (r['local_pressure_weight'] or 0))
            if not group:
                continue
            xs = [100*r['operations'][op]['bf16_activation_zero_fraction_lower_bound'] for r in group]
            ys = [100*r['operations'][op]['bypass_fraction'] for r in group]
            control = scope in ['0','1'] and pressure=='none'
            ax.plot(xs,ys,color=color,ls=ls,lw=1.6,marker='o',ms=9 if control else 5.8,
                    mfc='white' if scope=='0' else color,mec=color if scope=='0' else 'white',
                    mew=1.5 if scope=='0' else .7,zorder=5 if control else 4)
            plotted += [{'setting_id':r['setting_id'], 'x':x, 'y':y} for r,x,y in zip(group,xs,ys)]
            series.append({'panel':chr(97+i),'model':size,'operation':op,'kind':'trained',
                           'label':recipe,'color':color,'linestyle':ls,
                           'setting_ids':[r['setting_id'] for r in group]})
        for scope, color in [('0',STYLES[0][3]),('1',STYLES[1][3])]:
            group = sorted([r for r in size_rows if r['kind']=='clipping' and r['scope']==scope],
                           key=lambda r:r['p'])
            assert [r['p'] for r in group] == [p/10 for p in range(10)]
            xs = [100*r['operations'][op]['bf16_activation_zero_fraction_lower_bound'] for r in group]
            ys = [100*r['operations'][op]['bypass_fraction'] for r in group]
            ax.plot(xs,ys,color=color,ls=':',lw=1.3,zorder=2)
            plotted += [{'setting_id':r['setting_id'], 'x':x, 'y':y} for r,x,y in zip(group,xs,ys)]
            series.append({'panel':chr(97+i),'model':size,'operation':op,'kind':'clipping',
                           'scope':scope,'color':color,'linestyle':':',
                           'setting_ids':[r['setting_id'] for r in group]})
        if op == 'qkv_projection':
            ax.text(.07,.13,'Post-hoc',transform=ax.transAxes,color='#646970',fontsize=9.5)
        if op in ['qk_scores', 'probability_value']:
            endpoint = next(r for r in size_rows if r['kind']=='trained'
                            and r['scope']=='7' and r['pressure']=='all' and r['kappa']==.5)
            point = endpoint['operations'][op]
            ax.annotate(r'$T_7/P_{\mathrm{all}}$, $\kappa=0.5$',
                (100*point['bf16_activation_zero_fraction_lower_bound'],100*point['bypass_fraction']),
                xytext=(-7,10),ha='right',textcoords='offset points',
                fontsize=9.5,color=STYLES[5][3])
        if op == 'probability_value':
            base = next(r for r in size_rows if r['kind']=='trained' and r['family']=='A0')
            baseline = 100*base['operations'][op]['bypass_fraction']
            ax.axhline(baseline,color='#92969B',ls=(0,(2,3)),lw=.8,alpha=.8,zorder=1)
            ax.text(.04,baseline+4.5,f'Base model: {baseline:.2f}%',
                    transform=ax.get_yaxis_transform(),color='#6C7177',fontsize=9.5)
        ax.set_title(f'({chr(97+i)}) {size}: {label}', loc='left', pad=11)
        ax.set(xlim=(-3,103), ylim=(-3,103), xticks=[0,25,50,75,100], yticks=[0,25,50,75,100])
        ax.grid(axis='y', color='#E8EAED', lw=.6); ax.set_axisbelow(True)
        ax.tick_params(length=3,width=.65,labelbottom=True)
        ax.set_xlabel('Scalar zero-product opportunity (%)')
        if i % 3 == 0: ax.set_ylabel('Matrix instructions bypassed (%)')
        assert len(plotted) == (60 if size=='14M' else 42)
        assert len({p['setting_id'] for p in plotted}) == len(plotted)
        panels.append({'panel':chr(97+i),'model':size,'operation':op,'points':plotted})
    handles = []
    for scope,pressure,label,color,ls in STYLES:
        control = scope in ['0','1'] and pressure=='none'
        handles.append(Line2D([],[],color=color,ls=ls,marker='o',ms=9 if control else 5.8,
                              mfc='white' if scope=='0' else color,mec=color if scope=='0' else 'white',
                              mew=1.5 if scope=='0' else .7,lw=1.6,label=label))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.025),ncol=5,
               frameon=False,fontsize=11,handlelength=2.3,columnspacing=2.3,labelspacing=.7)
    fig.savefig(OUTPUT,metadata={'Title':TITLE,'CreationDate':None,'ModDate':None}); plt.close(fig)
    evidence = {'source':SOURCE.relative_to(HERE).as_posix(), 'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                'script':Path(__file__).name, 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'figure':OUTPUT.relative_to(HERE).as_posix(), 'figure_sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                'title':TITLE, 'displayed_settings':102, 'panels':panels,'series':series,
                'style_reference':REFERENCE.relative_to(HERE).as_posix(),
                'style_reference_sha256':hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
                'style_source':STYLE_SOURCE.name,'style_source_sha256':hashlib.sha256(STYLE_SOURCE.read_bytes()).hexdigest(),
                'layout':{'rows':4,'columns':3,'width_inches':12.8,'height_inches':11.8},
                'typography_pt':{'title':14,'panel_titles':11.5,'axes':11,'ticks':10,'legend':11,'notes':9.5},
                'x':'BF16 activation-derived scalar zero-product lower bound; excludes weight zeros and P underflow',
                'y':'Raw B_op including SIMT substitution and attention padding; no baseline subtraction'}
    (HERE/'data/operation-bypass-figure.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'pdf':str(OUTPUT),'settings_per_operation':102,'panels':12}))


if __name__ == '__main__':
    main()
