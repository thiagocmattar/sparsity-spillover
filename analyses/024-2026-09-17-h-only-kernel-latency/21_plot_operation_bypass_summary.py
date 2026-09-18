"""Two-size T4/T7 high-threshold instruction-bypass figure for the manuscript."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE/'data/operation-bypass.json'
OUTPUT = HERE/'figures/12-operation-bypass-summary.pdf'
OPS = ['qkv_projection', 'mlp_w1', 'mlp_w2', 'attention_output_projection',
       'qk_scores', 'probability_value']
LABELS = ['QKV\n($a$)', 'FFN\nup ($m$)', 'FFN\ndown ($h$)', 'Attn.\nout ($z$)', 'QK', 'PV']
RECIPES = [('4', r'$T_4/P_{\mathrm{all}}$', '#3679AD'),
           ('7', r'$T_7/P_{\mathrm{all}}$', '#C27539')]
TITLE = r'Matrix-Instruction Bypass at $\kappa=0.5$'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_settings(data):
    """Keep the four explicitly requested checkpoints; recompute bars from counts."""
    selected = []
    for size in ['14M', '70M']:
        for scope, label, color in RECIPES:
            matches = [r for r in data['settings'] if r['kind']=='trained'
                       and (r['model'],r['scope'],r['pressure'],r['kappa'])==(size,scope,'all',.5)]
            assert len(matches)==1, (size,scope)
            row = matches[0]
            operations = {}
            for op in OPS:
                value = row['operations'][op]
                issued, bypassed = value['issued_mmas'],value['bypassed_mmas']
                assert all(type(v) is int and v>=0 for v in [issued,bypassed])
                assert issued+bypassed==value['potential_mmas']>0
                fraction = bypassed/(issued+bypassed)
                assert fraction==value['bypass_fraction']
                operations[op] = {k:value[k] for k in [
                    'issued_mmas','bypassed_mmas','potential_mmas','simt_products',
                    'counter_method','base_model_bypass_fraction']}
                operations[op]['bypass_percent'] = 100*fraction
            selected.append({k:row[k] for k in ['setting_id','model','scope','pressure','kappa',
                             'checkpoint_key','weight_sha256','kernel','diagnostic_source']} | {
                                 'label':label,'color':color,'operations':operations})
    return selected


def main():
    data = json.loads(SOURCE.read_text(encoding='utf-8'))
    selected = selected_settings(data)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':11.5,
        'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
        'pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
    fig,axes = plt.subplots(1,2,figsize=(9.3,3.65),sharey=True)
    fig.subplots_adjust(left=.08,right=.985,top=.80,bottom=.28,wspace=.17)
    fig.suptitle(TITLE,fontsize=14,y=.975)
    width = .36
    for i,(ax,size) in enumerate(zip(axes,['14M','70M'])):
        for j,(scope,label,color) in enumerate(RECIPES):
            row = next(r for r in selected if (r['model'],r['scope'])==(size,scope))
            ax.bar([x+(j-.5)*width for x in range(len(OPS))],
                   [row['operations'][op]['bypass_percent'] for op in OPS],
                   width=width,color=color,label=label,zorder=3)
        ax.set_title(f'({chr(97+i)}) {size}',loc='left',pad=11)
        ax.set(xticks=range(len(OPS)),xticklabels=LABELS,ylim=(0,105),
               yticks=[0,25,50,75,100],xlim=(-.6,5.6))
        ax.tick_params(length=3,width=.65,labelleft=True)
        ax.grid(axis='y',color='#E8EAED',lw=.6); ax.set_axisbelow(True)
    axes[0].set_ylabel('Matrix instructions bypassed (%)')
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.025),ncol=2,
               frameon=False,fontsize=11,handlelength=2.3,columnspacing=3)
    fig.savefig(OUTPUT,metadata={'Title':'Matrix-Instruction Bypass at kappa=0.5',
                                'CreationDate':None,'ModDate':None})
    plt.close(fig)
    diagnostics = [ROOT/r['diagnostic_source'] for r in selected]
    for path in diagnostics:
        assert sha(path)==data['sources_sha256'][path.relative_to(ROOT).as_posix()]
    result = {
        'title':'Matrix-Instruction Bypass at kappa=0.5',
        'selection':{'models':['14M','70M'],'scopes':['4','7'],'pressure':'all','kappa':.5},
        'metric':data['metric'],'coverage':{k:v for k,v in data['coverage'].items()
                                        if k not in ['trained_14m','trained_70m','clipping']},
        'operation_order':OPS,'operation_labels':LABELS,'settings':selected,
        'bars':24,'figure':OUTPUT.relative_to(HERE).as_posix(),'figure_sha256':sha(OUTPUT),
        'script':Path(__file__).name,'script_sha256':sha(Path(__file__)),
        'sources_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in [SOURCE,*diagnostics]},
        'detailed_figure':'figures/10-operation-bypass.pdf',
        'style_reference':'figures/08-14m-70m-quality-sparsity-latency.pdf',
        'limits':[
            'Per-operation instruction bypass, not a fraction of operator time or model-wide savings.',
            'h/z bypass includes replacement with scalar work; SIMT product counts are retained.',
            'Attention counts include causal/padded work; PV baseline bypass is not intervention-induced.',
            'The figure contains no latency measurements; matched attention skip ablations are available only at 14M.',
            'Four selected checkpoints, one training seed; no uncertainty across seeds is estimated.']}
    (HERE/'data/operation-bypass-summary.json').write_text(
        json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'pdf':str(OUTPUT),'checkpoints':4,'bars':24}))


if __name__=='__main__':
    main()
