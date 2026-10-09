"""Tables and observations for the fixed, retained Analysis043 evidence."""
import csv
import json
from pathlib import Path
from evidence import HERE


def text_file(name, text):
    path=HERE/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text.strip()+'\n',encoding='utf-8')


def csv_file(name, rows):
    path=HERE/name
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def table(headers, rows):
    def line(row):return '| '+' | '.join(str(x).replace('|','\\|') for x in row)+' |'
    return '\n'.join([line(headers),line(['---']*len(headers))]+[line(row) for row in rows])


def dose(row):
    value=row.get('block_kappa')
    if value is None and row.get('dose_name')=='lambda':return f"λ={row['dose']:g}"
    return '—' if value is None else f'{value:g}'


def recipe(row):
    if row['group']=='base':return 'Base'
    pressure={'h':'Ph','all':'Pall','none':'P0','h+head':'Ph+head'}[row['pressure']]
    if row['group'].startswith('head'):return f"T2 + {row['head_gate']} / {pressure}"
    if row['scope']=='1':return f'ReLU h / {pressure}'
    scope='2' if row['scope']=='hz' else row['scope']
    return f"T{scope} {'tile' if row['group'].startswith('tile') else 'scalar'} / {pressure}"


def point_rows(points, edge):
    return [[r['model'],recipe(r),dose(r), '—' if r['head_kappa'] is None else f"{r['head_kappa']:g}",
             r['execution'],f"{r['loss']:.6f}",f"{r['latency_ms']:.6f}",r['session'],
             'yes' if r['id'] in edge else 'no'] for r in points]


def head_label(row):
    return f"Head{row['head_tile_tokens']} / {'Ph' if row['pressure']=='h' else 'Ph+head'} / h,z κ={row['block_kappa']:g} / head κ={row['head_kappa']:g}"


def tables(latency, geometry):
    points=latency['points'];edge=latency['global_frontier_ids']
    flat=[]
    for row in points:
        values=row.get('process_latency_ms',[])
        flat.append({k:row.get(k,'') for k in ['id','model','group','scope','pressure','gate_rule',
            'block_kappa','head_gate','head_kappa','execution','loss','latency_ms','session',
            'execution_cohort','latency_precision','quality_precision','source_attempt','source_coordinate_file']}
            |dict(global_nondominated=row['id'] in edge,
                  cohort_nondominated=row['id'] in latency['compatible_frontier_ids'][row['execution_cohort']],
                  process_min_ms=min(values) if values else '',process_max_ms=max(values) if values else '',
                  dose_name=row.get('dose_name',''),dose=row.get('dose','')))
    csv_file('results/all-executions.csv',flat)
    csv_file('results/global-frontier.csv',[r for r in flat if r['global_nondominated']])
    headers=['Model','Training recipe','Body gate κ or pressure λ','Head κ','Execution','Loss','Full host ms','Session','14M frontier']
    text_file('results/ALL-POINTS.md',"""# All retained execution coordinates

87 retained 14M executions, including native and skip-disabled controls, not 87 trained models.
Loss is the canonical final-checkpoint FP16 full-validation result; latency is
complete B1/T2048/full-logits forward on RTX 5090, using the declared timing precision.
All modes for a checkpoint share its quality coordinate. A native or no-skip mode
keeps the trained gates. The CSV records source paths, precision, gate shape,
execution cohort, process ranges where available, and stable point IDs.
λ appears only for the historical pressure-weight sweep; it is not a gate threshold.
T1 retains its historical ReLU intervention, rather than a κ threshold sweep.

"""+'\n\n'.join(f"## {cohort}\n\n"+table(headers,point_rows([r for r in points if r['execution_cohort']==cohort],edge))
                         for cohort in latency['compatible_frontier_ids'])+
        '\n\n[Complete machine-readable coordinates](all-executions.csv) · [Source identities](../provenance/latency-sources.json).')
    head_rows=[]
    for r in latency['head_boundaries']:
        for boundary,timing in r['timings'].items():
            counts=r['head_counts']
            for mode,value in timing['host_ms']['geomean_ms'].items():
                samples=[p['timings'][boundary]['host_ms']['geomean_ms'][mode] for p in r['processes']]
                head_rows.append(dict(condition=r['condition'],label=head_label(r),boundary=boundary,mode=mode,
                    canonical_loss=r['loss'],host_ms=value,cuda_ms=timing['cuda_ms']['geomean_ms'][mode],
                    process_min_ms=min(samples),process_max_ms=max(samples),
                    gate_empty_count=counts['empty_gate_tiles'],gate_tile_count=counts['gate_tiles'],
                    consumer_empty_count=counts['empty_consumer_tiles'],consumer_tile_count=counts['consumer_tiles'],
                    skipped_mma=counts['skipped_mma'],potential_mma=counts['potential_mma'],source=r['source_coordinate_file']))
    csv_file('results/head-boundaries.csv',head_rows)
    decomposed=[]
    d=latency['decomposition']
    for c in d['conditions']:
        for mode,r in c['modes'].items():
            decomposed.append(dict(model=c['model'],kappa=c['kappa'],mode=mode,
                block_threshold='corner 8x16' if c['tiled_threshold'] else 'scalar',head_threshold='off',
                **{k:v/1000 for k,v in r['semantic_categories'].items()},
                summed_activity_ms=sum(r['semantic_categories'].values())/1000,
                active_union_ms=r['active_us']/1000,activity_span_ms=r['span_us']/1000,
                activity_gap_ms=r['gap_us']/1000,independent_host_geomean_ms=r['host']['geomean']/1000))
    csv_file('results/semantic-decomposition.csv',decomposed)
    cat=d['semantic_category_order']
    text_file('results/DECOMPOSITION.md',"""# Measured full-model decomposition

Run064 only: complete CUPTI profiles for two retained 14M checkpoints. Categories
below are milliseconds of arithmetic mean CUDA activity durations across 12 traces
per condition/mode. They keep Analysis042's categories, colors, and mapping.
The total is a sum of activities, not the independently timed host geometric mean.
The CSV also preserves the GPU activity union, span, and gaps. Small differences
between summed activities and the union reflect the original timestamp reductions.

Both checkpoints have trained 8×16 corner thresholds at h/z.
Every vocabulary head here has its threshold OFF and dense arithmetic. The H/Z
output block groups gate, consumer/preparation, down projections and residual work;
fused 14M internals have no independent substep timing.

"""+table(['Model','κ','Mode']+cat+['Activity sum','Host geometric mean'],
        [[r['model'],r['kappa'],r['mode']]+[f'{r[k]:.6f}' for k in cat]+
         [f"{r['summed_activity_ms']:.6f}",f"{r['independent_host_geomean_ms']:.6f}"]
         for r in decomposed if r['mode'] in ('native','tile_sparse','sparse')])+
        """

Full decompositions for the newer head interventions: **N/A**. Runs067/068 retain
projection-only, gate+cast+projection, and complete-forward timings, measured as
separate overlapping boundaries. Their independent geometric means are never
subtracted or stacked to estimate unmeasured stages.

[All 10 execution decompositions](semantic-decomposition.csv) ·
[Original stage mapping and fusion limits](../../042-2026-10-08-model-latency-decomposition/LABELS.md) ·
[Original detailed stage table](../../042-2026-10-08-model-latency-decomposition/results/forward-stages.csv).
""")
    geometries=[]
    for r in geometry['conditions']+geometry['manuscript_context']:
        out=dict(id=r['id'],label=r['label'],family=r['family'],role=r['role'],block_kappa=r['block_kappa'],
                 head_kappa=r['head_kappa'],steps=r['steps'],zero_lr_boundaries=r['zero_lr_boundaries'],
                 eligible_tensors='/'.join(map(str,r['summary']['eligible_parameter_tensors'])),source=r['source_attempt'])
        for metric,values in r['summary'].items():
            if isinstance(values,dict):out.update({metric+'_'+k:v for k,v in values.items()})
        geometries.append(out)
    csv_file('results/pressure-geometry.csv',geometries)


def report(latency, geometry):
    points=latency['points'];edge=latency['global_frontier_ids']
    rows=latency['head_boundaries'];full=[];component=[];occupancy=[]
    for r in rows:
        t=r['timings']['full_forward']['host_ms'];v=t['geomean_ms'];n=r['head_counts']
        full.append([head_label(r),f"{r['loss']:.6f}"]+[f'{v[m]:.6f}' for m in ('dense','custom_no_skip','custom_sparse')]+
                    [f"{100*(1/t['dense_over_sparse']-1):+.2f}%",f"{100*(1-1/t['no_skip_over_sparse']):.2f}%"])
        component.append([head_label(r)]+[f"{r['timings'][b]['host_ms']['geomean_ms'][m]:.6f}"
            for b in ('projection_only','gate_cast_projection') for m in ('dense','custom_no_skip','custom_sparse')])
        occupancy.append([head_label(r),f"{100*n['empty_gate_tiles']/n['gate_tiles']:.3f}%",
            f"{100*n['empty_consumer_tiles']/n['consumer_tiles']:.3f}%",f"{100*n['skipped_mma']/n['potential_mma']:.3f}%"])
    geo=[]
    for r in geometry['conditions']:
        s=r['summary']
        geo.append([r['label']]+[f"{s[k]['median']:.3f}" for k in
            ['rho_task_percent','removed_pressure_percent','pre_projection_pressure_norm','pre_projection_norm_ratio','pre_cap_ratio']]+
            [f"{s[k]['percent']:.2f}%" for k in ['raw_gradient_conflict','adaptive_conflict']]+
            [f"{s['cap_active']['count']}/712",','.join(map(str,s['eligible_parameter_tensors']))])
    nondominated=[r for r in points if r['id'] in edge]
    nondominated.sort(key=lambda r:(r['loss'],r['latency_ms']))
    text_file('results/REPORT.md',"""# 14M tile/head quality–latency and pressure geometry

The 14M-only inventory contains **87 execution points** and **12 observed
nondominated points**. None of the 18 head execution points extends this observed
14M envelope. The specialized 14M T2 corner-tile model at κ=0.1 does appear on
it: loss **5.596746**, complete-forward **0.486441 ms** (Run064). This is a descriptive
comparison of retained measurements across sessions and implementations.

The head-aligned κ=0.5 kernel is **3.10% faster** than its own native dense execution,
but its loss is **5.882600**. Within the FP16 native-body/head cohort, the Head16 κ=0.1
native dense execution has both lower loss (5.560725) and lower point-estimate
latency (0.983987 ms) than Head16 κ=0.5 sparse (0.990146 ms). More bypass therefore
does not establish a better measured quality–latency trade-off in this inventory.

## Scope and comparability

At the user's request, the cohort selects only the 14M coordinates from the
previous inventory. It includes Analysis037's 14M historical study, Analysis041's
T2/T7 tile experiments and matched scalar references, and the executions in
Runs063/064/067/068. No quality or latency coordinate changes. It is a cohort of existing
evidence, rather than an exhaustive inventory of every repository experiment.
Different executions of the same retained checkpoint remain separate points.

All quality coordinates use canonical final-checkpoint full validation: 500
MiniPile documents, 338 complete 2,048-token blocks, 692,224 input tokens and
691,886 shifted predictions; the 1,444-token tail is excluded. Timing is RTX 5090,
B1/T2048 with all 50,304 vocabulary logits for every token. It is full-sequence
forward, not cached decoding or token sampling. Loss evaluation/softmax is outside
the vocabulary-projection timing. Native B1 qualification losses and BF16 kernel
anchor losses are not substituted for canonical FP16 quality coordinates.

Older optimized block executions use BF16. Runs067/068 use FP16, native dense
transformer projections, and a separately optimized head. No measurement combines
that head kernel with the earlier sparse block pipeline. There is no latency
rescaling, normalization to Base, extrapolated combined speedup, or gate-off
ablation. A checkpoint's trained gates remain active in native and no-skip modes.

Figures [01](../figures/01-global-quality-latency.pdf) and
[02](../figures/02-execution-cohort-frontiers.pdf) show both the global envelope and
the separate execution cohorts. There are 69 BF16 block executions (12 frontier
points) and 18 FP16 head executions (3 frontier points, all native dense). Session
differences remain within those cohorts. Frontier membership uses strict Pareto
dominance of point estimates, not significance tests; process ranges are not CIs.
The dotted step envelope does not assert measured intermediate executions.
Figure01 uses one 14M-only panel with linear axes and connected historical T2/Ph
and T7/Ph sweeps, following Analysis041 Figure04 panel (b). The latency range
expands to include the new head modes; none are cropped from view.

## What is tiled and what runs sparsely?

| Evidence | Trained body gate | Trained head gate | Measured sparse execution |
| --- | --- | --- | --- |
| Historical scalar T2 / Ph | h,z, elementwise | OFF | 14M h/z consumer |
| Historical scalar T4 / T7 | Declared 4 / 7 sites, elementwise | OFF | Site-specific kernel package; a gate alone does not imply bypass |
| T2 corner models, Runs060/061/063/064 | h,z, 8×16 corner tiles | OFF | Generic or corner-aware h/z consumer |
| T7 corner models, Run062 | Declared seven sites, 8×16 corner tiles | OFF | Retained T7 consumer package |
| Run066 training / Run067 timing | h,z, 8×16 tiles, κ=0/0.1 | 8×16 tiles, same κ | Head only; consumer M16/K16 requires two neighboring token tiles empty |
| Run068 | h,z, 8×16 tiles, κ=0 | 16×16 tiles, κ=0.1/0.5 | Head only; gate and consumer tiles aligned |

Tiles mean **tokens × input features**, not vocabulary-output columns. The trained
head gate acts on `head_in.final`, after learned affine final LayerNorm and before
the vocabulary matmul. It retains an entire tile iff its signed upper-left corner
is greater than κ; rejected tiles are zero, retained signed interiors are unchanged.
There is no gate or pressure on vocabulary weights or output logits. Ph means OL1
at post-gate h; Ph+head also includes the final post-gate head-input activation.

## New head points: full latency and quality

All times are synchronized host geometric means in milliseconds, pooled over
three fresh processes/checkpoint and 64 timing inputs × 7 paired passes/process.
The sparse-versus-dense column is a latency change: a positive value is slower.
Skip savings use the same custom kernel with skipping disabled as denominator.
Canonical loss is measured in the original training evaluator (B32), whereas B1
kernel qualification compares all complete validation blocks. All six head
checkpoints' custom logits and B1 loss match native exactly in their source runs.

"""+table(['Condition','Canonical loss','Dense ms','No-skip ms','Sparse ms','Sparse vs dense','Skip saves vs no-skip'],full)+
        """

Projection and head-path timings are separately measured boundaries, not additive
components of the independently timed full forward:

"""+table(['Condition','Projection dense','Projection no-skip','Projection sparse','Head path dense','Head path no-skip','Head path sparse'],component)+
        """

The head path includes the gate, cast, and vocabulary projection. Full CUPTI stage
decomposition is **unavailable** for these six checkpoints. Figure
[04](../figures/04-head-measured-boundaries.pdf) presents the overlapping boundaries
separately. Figure [03](../figures/03-existing-full-decomposition.pdf) and the
[decomposition table](DECOMPOSITION.md) retain the complete, consistently colored
profiles from Run064. Their vocabulary heads have threshold OFF.

Pooled B1 occupancy/instruction counts over the same complete validation workload:

"""+table(['Condition','Empty gate tiles','Empty M16/K16 consumer tiles','MMA bypass'],occupancy)+
        """

MMA bypass is omitted matrix instructions divided by potential instructions, not
a speedup. For the head8 models, two adjacent token tiles must both be empty before
an M16 consumer tile can skip. Alignment removes that mismatch in Run068. Native
training B32 occupancy can differ slightly from the benchmark's B1 count; these
rows deliberately use benchmark pooled integers throughout.

## Orthogonalization resistance: global Ph versus Ph+head

Figure [05](../figures/05-orthogonalization-head-comparison.pdf) shows all eight
retained tile/head training conditions. Figure
[06](../figures/06-manuscript-style-geometry.pdf) adds the original manuscript's
20 14M T4/T7 Ph/Pall conditions in its two-panel geometry view. Every condition
uses all 712 optimizer boundaries, including one initial zero-learning-rate
boundary. λ=b=1 and ε=10⁻¹². The eight recent conditions share original random
initialization and data schedule; one training seed per condition is available.

Let u be the task Adam direction and w the pressure gradient preconditioned with
the same task second moment, pooled over eligible parameters. Let
α=max(0,−u·w)/(‖u‖²+ε) and w_safe=w+αu in the conflicting case.

| Quantity | Exact meaning |
| --- | --- |
| Manuscript opposing component, 100ρopp | 100α, removed component norm relative to task-direction norm |
| Percent of pressure component removed | 100α‖u‖/‖w‖; norm fraction, not pressure energy or coordinates |
| Raw-gradient conflict | Frequency of g_task·g_pressure < 0 |
| Adaptive-direction conflict | Frequency of u·w < 0 with ‖u‖²>ε; equals projection_applied |
| Pressure norm before projection | ‖w‖, absolute adaptive L2 norm |
| Pressure/task norm before projection | ‖w‖/(‖u‖+ε) |
| Manuscript pre-cap ratio r/b | λ‖w_safe‖/(‖u‖+ε)/b, after projection and before norm capping |
| Cap active | Actual trust_scale < 1 |

100ρopp is not bounded by 100%; it has the task norm as denominator. The extra
pressure-normalized fraction is a different metric. The manuscript's r/b is also
different from the pre-projection norm ratio requested here; both are reported.
The table gives step medians for continuous quantities and pooled boundary counts
for conflict/cap frequency. Quartiles and full traces are retained in the data.

        """+table(['Condition','ρopp %','Removed pressure %','‖w‖ before','‖w‖/‖u‖ before','r/b after','Raw conflict','Adaptive conflict','Cap active','Eligible tensors'],geo)+
        """

In the matched head8 κ=0.1 pair, adding head-input pressure changes median opposing
component from **2.227% to 8.503%** of task-direction norm, removed pressure component
from **9.377% to 23.625%**, and raw-gradient conflict from **65.03% to 85.39%**.
Adaptive projection is already frequent in both (**99.72% / 99.30%**), so conflict
frequency alone hides the larger opposing component. At κ=0, the matched changes
are **4.680%→6.222%** of task norm and **63.34%→81.74%** raw conflict.

Higher measured opposition does not imply worse quality in these pairs: Ph+head
has lower canonical loss than Ph by **0.031284 / 0.040847** at the same head8
κ=0 / 0.1. The global geometry is a diagnostic, rather than a standalone
predictor of the resulting quality or inference speed.

These recent conditions activate the norm cap in only **1–3 of 712 boundaries**,
including the initial zero-LR boundary. They do not reproduce the strong cap
saturation of the manuscript's T7/Pall context (3,453/3,560 boundaries, **96.99%**
pooled over its five thresholds). Pressure can face more conflict without reaching
the cap. Geometry at a zero-LR boundary does not imply a realized parameter move.

This is **global pressure geometry**, as confirmed by the user. There is no head-only
pressure run. Ph averages six h tensors with weight 1/6 each; Ph+head averages
those six plus one final head-input tensor with weight 1/7 each. The eligible pool
also expands from 65 to 75 parameter tensors, because head-input pressure reaches
more upstream parameters; vocabulary output weights are excluded. Consequently
these results cannot isolate the head's local gradient, assign its resistance to
a specific layer, or claim that only one site caused the difference. Comparison
of head8 with head16 additionally changes alignment and, at head κ=0.1, body κ.

## Observed 14M frontier

Sorted by increasing canonical loss; all native/skip-disabled executions were
eligible. Ties remain nondominated unless another point is strictly better in at
least one objective. A cheaper training intervention is not inferred from this
inference-only frontier.

        """+table(['Model','Training recipe','Body gate κ or pressure λ','Head κ','Execution','Loss','Full host ms','Session','14M frontier'],point_rows(nondominated,edge))+
        """

[All 87 14M points](ALL-POINTS.md) · [All-execution CSV](all-executions.csv) ·
[Frontier CSV](global-frontier.csv) · [Head-boundary CSV](head-boundaries.csv) ·
[Geometry CSV](pressure-geometry.csv) · [Analysis scope and reproduction](../README.md).

This is a descriptive analysis of retained results. No new GPU run, manuscript
change, or research-finding promotion was performed.
""")


def observations():
    items=[
      ('01-global-quality-latency','Does vocabulary-head intervention extend the observed 14M quality–latency envelope?',
       '03_plot_frontiers.py','latency','87 existing 14M execution points; all measured modes retained.',
       'Canonical final validation loss versus synchronized complete-forward host geometric mean. Pareto dominance minimizes both, with at least one strict inequality.',
       'One 14M-only detail panel with linear axes, following Analysis041 Figure04 panel (b). Historical five-threshold T2/Ph is dashed blue; T7/Ph is solid orange. Base has a vertical loss guide. Green and purple mark T2/T7 body tiles with head OFF; pink/orange mark head8/head16. Open markers mean native dense arithmetic with the trained gates; × is no-skip. The dotted step envelope is the observed Pareto set. Whiskers show process min/max, not confidence intervals. The larger latency range includes every new head execution.',
       '12 14M nondominated points; none of the 18 head execution points extends the envelope. Run064 T2 tile κ=0.1 is nondominated at loss 5.596746 and 0.486441 ms.',
       'Mixed sessions, BF16 optimized body kernels versus FP16 native-body/head kernels. No combined body/head timing exists. Connected sweep segments show threshold order, not interpolation; the descriptive step envelope does not assert measured intermediate executions. The detail axes do not start at zero.'),
      ('02-execution-cohort-frontiers','Does the global conclusion survive separating the execution packages?',
       '03_plot_frontiers.py','latency','14M only: 69 BF16 block executions and 18 FP16 native-body/head executions.',
       'Recompute strict Pareto dominance independently inside each declared cohort using original measured coordinates; no latency rescaling.',
       'Two explicitly labeled detail panels have their own axes because latency ranges differ. Colors, symbols, native/no-skip encoding and process-range whiskers match Figure01. Each panel has its own dotted point-estimate envelope.',
       '12 BF16 frontier points, 3 FP16 head frontier points. All 3 head-cohort frontier points use native dense arithmetic. Head16 κ=0.5 sparse is point-estimate dominated by Head16 κ=0.1 dense.',
       'Timing sessions and checkpoint weights still differ inside the cohorts. This separation does not create a paired gate-off ablation or statistical frontier.'),
      ('03-existing-full-decomposition','Where is 14M complete-forward CUDA work spent under the existing semantic labels?',
       '04_plot_latency.py','latency','Run064, 2 retained 14M checkpoints, κ=0/0.1, native and selected sparse modes; 12 CUPTI traces per condition/mode.',
       'Preserve Analysis042 original arithmetic CUDA activity means; group into nine fixed categories. Stack activity durations without rescaling. Independent complete-host geometric means appear as text.',
       'One 14M panel. Input setup gray; block normalization sky blue; QKV blue; RoPE/context orange; attention yellow; MLP up green; H/Z output pink; final norm black; vocabulary projection vermillion. Patterns reinforce colors. Both checkpoints have h/z thresholds in 8×16 tiles. Every head threshold is OFF.',
       'Vocabulary projection remains dense and prominent in the optimized 14M profiles. The H/Z aggregate preserves the measured fused consumer without inventing internal timings.',
       'Bars are summed activities, not a timeline or host latency. Fusion prevents independent timing of several H/Z internals. New head experiments have no full CUPTI stage profile; do not infer it by subtracting independent boundary means.'),
      ('04-head-measured-boundaries','How does each head intervention change projection, head-path, and complete latency?',
       '04_plot_latency.py','latency','Runs067/068, 6 checkpoints × 3 modes × 3 timing boundaries; 3 processes/checkpoint, 64 inputs × 7 paired timing passes/process.',
       'Use source synchronized host geometric means unchanged. Error bars are process min/max, not confidence intervals. Gate/consumer occupancy fractions pool full-validation integer counts.',
       'Projection-only, gate+cast+projection, and complete-forward panels use separate labeled zero-based scales. Blue native dense, orange dotted custom no-skip, green hatched custom sparse. Every tick declares head tile size, Ph/Ph+head, body κ and head κ. Body projections remain native dense FP16.',
       'Head16 κ=0.5 sparse saves 15.55% on the head path and 3.10% on full forward versus native dense; κ=0.1 changes full forward by +1.03%. Head8 full sparse execution is 7.19–8.86% slower than dense despite bypass savings.',
       'Boundaries overlap and are measured separately; never stack or subtract them. No full head-model CUPTI decomposition or head-gate-off ablation is retained. Absolute process variability is visible for Run068, and the matched ratios are retained in data.'),
      ('05-orthogonalization-head-comparison','Does adding head-input pressure increase global orthogonalization resistance relative to Ph?',
       '05_plot_geometry.py','geometry','8 retained 14M conditions from Runs060/066/068, 712 optimizer boundaries each (5,696 records).',
       'Recompute per-boundary geometry from raw training events and audit the original adaptive dot/norm/cap identities. Display medians and IQRs for norms/ratios, integer counts/712 for conflict. Include the initial zero-LR boundary.',
       'Six panels: manuscript removed component/task norm; removed component/pressure norm; raw/adaptive conflict; absolute pressure norm before projection; pressure/task norm before projection; post-projection pre-cap r/b. Ph blue, Ph+head vermillion; no-head circle, head8 diamond, head16 star. Error bars are step quartiles. The dotted r/b=1 line marks the cap.',
       'Matched head8 κ=0.1: removed/task norm 2.227→8.503%, removed/pressure norm 9.377→23.625%, raw conflict 65.03→85.39%; adaptive conflict stays approximately 99%. Cap is active only 1–3/712 boundaries in these recent conditions.',
       'Global pooled eligible-parameter geometry, not isolated head resistance. Tensor averaging changes 1/6→1/7 and eligible pool 65→75. No head-only pressure condition. One seed; IQRs describe training evolution rather than seed uncertainty. Norm fraction is not energy or coordinate count.'),
      ('06-manuscript-style-geometry','How do the newer tile/head conditions compare with the manuscript OL1 geometry view?',
       '05_plot_geometry.py','geometry','20 original manuscript 14M conditions plus 8 retained tile/head conditions: 19,936 optimizer-boundary records, 712 each.',
       'Audit raw logs and reproduce original manuscript medians. Left: condition median/IQR of 100ρopp on categorical κ positions. Right: unsmoothed pointwise median r/b over available thresholds in each family, with faint individual traces.',
       'Dashed context families are T4/Ph, T4/Pall, T7/Ph, T7/Pall (5 thresholds each); solid newer families are T2 tile/Ph, head8/Ph, head8/Ph+head, head16/Ph+head (2 each). Both metric axes are logarithmic. Head16 varies head κ at body κ=0; head8 varies both. r/b=1 marks the norm cap.',
       'New head-pressure recipes have larger removed components than the matched Ph head8 recipes, but rarely saturate the cap. Original T7/Pall cap activity is 3,453/3,560 boundaries (96.99%) across five κ conditions.',
       'Family medians pool different numbers/choices of thresholds and are descriptive; head16/body-threshold axes are not a matched common sweep. Manuscript r/b is after projection, not the pre-projection norm ratio. Global geometry and one-seed limitations remain; manuscript text is unchanged.')]
    for stem,question,script,kind,coverage,method,caption,result,caveat in items:
        source=('Analysis037/041 retained 14M cohort coordinates; Run063 process timing samples; Analysis042 Run064 14M decomposition and original checkpoint metadata; Runs066/068 canonical validation and Runs067/068 verified timing results. Plot style reference: Analysis041 Figure04 panel (b) and its 02_quality_latency_frontier.py script.' if kind=='latency' else
                'Runs060/066/068 raw events.jsonl and manifests; Analysis024 original OL1 appendix and its 20 14M source conditions; Analysis021 retained geometry auditor. Manuscript draft/05-experimental-study.tex and draft/08-appendix-interventions.tex define the contextual metrics.')
        text_file(f'observations/{stem}.md',f"""# {stem[3:].replace('-',' ').capitalize()}

Question: {question}

Sources: {source} Exact identities are in [{kind} source receipts](../provenance/{kind}-sources.json); reduced values and source attempts are in [{kind}.json](../data/{kind}.json).

Coverage: {coverage} Validation coordinates/counts use all 500 MiniPile documents, 338 complete 2,048-token blocks, 691,886 predictions and the declared 1,444-token excluded tail. Geometry observations use training boundaries rather than validation batches.

Method: {method}

Caption and legend: {caption}

Observed result: {result}

Caveats: {caveat}

Source script: [{script}](../{script}), with [{('01_collect_latency.py' if kind=='latency' else '02_collect_geometry.py')}](../{('01_collect_latency.py' if kind=='latency' else '02_collect_geometry.py')}) collecting the evidence. Output: [{stem}.pdf](../figures/{stem}.pdf). Tables: [report](../results/REPORT.md). No manuscript edit or promoted claim.
""")
    text_file('observations/INDEX.md','# Observation index\n\n'+
        '\n'.join(f'- [{stem[3:].replace("-"," ").capitalize()}]({stem}.md): {question}' for stem,question,*_ in items))


def main():
    latency=json.loads((HERE/'data/latency.json').read_text(encoding='utf-8'))
    geometry=json.loads((HERE/'data/geometry.json').read_text(encoding='utf-8'))
    tables(latency,geometry);report(latency,geometry);observations()
    print({'status':'reports-written','executions':len(latency['points']),'observations':6},flush=True)


if __name__=='__main__':main()
