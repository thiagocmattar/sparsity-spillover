# Figure 2: Thresholding and pressure act on independently selected activation sites

PDF: [02-intervention-sites.pdf](../figures/02-intervention-sites.pdf).
Manuscript placement: [Methodology](../../../manuscript/draft/methodology.tex),
`sec:interventions` and `fig:pythia-block`; recipe assignments belong in
[the intervention table](../../../manuscript/draft/experimental-study.tex), `tab:interventions`.

## Question, method, and coverage

Where do the named interventions act? This is an architectural diagram, with no
new measurements. It adapts the manuscript's existing Pythia block map, preserving
parallel attention/FFN branches, separate LayerNorms, and the residual sum.
The nonlinearity output h is immediately before W2. The q/k candidates are after
RoPE. The probability tensor p is styled as a noncandidate operator/result.

## Publication caption

**Thresholding and pressure act on independently selected activation sites.**
One shared Pythia/GPT-NeoX block is shown. The projection-input sites a,m,h,z
feed QKV projection, FFN-up, FFN-down, and attention-output projection, respectively.
The internal attention sites q and k occur after RoPE, and v is the value tensor;
p denotes attention probabilities and is not an intervention site in these recipes.
The threshold sets are T4={a,m,h,z} and T7=T4∪{q,k,v}. Colored candidate locations
do not imply that pressure is applied there: the recipe table separately specifies
no pressure, P_h={h}, or P_all=T_i. Site counts mean site types per block, repeated
across layers, rather than the number of layers or total individual interventions.
The language-model head is untargeted and omitted from this block diagram.

## Result and proposed manuscript writing

> We specify threshold and pressure scopes independently. T4 and T7 identify
> the sites at which trained gates act; P0, P_h, and P_all identify pressure
> targets. The nonnegative gate operates at a,m,h,z, while the signed-magnitude
> gate operates at q,k,v. Comparing T7/P_h against T4/P_h therefore changes
> threshold placement while holding pressure on h. Comparing T_i/P_all against
> T_i/P_h instead broadens pressure at fixed threshold placement.

## Caveats and provenance

This schematic defines locations; it is not an estimate of intervention reach or
evidence that every site is pressured. The operational hook placement is defined
by [METHODS.md](../../../research/METHODS.md).
Source: [paper-architecture.tex](../paper-architecture.tex), adapted from
[the existing manuscript diagram](../../../manuscript/draft/figures/pythia-architecture-map.tex),
compiled by [13_rebuild_paper_figures.py](../13_rebuild_paper_figures.py).
The proposed paragraph is not inserted into the manuscript by this task.
