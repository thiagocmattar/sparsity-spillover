# Observations

- [031: Main/appendix quality split](031-quality-main-appendix.md):
  all 14M/70M recipes directly after Table 1, followed by base-model speedup;
  complete 410M panel in the appendix. Exact partition of all 74 trained
  and 60 control-clipping records, with the original figures retained.

- [030: Base-model speedup with clipping](030-base-speedup-clipping.md):
  a new two-panel 14M/70M scatter plot using A3's optimized A0 references,
  Figure 08's 44 trained checkpoints and style, and all 40 measured control
  clipping evaluations. Every ratio uses the same reference within its size.
  Adopted unchanged; now manuscript Figure 4 after the 14M/70M quality overview.

- [029: High-threshold bypass summary](029-operation-bypass-summary.md):
  two size panels, T4/Pall versus T7/Pall at kappa=0.5, and 24 count-based bars.
  Retained as manuscript Figure 6 in the merged sparsity/runtime explanation;
  identifies projection benefits and the unresolved attention implementation limit.

- [028: All-model quality-sparsity](028-all-model-quality-sparsity.md): all 74
  trained paper conditions in a 14M/70M/410M three-panel overview, uniform
  final loss, sixty fixed-control clipping settings and ceiling guides.
  Adopted as manuscript Figure 3 with the rewritten Section 4.1; documents
  the common 712-step budget and limits of the cross-size interpretation.

- [027: Operation bypass](027-operation-bypass.md): all 102 evaluation settings,
  six operation families, per-setting pooled MMA bypass, scalar opportunity and
  matched 35-setting 14M skip ablations. QK/PV can bypass substantial work without
  a timing benefit; twelve size-specific panels follow Figure 08's style, with
  an explanatory paragraph on h/z specialization and runtime attribution.

- [026: Operation grid](026-operation-grid.md): six absolute-contribution panels
  with 14M/70M rows and κ=0,.05,.5 columns; 24 matched Ph/Pall checkpoint stacks,
  separate row scales, a common palette and a concise contribution axis label.

- [025: Combined quality and latency](025-matched-quality-latency.md): Figure 6
  extended to 14M/70M rows, with 44 trained checkpoints, 40 measured control
  clipping latencies, one shared legend and explicit quality-view/session limits.
  Adopted unchanged as manuscript Figure 1, with updated caption and results scope.

- [024: Control clipping latency](024-controls-posthoc-final-latency.md): all 40
  dense/ReLU clipping settings at 14M/70M,120 qualified processes, verified raw
  timings and diagnostics, with a full-range final-kernel latency PDF styled
  like Figure3 and a tiling/clipping implementation note for later text.

- [023: Side-by-side 14M variant](023-14m-quality-latency-variant.md): matched
  22-checkpoint quality/sparsity and sparsity/latency panels, circular markers,
  T/P labels, wider layout, one post-hoc note and analytic ceiling guides.
  Previous manuscript Figure 1; superseded there by the two-size Figure 8.

## Current task.md paper figures and captions

- [013: Figure 1](013-quality-sparsity-caption.md): dense-relative quality/sparsity,
  54 trained endpoints, control clipping, coverage guides, and measured nondominance.
- [014: Figure 2](014-intervention-sites-caption.md): retained architecture map,
  independent threshold/pressure scopes, candidate sites and untargeted head.
- [015: Figure 3](015-pressure-scope-caption.md): 20 matched all-minus-h pressure
  contrasts at 14M and 70M; six loss/sparsity/latency panels, blue/orange topology
  curves, uniform circles, explicit T/P references and timing-session caveats.
  Adopted as manuscript Figure 4 with a lean Section 4.2 analysis; width later
  reduced to 90% (11.52 inches), retaining the 5.8-inch height and font sizes.
- [016: Figure 4](016-operation-changes-caption.md): twelve absolute operation
  stacks at 14M, T/P recipe labels and the manuscript operation palette; each
  count-based stack reconciles to model-wide sparsity.
  Adopted as manuscript Figure 5 in the merged Section 4.3 on sparsity and runtime.
- [017: Figure 5](017-quality-latency-caption.md): qualified sparsity/latency and
  quality/latency panels, finite measured frontier and session limitations.
- [018: A1](018-complete-quality-caption.md): all 340 retained post-hoc points
  and 54 trained endpoints, full-range absolute loss.
- [019: A2](019-pressure-none-caption.md): 20 explicit 14M pressure-versus-none
  contrasts, without pooling thresholds into boxplots.
- [020: A3](020-dense-speedup-caption.md): speedup relative to each size's optimized
  A0 implementation, with exact reference/session definitions.
- [021: A4](021-instruction-caption.md): the historical 30-checkpoint 14M
  instruction/ablation cohort, without new regression claims.
- [022: Table 2](022-threshold-table-caption.md): seven-minus-four-site effects
  at fixed h-only pressure, preserving all five thresholds and both sizes.

Each current observation includes the PDF/table, publication caption, associated
proposed manuscript writing, limitations, and generating source. See also
[the caption index](../CAPTIONS.md).

## Historical observations (PDFs archived)

- [001](001-sparsity-speedup.md): final K050 speedup, five new A7 h-only points
  and 35 retained historical points, including A4 h-only.
- [002](002-absolute-latency.md): paired native and K050 absolute latency;
  clarifies the checkpoint-specific speedup denominator and session boundary.
- [003](003-final-latency-topology.md): final K050 latency only, 36 checkpoints
  in four topology groups; single-site naive L1 excluded, subtle recipe labels.
- [004](004-70m-final-latency-topology.md): qualified70M shape port, all22 available
  checkpoints and four pressure-family curves; matched Run033 measurement protocol.
- [005](005-paired-topology-effects.md): distributions of five matched A7-minus-A4
  contrasts per pressure recipe, with loss, sparsity and latency panels.
- [006](006-paired-pressure-effects.md): OL1(all) minus OL1(h), with five
  matched contrasts per topology and the same three distribution panels.
- [007](007-pressure-vs-none-effects.md): 2-by-3 distributions of all-site and
  h-only OL1 minus no pressure, with four-/seven-site boxes in each panel.
- [008](008-combined-final-latency.md): 14M and 70M sparsity versus latency in
  one two-panel figure, retaining all 36/22 points from Figures03/04.
- [009](009-matched-combined-latency.md): one shared panel with 22 matching
  conditions per model; baseline, ReLU and A4/A7 OL1(h)/OL1(all) grids only,
  with separate linear-y and log-y versions.
- [010](010-a0-normalized-speedup.md): 42 points normalized to each model size's
  final-kernel A0 latency; 1-site controls excluded, linear axes and size labels.
- [011](011-matched-quality-sparsity.md): validation loss versus model-wide
  sparsity for 44 matched 14M/70M checkpoints, including both 1-site controls.
- [012](012-quality-sparsity-clipping.md): Figure11's cohort plus four complete
  A0/A1-H post-hoc clipping frontiers, showing all 40 clipping measurements.
