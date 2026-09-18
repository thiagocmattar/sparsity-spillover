# Observations

- [024: Control clipping latency](024-controls-posthoc-final-latency.md): all 40
  dense/ReLU clipping settings at 14M/70M,120 qualified processes, verified raw
  timings and diagnostics, with a full-range final-kernel latency PDF styled
  like Figure3 and a tiling/clipping implementation note for later text.

- [023: Side-by-side 14M variant](023-14m-quality-latency-variant.md): matched
  22-checkpoint quality/sparsity and sparsity/latency panels, circular markers,
  T/P labels, wider layout, one post-hoc note and analytic ceiling guides.
  Main Figure 1 in the introduction; refreshed artwork and a rebuilt paper preview.

## Current task.md paper figures and captions

- [013: Figure 1](013-quality-sparsity-caption.md): dense-relative quality/sparsity,
  54 trained endpoints, control clipping, coverage guides, and measured nondominance.
- [014: Figure 2](014-intervention-sites-caption.md): retained architecture map,
  independent threshold/pressure scopes, candidate sites and untargeted head.
- [015: Figure 3](015-pressure-scope-caption.md): 20 matched all-minus-h pressure
  contrasts at 14M and 70M; six loss/sparsity/latency panels, blue/orange topology
  curves, uniform circles, explicit T/P references and timing-session caveats.
- [016: Figure 4](016-operation-changes-caption.md): eight integer-count operation
  decompositions, signed contributions and exact net changes at 14M.
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
