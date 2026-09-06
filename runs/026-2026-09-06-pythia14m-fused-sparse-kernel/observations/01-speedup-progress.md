# 01: Can exact fused kernels exceed 1.6x for Pythia-14M?

## Question and method

For the retained Run 014 `14m/a7-0p5` checkpoint, can run-local CUDA gate/consumer
fusion exceed 1.6x full-model speedup without changing the trained function?
The source is Run 025's hash-checked input manifest. Keep BF16, B=1, T=2048,
full 50,304-vocabulary logits, uncached causal SDPA and the A7-Z-POST gates fixed.
RTX 5090, Torch 2.11.0+cu128, Transformers 5.12.1. No new training or weights.

K017 fuses exact one-sided h/z gates and sparse projections. K018 combines sparse
W2/Wo, biases and BF16-rounded parallel residual additions. K019 fuses exact
partial RoPE, symmetric Q/K/V thresholds and head-major layout; the winner
composes K019 and K018. Candidate code was frozen before final timing.

## Sources and coverage

`../autoresearch/artifacts/001-...` through `019-...` contain results, manifest
hashes, event logs, complete numerical gates and raw paired samples. Use the
literal attempt names in `../autoresearch/progress.csv`; no unsuccessful mode
is omitted from `../autoresearch/all-variants.csv` (69 mode/scope rows).
`../runtime/primitive*.json` contains 146 passing GPU correctness cases.
`../runtime/diagnostics/` contains independent untimed full-validation activation,
weight, row-occupancy and profiler evidence. Checkpoint identity is retained in
each manifest; no weight/data copy is committed here.

Development: first 16 of 64 training-split blocks, cohort seed2500, five paired
passes with seed2504. Every full-model attempt also checks all 338 complete
validation blocks from all 500 documents (692,224 input tokens; 1,444 excluded
tail tokens; 691,886 next-token targets). Final timing: 64 validation identities
chosen with numpy default_rng(2504), seven paired passes, three fresh processes
015-017. Ablations 018/019 use the same identities but one process each.
Timing stages identical resident inputs outside the clock and includes complete
forward/logits plus CUDA synchronization. No gradients/cache/decode timing.

## Reduction

`../autoresearch/reduce_final.py` audits source/input/config hashes and coverage,
then computes exp(mean(log(native_ms/candidate_ms))) across 1,344 paired cells.
Latencies are pooled sample medians, not the arithmetic average of speedups.
Intervals are 10,000 percentile-bootstrap draws, seed2606, independently
resampling process identities and shared input identities while retaining all
seven repeats within each selected process/input cell. They are descriptive,
not a preregistered inferential design, and only three processes constrain
between-process uncertainty; the one-process ablation intervals are narrower
in scope. No timing outliers or failed numerical modes are silently removed.

R_model pools source integer counts: 1,748,738,568,719 / 6,363,055,915,008 =
0.27482684296304843. This canonical full-validation logical-opportunity estimand
is constant because implementations target the same checkpoint/topology. It
is neither speedup nor a newly measured BF16 per-variant rate. R_model_max's
integer all-zero analytic ceiling is retained alongside the source record.

## Result

The frozen bundle reaches **1.8149029x**, interval **[1.8098012,1.8192157]**.
Median latency falls from **2.6929291 to 1.4846455 ms**. Each fresh process passes
all 338 blocks with bitwise-identical full logits and identical loss
5.83130066301001 nat/token. Process speedups are 1.8179186, 1.8094127, 1.8173899.
The dense-projection/QKV-fusion ablation reaches 1.4573163x; sparse joint W2/Wo
alone reaches 1.1760117x against its own paired stock reference. The combined
gain is therefore not solely attributable to avoiding zero products.

Stock graph and compile modes are faster but fail sealed numerical gates.
Lazy-import hoisting, isolated compiler models, fixed cuBLAS workspace and
Inductor precision-cast emulation do not repair them. In the untimed graph
diagnostic, the first changed tap is attention context z, with some threshold
crossings. This is a localization, not a complete root-cause proof. The final
reference uses original canonical eager attention and default workspace.

## Figure caption and legend

**Figure 01 - Exact fused-kernel search and frozen confirmation.** Left: all
14 development iterations before freeze; blue circles pass complete numerical
qualification, orange crosses fail it, blue staircase is best qualified paired
speedup so far. Dense screening iterations plot the selected qualified stock
control at 1x; other screened modes remain in the CSV. Right: frozen ablations
and combined winner, with crossed-bootstrap 95% intervals. Gray solid line is
parity; dashed line is the requested 1.6x target. Canonical R_model is constant
27.4827%. Left/right y scales differ (0-4.5x and 0-2.2x) and both start at zero.
Workspace/reference-wrapper conditions changed during development, so absolute
latencies are not directly comparable across those iterations. Each ratio uses
its own paired matched-setting stock reference. The historical 1.8482x best
development value is not substituted for the frozen 1.8149x result.

Source script: `../17_plot_progress.py`; output:
`../figures/01-speedup-progress.pdf`. The single-page PDF was rendered with
Poppler and visually checked for clipping, overlap and labels.

## Limits and nonclaims

This supports the narrow requested >1.6x goal **over the qualified stock PyTorch
reference**, not over every possible custom dense baseline. The QKV-fusion
ablation is already an optimized dense-projection implementation, and the
incremental sparse advantage beyond it is smaller; it is not directly measured
by dividing unrelated absolute medians. There is no >1.6x sparse-only claim,
no universal runtime ceiling, no deployment concurrency/graph safety guarantee,
no training speedup, no cached autoregressive decoding claim, and no extension
to other checkpoints, sparsity levels, sequence lengths or GPU types.

The adapters use static inference buffers and exact trained gate contracts.
Source weight initialization/training provenance remains unchanged. Native-BF16
diagnostics are separated from canonical count provenance and no gradient
interaction can be reconstructed here. No manuscript text or approved cross-run
finding was changed. The result bears on runtime qualification of manuscript
eq:r-model-measured / eq:r-model-max, not on their logical definitions.
