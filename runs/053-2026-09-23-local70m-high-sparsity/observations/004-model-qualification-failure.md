# Full-model gains remain unqualified at kappa .1

Question: do the component gains survive full-model execution, numerical
qualification and comparisons with strong tested controls?

Method: freeze one training-selected policy across Base and T2/Ph .5/.05/.1.
Compare six CUDA-graph implementations,64 validation timing inputs, seven
randomized paired passes and one fresh process per checkpoint. Check all500
validation documents /338 complete2048-token blocks, excluding1444 tail tokens.
Profiles and counted diagnostics are separate from timing. No figure generated.

Result: all four smokes pass. Base/.5/.05 qualify on full validation. At .1,
only the new candidate fails: input78 violates the per-logit bound, despite
relativeL2 .00056935 and pooled loss delta -0.0000107655. The agreed bounds
remain unchanged. The failed candidate's timing is retained but cannot count
as a qualified speedup. Its five controls pass.

The descriptive host table is in `results/model-001-table.md`. Native Base
takes6.0552ms; candidate .05 takes4.6271ms and qualifies. Candidate .1 takes
4.2282ms but does not qualify. Relative to the previous locally selected
sparse/dense policy, the candidate's measured improvement is only about0.62%
at .05 and1.46% at .1. These one-process differences have no process-replicated
significance claim. The raw .05-to-.1 drop also mixes checkpoint effects and
between-process/device drift; the dense comparator changes substantially.
There is no matched local14M delta, and no manuscript target is established.

The complete .1 numerical trace reproduces three elementwise-bound violations.
Layer0 agrees. The first difference is one output of the layer1 z projection:
reference0.032958984375 versus candidate0.033203125, on identical inputs.
Its joint residual output differs at one element by0.00048828125. Layer2 then
has one z gate-membership change; subsequent layers have additional h/z gate
changes, and final logits differ by up to0.5. This traces how a tiny projection
rounding discrepancy is amplified by hard thresholds. It does not prove the
specific internal floating-point instruction that produced the first mismatch.

Full candidate exact/near-zero counts, moments/RMS, weight norms, row/group
occupancy and independently checked executed reduction tiles are retained for
all338 blocks at all three kappas. At .1, scripts18--19 collect this inventory
and separate profiles after the qualification stop, explicitly as diagnostics
of the unchanged failed candidate. The diagnostic completes in17.8seconds;
all12 files /2,063,330bytes are retrieved and hash-verified. The original model
attempt's153 files /39,710,437bytes are likewise verified.

Next bounded variation: apply128/256-output tiles to the aligned K16 tile-skip
family with16/32-row groups. It retains feature positions and skips wholly
empty reduction tiles rather than compacting individual features. This may
avoid changes in reduction grouping, but numerical equivalence is a hypothesis
to test. It may also lose speed where feature unions occupy most aligned tiles.
The original bounds, moderate-endpoint promotion rule, controls and training
coverage remain fixed. Do not use the failing validation input as a dispatch
special case or relax the qualification gate.

Sources: scripts15--20; `provenance/local-policy.json`;
`results/model-001-summary.json`, `results/model-001-retrieval.json`,
`results/diagnosis-001-retrieval.json`, and source-hashed numerical traces.
Local laptop results remain separate from RTX5090 evidence; F003 and the
manuscript are unchanged.
