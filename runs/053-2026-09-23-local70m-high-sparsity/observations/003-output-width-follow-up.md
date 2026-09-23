# Wider output tiles pass the component shortlist

Question: was repeated consumer work from narrow output tiles suppressing the
wide-union algorithm's benefit at moderate thresholds?

Method: retain the feature-union algorithm and test row groups32/64 with output
tiles128/256. Same three checkpoints, two32-block training prefixes split16/16,
five paired passes,20 complete invocations per graph, dense/prior controls,
skip-disabled variants, numerical bounds and joint5% promotion rule as
observation002. The same inputs are reused for adaptive development; this is
not an independent holdout. No figure generated.

Result: all96 operator checks,32 changed-input graph checks and13,824 real-input
numerical comparisons pass. No compiler spills are reported. The stage takes
301.8seconds. All2,334 returned files /84,786,981bytes pass local SHA256 and size
verification, including a complete compiler-cache snapshot.

The32-row,256-output union configuration clears the component rule at eight
sites: h.2/h.3/h.4 and z.1/z.2/z.3/z.4/z.5. Worst ratios over both moderate
kappas and both prefixes are0.924/0.911/0.919 at those h sites and
0.618/0.681/0.830/0.871/0.804 at those z sites. All eight also beat the best
measured control by at least5% at high kappa in both prefixes. At z.1, the
moderate candidate takes17.6--18.2us; its matched no-skip variant adds
16.1--17.9us. These are per-component graph throughput timings, not model
latencies or statistical promotion.

For a32-row group, increasing output width64 to256 reduces consumer CTAs
from512 to128 on the measured2048x512 output, avoiding repeated activation
and index reads across output tiles. The follow-up establishes that layout
matters for this family on the local GPU. It does not separately measure
memory traffic, quantify each bottleneck, or prove the same gain on RTX5090.

A fixed full-model candidate is selected before validation. At the eight
qualified sites use union_m32_n256. Elsewhere retain the best measured prior/
dense choice pooled over the moderate training prefixes. This gives old union
paths at h.1/h.5 and dense paths at h.0/z.0. The same policy applies at every
kappa. The dense comparator selects among the three tested dense choices;
the prior comparator uses the strongest tested dense/prior choices without
the new kernels. Neither is claimed optimal over all possible dense kernels.
The all-sparse-skip-disabled comparator preserves gates, metadata producers,
layout and weights. Selection and provenance are in `provenance/local-policy.json`.

Next: smoke then full-validation model comparison, preserving complete338-block
coverage, all500 documents,1444 excluded tail tokens, profiles, raw timings,
activation counts/moments, weight norms, occupancy and independently checked
work counts. A single local process per condition cannot establish the final
cross-size delta or a statistically replicated manuscript speedup.

Infrastructure note: the older fetcher's bidirectional tar file-list pipe
stalled on the larger cache inventory. Only that transfer process was stopped.
`14_fetch_large.py` passes the five short directory names as arguments and
drains stdout immediately. Retrieval then passed all hashes. An early reducer
call before retrieval finished found no local operator result and wrote no
summary; the successful reduction followed verified retrieval. No GPU
measurements or scientific inputs were changed.

Sources: scripts10--14, `wider-output-config.json`,
`results/wider-outputs-001-summary.json`, `results/wider-outputs-001-table.md`,
`results/wider-outputs-001-retrieval.json`, and the derivation/source manifests.
