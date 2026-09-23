# Bounded search: no candidate passed promotion

Question: did the approved eight initial plus eight follow-up configurations
produce a qualified replacement for the current moderate-kappa70M policy?

Method and coverage: all16 normal configurations and their skip-disabled
counterparts passed the operator suite (both512 and2048 input features; zero,
dense, sparse, threshold equality/adjacency, long rows, disjoint support and
changed-input graph replay). Both screens covered all128 retained training
blocks at both kappas, all12 h/z sites and five paired timing passes. Every
screened component passed numerical qualification. The fixed promotion rule
requires5% lower complete-path host latency at both kappas in both64-block splits.

Result: zero promoted sites. No full-model candidate was frozen or evaluated as
a final candidate. The original user goal remains unmet: no new kernel has
established both endpoint gains and a larger positive Delta70 than Delta14 with
controlled sparse execution attribution. The matched reference result remains
Delta70=14.671us versus Delta14=14.753us, with a cross-size difference interval
that includes zero. Neither stage establishes broad h/z execution gains.

The follow-up `f_cta_v2` improved consistently at four sites, but its weakest
cell fell below the5% margin: h.3=1.97%, h.4=4.12%, z.1=3.63%, z.2=.55%.
Thus this result must not be described as "every new kernel is slower" or
"T2/Ph lacks useful sparse structure." The per-row compaction variant is a
substantial implementation improvement over the initial warp variant. Its
small local gains are not a demonstrated full-model speedup.

For z.1, the complete-path savings were1.10--1.42us at .05 and1.56--2.00us at
.1 across the two training splits. This is a possible threshold-responsive
mechanism, but not yet an accepted candidate or a full-model Delta result.
Other small-gain sites do not show the same ordering: h.4 saves more at .05.
Adding component savings is not a valid forecast of total latency. CUDA-event
measurements agree with the direction of these four sites' small gains.

The compiled follow-up binary's resource reports show no local-memory spills
or stack allocation. SASS retains64/128-bit global-load instructions for the
vectorized variants. These static facts neither measure cache/DRAM traffic nor
prove that weight bandwidth is the remaining bottleneck. The logical counters
report requested work separately.

Decision: stop candidate development at the approved16-configuration bound;
do not lower the5% promotion margin after seeing the results. Complete the
agreed full-validation diagnostics for the unchanged reference paths. The
diagnostic-only reference policy is explicitly not a final candidate selection.
Keep the GPU available for the user's guidance within the approved lease, with
the hard stop unchanged. No manuscript text or consolidated finding is changed.

Potential next design, requiring a new decision: measure whether the small,
consistent per-row-compaction gains survive full-model integration under an
explicitly revised promotion rule, or test producer-emitted support metadata
to remove a separate support scan. The former would investigate untested
integration, and the latter would change the producer/consumer implementation;
neither is an achieved result of this run. Any continuation must preserve both
endpoints, full numerical qualification, strong dense controls and skip-off
attribution, without manufacturing Delta through a .05 regression.

Sources: `03_operator_checks.py`, `04_screen.py`, `05_select.py`,
`followup_sparse.cu`, `primitives_v3.py`, the two retained screen attempts,
`results/component-search-summary.json` and `results/component-search-table.md`.
The complete262-member screen archive passed local SHA256 and size verification.
No figure was generated.
