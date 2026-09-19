# Proposed short test: extend h/z load avoidance to a/m

Status: **awaiting design confirmation**, 19 September 2026. No numbered
run, experiment implementation, GPU execution or billable resource has been
created for this proposal. Launch approval will follow implementation and
prelaunch verification, as required by the repository's AGENTS.md.

## Question and manuscript relevance

Does directly adapting the final h/z sparse projection strategy to a and m
improve full-model latency for the same 14M T7/Pall kappa=0.5 checkpoint used
in Run037? This tests the specific untested alternative raised while reviewing
`manuscript/draft/kernel-autoresearch.tex`, rather than assuming the selected
post-load a/m checks are optimal.

The earlier K004 implementation tested pre-load detection at 16x32 granularity;
K021/K028 tested scalar nonzero traversal. Neither is the final K049 8x16
hybrid port proposed here. The new comparison changes implementation only.
No new training, threshold, pressure or checkpoint selection is proposed.

## Fixed scientific contract

- Retained Run029 c30 / Run037 checkpoint: random-initialized Pythia-14M,
  six layers, width128, FFN512, vocabulary50304, initialization seed1234,
  optimizer step712, 1493172224 training input tokens. Preserve the exact
  archived AdamW configuration, training-data order and checkpoint provenance;
  there is no optimizer, pressure update or backward pass in this test.
- Weight SHA256:
  `f83aef36ddbddbd94efb915d574c4645c687206bb68f51886ffc6e979985b425`.
- T7/Pall, kappa=0.5: G-plus at a,m,h,z; G-pm at post-RoPE q,k and at v.
  Training pressure was orthogonal L1 on all seven sites, lambda=b=1.
  Every execution mode retains the same gates, threshold, weights and biases.
- BF16, batch1, length2048, uncached causal inference, full50304-logit output,
  CUDA graphs, one physical RTX5090 without concurrent scientific workloads.
  Match Run037's Python3.12 / PyTorch2.11.0 / Transformers5.12.1 / CUDA12.8
  environment and record its exact dependency lock and source hashes.
- Runtime seed2801; timing seed2504, exact same64 validation identities,
  seven paired passes and three fresh processes per mode. Randomize the
  complete process order deterministically and retain that order.
- All500 MiniPile validation documents:338 complete2048-token blocks,
  692224 input tokens,691886 prediction tokens,1444 excluded tail tokens.
  Check every candidate against the unchanged eager reference and native
  graph. Preserve logit atol0.25, rtol0.02, relative-L2 bound0.02 and pooled
  loss absolute-difference bound0.001. Report actual errors, including whether
  bitwise equality is attained; never relax bounds after seeing results.

## Port and five comparisons

Adapt K049's load-avoiding hybrid to separate linear operations with K128 and
output widths N384 (a -> QKV) and N512 (m -> FFN-up). Retain eight real rows,
K16 support masks, ascending K traversal, scalar execution for eligible rows
with at most two nonzeros, and native-order matrix instructions for remaining
rows. Empty groups are identified before requesting their weights. The matrix
fallback retains the original padding from eight to sixteen rows; account for
that explicitly rather than treating its instruction potential as unchanged.

Only the output-width indexing and standalone linear epilogue are adapted.
Consume the already gated a/m values from the existing fused LayerNorm path;
do not apply another gate. Preserve BF16 bias/output rounding. Do not fuse
the two new a/m operations together. Keep h/z fusion, attention, residuals,
LayerNorm and the vocabulary head unchanged. This is one bounded port, with
no tile sweep, extra fusion search or checkpoint-specific dispatch.

| Mode | a projection | m projection | Purpose |
|---|---|---|---|
| frozen | Original K050 | Original K050 | Same-session reference |
| port-a | New hybrid | Original K050 | Isolate extending a |
| port-m | Original K050 | New hybrid | Isolate extending m |
| port-am | New hybrid | New hybrid | Joint extension |
| port-am-dense | New layout, sparse paths disabled | New layout, sparse paths disabled | Separate net sparse-path benefit from changed layout |

For the dense control, compile out support scanning, zero skipping and
short-row replacement, while retaining the port's layout, padding, arithmetic
and output rounding. Its comparison with port-am measures the net benefit of
the port's sparse machinery. Comparisons with frozen answer whether the port
actually improves the selected implementation. Do not combine these ratios
with historical-session absolute latencies or call the dense control A0.

## Staged checks, diagnostics and outcome rules

1. Focused CPU tests plus the full bootstrap suite before launch. Direct GPU
   checks cover both output widths: all-zero, one/two-nonzero rows, dense,
   mixed, tile-boundary, signed-zero and unsafe-range cases. Test sparse and
   dense port outputs against the original projection, including real gated
   a/m tensors from the fixed development training inputs. No new performance
   tuning is selected from validation timing.
2. Bounded full-model smoke: eight correctness blocks, four timing inputs,
   two passes per mode. These outcomes are diagnostic only, not paper evidence.
3. If direct and smoke checks pass, evaluate five modes x three fresh
   processes, with full338-block correctness and the existing64-input timing
   protocol. A failed mode remains a reported failure; no speedup from it is
   qualified. Stop dependent candidates if a shared implementation bug appears.
4. Untimed full-validation diagnostics retain a/m exact and near-zero counts
   at0,0.001,0.01; activation RMS; row nonzero histograms; 8x16 and16x16 empty
   tile counts; issued/bypassed matrix instructions; scalar replacement work;
   and source-level avoided weight-load requests. Pool integer counts before
   division. Counters must agree with an independent operand-derived oracle.
   Load-request counts are not measured DRAM traffic or a time decomposition.
5. Retain raw paired host/device timings, per-process means and ranges, all
   per-block output checks, pooled loss, failures, logs, source/config/runtime
   hashes, exact checkpoint/cache identities and gate settings. Preserve the
   existing per-layer weight norms and final checkpoint. No clipping is added;
   gradient-interaction measurements are unavailable in this inference test.
   Confirm any additional measurements needed later before launch.

A numerically qualified and repeatable latency reduction versus frozen
supports this direct extension. A slower result, an effect unresolved against
process variation, or numerical failure does not support it. Report component
occupancy/counters alongside the net full-model result, without attributing
individual microseconds to memory, detection or arithmetic. Process ranges
are descriptive, not confidence intervals, and site effects need not add.

A negative result only concerns this port, checkpoint and workload. In
particular, a/m sparsity is lower here than at some T4/Pall endpoints; failure
does not show that a/m load avoidance cannot help elsewhere. A positive
result would require revising the current implementation-limited explanation.
Neither outcome establishes an optimal kernel or a general architectural limit.

## Preliminary execution estimate, not launch authorization

The local RTX5070Ti Laptop has12GB and about10.3GB free in the read-only check,
but does not match the RTX5090 timing hardware. Use one RunPod RTX5090 for the
proposed comparison. Run037's30 scientific processes took18.5 minutes;
fifteen comparable warm processes suggest roughly9-12 minutes, plus new
compilation, checks, environment setup and verified transfers. Provisional
post-implementation ETC:35-65 minutes, with setup uncertainty. This is not a
calibration of the new code. Re-estimate from the bounded smoke phase.

Live catalog read at2026-09-19 15:20:21 UTC: no Pods; RTX5090 Community stock
NONE; Secure stock LOW in EUR-IS-1 and EUR-NO-1, listed atUSD0.99/hour.
Availability and prices will be re-read before the separate launch proposal.
Tentative ceiling:90 aggregate GPU-minutes /USD2 incremental total, including
temporary storage and infrastructure retries. No new shared volume is needed.
Use the existing verified image/runtime route with a temporary persistent Pod
volume. Monitor every60s and refresh progress, loss/errors, throughput and ETC.
Retain a stop guard and retrieval reserve; copy and SHA256-verify completed and
failed artifacts before deleting the new Pod and checking remaining resources.

The launch proposal must report exact code/test results, resource fit, live
price, image/storage/transfer inventory and deadline. Historical approval for
Run037 is not treated as authorization for this new scientific input.

## Evidence pointers and record status

- [Current manuscript evidence](observations/034-operation-latency-manuscript.md)
- [Completed Run037](../../runs/037-2026-09-19-pythia14m-operation-latency/observations/001-operation-latency.md)
- [K049 implementation](../../runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k049/joint.cu)
- [K042 a/m implementation](../../runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k042/projection.cu)

Some older adoption notes in Run037/its design still say the manuscript uses
the grouped-control table. Observation034 and the current TeX now contain
Run037's operation ablations; the older wording is historical, not the current
manuscript state. No user manuscript edits are modified by this proposal.
