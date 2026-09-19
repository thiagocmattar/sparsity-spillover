# Proposed 14M T7/Pall latency attribution

Status: the user explicitly approved this design at kappa=0.5 on 19 September
2026. Implementation is authorized; a separate launch proposal follows tests
and resource assessment. The manuscript revision uses only the existing
group-level timings in Observation 033 until new measurements are qualified.

## Question and interpretation

For the retained 14M T7/Pall checkpoint at **kappa=0.5**, which sparse execution
paths reduce full-model latency, and which add overhead? This selects the
high-threshold endpoint discussed in the subsection, not a new threshold sweep.

Propose six operation groups covering all seven sites:

| Toggle | Site(s) | Operation |
|---|---|---|
| a | a | QKV projection |
| m | m | FFN-up |
| h | h | FFN-down |
| z | z | Attention-output projection |
| QK | q and k jointly | Attention scores |
| PV | v; unchanged attention probabilities | Probability/value product |

Q and K contribute to the same matrix product, and their zero-based savings
overlap. Counting QK jointly avoids inventing an allocation of the same saved
instructions between q and k. These are conditional execution-path effects,
not effects of removing gates, pressure, or retraining individual sites.

## Fixed scientific contract

- Model: random-initialized Pythia-14M, six layers, width128, FFN512, vocabulary
  50304. Retained Run029 c30 final checkpoint (training seed1234, step712,
  1493172224 input tokens); no training or optimizer step is performed.
- Weight SHA256:
  `f83aef36ddbddbd94efb915d574c4645c687206bb68f51886ffc6e979985b425`.
  The original initialization, training order and AdamW configuration are
  retained through this exact checkpoint and its archived training provenance.
- Gates: G-plus at a,m,h,z and signed G-pm at q_post,k_post,v, kappa=0.5.
  The checkpoint was trained with orthogonal L1 on all seven sites,
  lambda=1, b=1. All gates remain enabled at identical values in every mode;
  pressure is not applied during inference.
- Frozen K050 is the starting implementation. Preserve layout, precision,
  fusions, thresholding, residuals and output rounding. In particular, retain
  the joint h/z kernel rather than splitting it into separate timed kernels.
- Execution: one RTX5090, no concurrent workload; BF16, batch1, length2048,
  uncached causal inference, complete50304-logit output, CUDA graphs. Match
  Run029 Python3.12, PyTorch2.11.0, Transformers5.12.1 and CUDA12.8 environment
  where available; record any necessary infrastructure differences explicitly.
- Runtime seed2801; timing sample seed2504 and the exact Run02964 validation
  identities; seven paired passes and three fresh processes per mode.
- Qualification: all500 MiniPile validation documents, all338 complete blocks,
  692224 input tokens, 691886 prediction tokens, excluded tail1444. Compare
  every mode with the unchanged eager reference and qualified native graph.
  Retain Run029 logit atol0.25, rtol0.02, relative-L2 bound0.02 and pooled
  loss absolute-difference bound0.001; no relaxation after seeing results.

## Comparisons

Ten modes, each measured afresh on the same physical GPU:

1. Unmodified frozen K050 anchor.
2. New toggle-capable implementation with every sparse path enabled.
3. Same implementation with every sparse path disabled.
4. Projection skipping only, matching the historical grouped comparison.
5-10. Six leave-one-operation-out modes: disable only a, m, h, z, QK or PV.

All-enabled must pass numerical qualification and reproduce the untouched
anchor's timing within observed process variation before being treated as
an attribution of K050. A detectable implementation overhead is a finding
to report and resolve, not something to hide in the historical denominator.

For each path, report the full-model latency difference (path disabled minus
all enabled) and ratio (path disabled divided by all enabled). Positive
differences identify time saved by that path with the others enabled.
Retain process-level results and their spread; do not treat every inner timing
repeat as an independent experimental replicate. These conditional effects
need not sum to the total gain, especially for the fused h/z path.

The primary manuscript claim is supported if h/z paths reduce latency while
QK/PV do not, after qualification and comparison with process variation. A
different sign, negligible differences relative to variation, failed fidelity
to frozen K050, or failed numerical qualification limits/refutes that account.
No inference about another threshold, recipe, size, GPU or cached decoding
will follow from this single checkpoint. This is not a profile separating
the time of memory reads, zero detection, softmax and arithmetic.

## Diagnostics, retention and next approval

Retain raw paired host/device timings, execution masks and ordering, complete
per-block output checks and pooled loss, source/config/environment hashes,
checkpoint/cache identities, exact/near-zero counts, activation RMS, row/tile
occupancy and per-operation issued/bypassed/scalar counters in untimed passes.
Per-layer weight norms and the final checkpoint are already retained; preserve
their identities. No clipping settings change. Training gradient interactions
are not reconstructible here and are outside this inference-only question.
Confirm whether additional measurements will be needed before launch.

After design confirmation: create a new numbered run, implement and test the
independent switches, then provide focused/full-bootstrap results, a limited
smoke/calibration scope, resource fit, ETC and an explicit launch definition.
If RunPod is needed, discover resources and current prices first, specify a
maximum billable envelope and persistent storage, monitor progress/loss/error/
throughput/ETC, retrieve and hash-verify every artifact before teardown, and
confirm no unintended billable resources remain. Historical cloud budgets
and prices are not reused as current authorization.
