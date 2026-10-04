# Run061: scalar versus causal corner-tile inference latency

Status: design explicitly confirmed on 4 October 2026; implementation and
pre-launch checks complete. Cloud launch approval is pending. No billable
resource has been created and no accepted latency result exists yet.

Question: do the additional aligned empty tiles in Run060 lower full-model
inference latency relative to the matched scalar T2/Ph checkpoints of Run058?
The four final step 712 seed 1234 checkpoints are scalar and corner 8x16 at
kappa 0/.1. No P2/Pall or a/m gate is included. The source recipes used matched
random initial bytes, data schedule and 712 updates, AdamW task moments plus
orthogonal_l1 only at h, lambda 1 and relative pressure budget 1. Inference loads
these retained weights without retraining, optimizer steps or new pressure.

The native operator is copied byte-for-byte from Run060. A corner <=kappa
rejects its fixed 8-token by 16-feature tile; an admitted tile preserves every
original value, including negative and subthreshold interiors. Scalar equality
survives. Both thresholds are converted to BF16 like their native wrapped-scalar
comparisons. No scalar threshold is reapplied inside active corner tiles.

## Execution and comparison

Use one physical RTX5090 and the prior BF16 B1/T2048, uncached, full 50304-logit
workload. Three fresh processes per threshold retain 64 fixed validation inputs,
seven randomized interleaved timing passes and 1344 paired samples. Runtime
seed 2801 and timing seed 2504 are unchanged. Disable TF32 and reduced-precision
BF16 reductions. Report synchronized host latency as primary, CUDA events,
tokens/s, all process ratios, peak memory and setup/compile time separately.
Input staging and setup are excluded; recurring gates, support inspection,
packing, attention and full logits remain timed. Process repeats are not new
training seeds.

Each checkpoint has native, sparse and no-h/z-skip graph implementations,
qualified against its own unmodified eager checkpoint. Native graphs use the
retained fixed uncached scaffold, avoiding capture-induced SDPA dispatch drift.
The adapted K049 consumer fuses the trained scalar/corner rule in support
inspection and activation-fragment loads. Its rejected tiles bypass the
corresponding source-level weight loads and matrix instructions. K050 dense
components, attention and residual rounding remain matched. The no-skip
control disables only h/z tile bypass and short-row substitution, together;
it is not an ablation of every zero test or a pure tile-only effect. Historical
kernel sources and previous run inputs remain unchanged. The small derivation
is retained in `provenance/joint-derivation.diff`.

Every scientific process checks all 500 MiniPile validation documents:
338 complete 2048-token blocks,692224 input tokens,691886 predictions,
1444-token excluded tail. Logit atol.25/rtol.02, relativeL2<=.02 and pooled
loss delta<=.001 must all pass. A failed component/smoke/full numerical gate
stops the cohort and prevents an accepted speedup claim. Loss in BF16 retains
its own precision identity; original FP16 quality remains in Run058/060.

## Diagnostics and evidence limits

The user explicitly confirmed the proposed post-hoc inventory as sufficient:
raw timings, numerical/loss checks, per-site/layer exact/near-zero counts at
0/.001/.01, RMS/L2, weight norms, row occupancy and aligned 8x16 empty-tile
integer counts; h/z instrumented MMA/SIMT counters with independent operand
verification. The retained BF16 logical diagnostic is explicitly a lower bound
excluding weight zeros and softmax-probability underflow, as in Run059. It is
not canonical FP16 R_model. Full original logical counters and training-only
gradient/OL1 boundary evidence remain in the source runs; all final models and
recovery states are already local. No clipping sweep is selected.

Tile/scalar sparse speedup>1 with consistent fresh-process results supports
lower latency in this workload. <=1 refutes that advantage. Each checkpoint's
matched no-skip/sparse comparison distinguishes the contribution of h/z sparse
execution from other dense optimizations. Empty tiles and issued MMA counters
do not measure DRAM bandwidth. One GPU/seed supplies no portability, decoding,
training-speed or equal-quality conclusion: the tile models have substantially
higher original validation loss.

The manuscript question is actual sparse execution versus logical opportunity
(`draft/12-appendix-kernel-validation.tex`), with the exact trained Run060 gate
as operational definition. No manuscript change or finding promotion is part
of this run. Tile pressure remains deferred.

## Implementation and pre-launch checks

- `01_prepare.py` packages four verified original models, full validation and
  byte-identical frozen dependencies. All 1403 source/input identities pass.
- `joint.cu`, `kernel.py`, `replay.py` implement exact fused h/z gates and the
  matched no-skip control; `00_component.py` independently checks CUDA values,
  changed-input graph replay, branch/residual rounding and work counters.
- `02_benchmark.py` and `03_execute.py` collect the six fixed paired processes;
  `diagnostics.py` reconstructs the actual gated signed operands for untimed
  integer counts. `20_analyze.py` rejects incomplete or numerically unqualified
  evidence. `21_figure.py` produces one PDF and its source-linked observation.
- `04_setup.sh` installs pinned runtime and immutable vendor sources on fast
  local Pod disk before any compilation; logs, original inputs and outputs
  stay on the workspace volume. `11_verify_cache.py` checks every fast copy.
  `00_precompile.py` uses the same source/runtime paths as measurement.
- `06_transfer.py`, `07_local_stop_guard.ps1`, `08_deadline_guard.py` provide
  identity-scoped transport, retrieval and absolute-deadline backstops.

Local component CUDA qualification passes 16 checks on RTX5070TiLaptop in
36.75 s including compilation. Four real native checkpoints each pass two
B1/T2048 blocks against eager with zero observed logit error; the four graphs
resident together peak at 3.912 GiB reserved. This is development/memory evidence,
not full K050 qualification or RTX5090 latency. Seven focused CPU checks pass
in 4.33 s, including all four original checkpoint forwards and eight optimized
installation cases. All 242 bootstrap tests pass in 11.03 s. See `prelaunch/`.
Mandatory real RTX5090 smokes of both threshold pairs precede complete
measurement after launch approval.

## Proposed launch

See [LAUNCH.md](LAUNCH.md) for current inventory, live quote, resource fit,
ETC, maximum duration/cost, storage, transfer, monitoring and teardown. Source
checkpoints remain local. No cloud launch is authorized by this implementation
commit or by design approval alone.
