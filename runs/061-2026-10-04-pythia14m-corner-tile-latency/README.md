# Run061: scalar versus causal corner-tile inference latency

Status: completed on4 October2026 after explicit design, diagnostic and launch
approvals. Six full processes, numerical qualification, diagnostics and transfers
are verified locally; the owned RTX5090 Pod is terminated. Sparse corner tiles
reduce latency10.990%/10.067% versus sparse scalar at κ=0/.1, with higher loss.

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

## Launch packet and execution record

See [LAUNCH.md](LAUNCH.md) for current inventory, live quote, resource fit,
ETC, maximum duration/cost, storage, transfer, monitoring and teardown. Source
checkpoints remain local. The preparation commit did not authorize cloud launch;
the user's later explicit "Aprovado" authorized the quoted envelope.

## Verified outcome

Accepted evidence is the six-process controller cohort002; `20_analyze.py
--tag 002` verifies120 transferred members, unchanged measured sources/config,
the exact module-selection bootstrap per process, checkpoint identities, same
physical GPU, all paired timings, full numerical checks and diagnostics.
Each threshold has1344 same-input/pass scalar/tile pairs from three fresh
processes. [Full table](results/tables.md), [machine result](results/results.json)
and [observation](observations/01-corner-tile-latency.md) retain provenance.

| Gate | κ | Native ms | No H/Z skip ms | Sparse ms | Native/sparse | No-skip/sparse | BF16 loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| scalar | 0 | .645658 | .601531 | .606922 | 1.0638x | .9911x | 5.128739 |
| tile8x16 | 0 | .746999 | .603671 | .540220 | 1.3828x | 1.1175x | 5.556248 |
| scalar | .1 | .646298 | .603103 | .565242 | 1.1434x | 1.0670x | 5.153716 |
| tile8x16 | .1 | .747714 | .606030 | .508340 | 1.4709x | 1.1922x | 5.598785 |

Direct sparse tile/scalar ratios are1.123472x and1.111937x: latency decreases
10.990186%/10.066859%, throughput increases12.347%/11.194%. The three process
ratios are1.124849/1.121217/1.124353 at0 and1.110059/1.112559/1.113195 at.1.
Sparse execution reduces the tile checkpoints' own no-skip time by10.511% and
16.120%; scalar κ=0 receives no observed h/z skip benefit (.9911x).
The no-skip control disables both tile bypass and short-row SIMT substitution,
so it does not isolate tile-only skipping. Larger native/tile speedups also
include differences in native gate overhead and other optimized components.

All three graph modes have zero observed own-eager logit error and exactly
equal pooled loss in every accepted process over338 blocks/500 documents,
691886 predictions, with1444 tail tokens excluded. In this BF16 benchmark the
tile/scalar loss penalties are+.427510/+.445069. Original canonical FP16 quality
remains in [Analysis041](../../analyses/041-2026-10-04-corner-tile-versus-elementwise/README.md):
loss+.426737/+.444400 and perplexity+53.225%/+55.955%. No equal-quality claim.
Peak allocated memory is4.125GiB; reserved4.617--4.623GiB. These are full-logit
uncached B1/T2048 results on one RTX5090, not cached decoding or model scaling.

Initial attempt001 stopped at a diagnostic import collision after valid timing
and numerical checks. Its36-member verified download and failed attempt are
preserved and excluded from accepted estimates. The append-only
[retry record](infrastructure/retry-002/README.md) explains the infrastructure
bootstrap; all scientific inputs stayed byte-identical. Both smokes were repeated.

Creation14:40:30.515 UTC; termination confirmed14:58:52 UTC,18m21s of Pod life.
Estimated GPU costUSD.302908, plus small storage; scoped official billing had
no posted records at closeout, so this is provisional rather than an invoice.
Both inventories/archives are SHA-verified; final transfer has120 files and
19,607,531 uncompressed bytes. No Run061 billable resource remains, its local
deadline guard is disarmed, and foreign Pods/volume were untouched. See
`prelaunch/closeout-001.json`. The single PDF is visually verified; no manuscript
edit or finding promotion was requested. Tile pressure remains deferred.
