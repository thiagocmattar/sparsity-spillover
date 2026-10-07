# Run063: corner-specialized T2/Ph inference consumer

Status: complete and verified. Specialized sparse latency falls 10.526%/8.896%
at kappa 0/.1; all five graphs match eager logits and loss exactly over full
validation. Artifacts are local and the owned Pod is deleted. [Observation](observations/01-generic-versus-specialized.md).
The user approved option 2 on 7 October 2026: specialize the consumer for the
two existing 14M T2/Ph Run060 corner8x16 checkpoints at kappa 0/.1. Separate
billable launch was approved on 7 October 2026 ("Ok. Go on.. execute it").
The user retained a flexible maximum instead of using the revised ETC as a
timeout; the two-hour/$3 backstop was retained and disarmed after teardown.

Owned Pod `yp38ht95t3ag67` was created at 18:42:31.334 UTC in EUR-NO-1 at $.99/h,
host CUDA 12.8, with the pinned application and sealed input bundle unchanged.
No other Pod or network volume was created. Initial regional capacity and
transport/readiness issues are recorded in [retry001](infrastructure/retry-001/README.md).
Pinned-wheel setup001 then failed on a 30-second network timeout before any GPU
check. Its logs were retrieved and hash-verified; [retry002](infrastructure/retry-002/README.md)
uses the same Pod, package pins and scientific inputs, longer network timeouts
and fresh scientific tag002. The accepted results below come from tag002.

Question: does consuming known corner-tile structure directly reduce full-model
latency relative to the generic Run061 consumer, at unchanged weights and gate?
No training, tile-statistic gate, pressure change, or manuscript edit is included.
Both step712 seed1234 models, their random-initialization and schedule identities,
and the full MiniPile validation cache are hash-verified copies of Run061 inputs.
Original FP16 losses remain 5.5537703333521735 and 5.5967455502797865.

The gate remains exactly Run060: corner <=kappa zeros its nonoverlapping 8-token
by16-feature tile. An admitted tile preserves all signed interior values. h is
the FFN nonlinearity replacement, z is the context immediately before Wo;
there are no a/m gates. BF16 thresholds use the original wrapped-scalar rounding.

## Consumer and controls

`generic.cu` is a byte-identical copy of Run061 `joint.cu`. It inspects every
row/element, supports <=2-nonzero SIMT substitution and bypasses empty M8/K16
products before loading weights. `joint.cu` derives a bitmask directly from
corners, uses four projection warps rather than eight inspection/projection
warps, and loads admitted activation pairs without rechecking each corner.
It retains the same padded M16 MMA, ascending K order, biases, separate BF16
branch rounding, BF16 branch sum and residual rounding. All-dead blocks still
write the ordered bias/residual result. There is no short-row SIMT substitution.
This is a consumer package comparison, not an isolated causal effect of just
one instruction, thread count, or memory transaction.

Each threshold process retains five graphs: native, generic sparse/no-h/z-skip,
specialized sparse/no-h/z-skip, plus its own native eager reference. Generic
no-skip disables tile bypass and short-row substitution; specialized no-skip
disables tile bypass. Dense K050 components and attention remain identical.
Source-level avoided weight loads do not establish measured DRAM traffic.

## Measurements and qualification

One physical RTX5090, BF16 B1/T2048, uncached full50304 logits. Runtime seed2801,
timing seed2504, TF32/reduced BF16 reductions off. Three fresh processes per
threshold,64 fixed inputs x7 randomized paired passes:1344 paired observations
per comparison. Primary synchronized host latency excludes equal input staging
and all setup/compilation; recurring gates, support and projection costs remain.
CUDA-event times, individual process ratios, tokens/s, setup and peak memory
are retained. Repeated processes are not additional training seeds.

Every graph must pass against its own eager model on all500 MiniPile validation
documents:338 complete2048-token blocks,692224 inputs,691886 predictions;
the1444-token tail is excluded. Bounds:logit atol.25/rtol.02, relativeL2<=.02,
pooled loss delta<=.001. Stop on failed component, smoke, or full qualification.

The user confirmed the post-hoc set sufficient: raw timings, full numerical/loss
qualification, per-site/layer zeros/near-zero mass at0/.001/.01, RMS/L2, weight
norms,8x16 empty tiles, independent MMA/SIMT counts and standalone inspection
cost. `inspection.cu` copies the generic inspection verbatim; its standalone
256-thread support diagnostic compares row masks to corner masks on one
real input's twelve h/z operands.100-call graphs x10 randomized paired repeats
are hot-resident component timings, not additive end-to-end attribution.
Final checkpoints and training gradient/OL1 evidence remain retained in Run060.

`20_analyze.py` rejects incomplete/changed/unqualified or unpaired evidence.
Speedup>1 consistently across processes supports an advantage for this exact
consumer/workload. A result<=1 refutes that advantage here. Quality-equivalent
kernel execution does not remove the trained corner gate's quality penalty.
No decoding, other model size/family/GPU, or equal-quality frontier claim follows.

## Pre-launch evidence

-1389 immutable source/input identities verified; the generic CUDA SHA equals
  Run061. The two models and validation coverage are unchanged.
-22 local CUDA component checks passed, including threshold equality, signed
  admitted interiors, zero/two-value blocks, bias/residual rounding, counters,
  changed-input replay and standalone mask OR equality.
-10 focused CPU tests and242 bootstrap tests passed together in10.48 seconds
  on final closeout (the earlier combined check took15.90s).
-Full local consumer qualification passes at both thresholds: all five graphs
  match own eager logits and loss exactly over338 blocks/500 documents per
  checkpoint. This development check uses matched native SDPA on the laptop;
  it does not qualify the frozen RTX5090 attention path. Source-hash receipt and
  full numerical rows are in `prelaunch/native-local-receipt-005.json` and
  `prelaunch/native-cuda-005.json`.296.68s including cold compilation;
  peak4.016GiB allocated/4.520GiB reserved.
-All23 run-local Python sources parse; both shell scripts and the PowerShell
  stop guard pass syntax checks. All1422 sealed upload members pass byte-count
  and SHA256 verification. Source/input verification covers1389 identities.
-The previous local numerical failure is retained and isolated to shared K035
  attention, not the specialized consumer. See the
  [qualification observation](observations/01-local-numerical-qualification.md).
  Numerical bounds, weights and gates were unchanged. Production remains the
  original frozen K050/K035 path and must pass fresh RTX5090 component, smoke
  and full qualification after billable launch approval.

Preparation fixes: `01_prepare.py` now reads the manifest's actual
`generic_sha256` field; benchmark/precompile diagnostics resolve run-local files
by explicit path despite archived helpers modifying `sys.path`. Failed full
qualification stops before diagnostics. The scientific consumer CUDA sources
remain identical to the passing22-check component receipt.

The source scripts, evidence and launch envelope remain in this run. PDFs and
source-linked observations will be produced after accepted results. Inputs and
scientific code become append-only once an executed cloud attempt begins.

## Accepted RTX5090 results

Three fresh processes per threshold on one physical RTX5090,64 fixed inputs
x 7 paired passes,1344 paired observations per comparison. Geometric mean host
latency includes the recurring gates/support/projections and full 50304 logits,
and excludes equal staging and setup. BF16 B1/T2048 uncached workload.

| kappa | Generic sparse ms | Specialized sparse ms | Speedup | Latency reduction | Process speedups |
|---:|---:|---:|---:|---:|---|
|0|0.578467|0.517577|1.117644x|10.526%|1.113239, 1.112600, 1.127152|
|.1|0.545248|0.496740|1.097651x|8.896%|1.100685, 1.100174, 1.092116|

Both no-skip controls were measured. The specialized package is 3.409%/4.775%
slower than generic with skipping disabled; the advantage therefore depends
on this sparse workload and skip handling. This package comparison includes
corner support consumption and warp/block changes. It does not isolate their
causal contributions or establish measured DRAM traffic savings. The complete
five-graph host table and CUDA-event measurements are in the observation.

All 22 component checks and both eight-block smokes passed on RTX5090. Every
full process qualified all five graphs against its own eager checkpoint on
all 500 documents/338 blocks,692224 inputs and 691886 predictions, excluding the
1444-token tail. Across 10,140 graph/block comparisons, maximum absolute logit
error and relative L2 are zero. Pooled BF16 losses match eager exactly:
5.556248486078721 and 5.5987847543876. Bounds, weights and corner-gate semantics
were unchanged. Activation and tile diagnostics agree exactly between consumers.
Peak memory is 3.898 GiB allocated/4.354 GiB reserved.

The answer is yes for these retained models and this workload: the tailored
consumer recovers additional speed with unchanged inference outputs. This
does not remove the existing trained tile models' quality penalty or establish
performance for decoding, another batch/sequence/GPU/model, or the deferred
tile-selection metric experiment.

Evidence: [results](results/results.json), [generated table](results/tables.md),
[PDF](figures/01-generic-versus-specialized.pdf), and the source-linked
[observation](observations/01-generic-versus-specialized.md). The accepted
reduction is `20_analyze.py --tag 002`; `21_figure.py` generated the visually
verified single-page PDF. No manuscript edit or finding promotion is included.

All 86 returned files/14,104,050 bytes passed inventory SHA256/byte checks before
termination. Original checkpoints remain retained locally; this run created
none. Creation 18:42:31.334 UTC, deletion confirmed 19:34:24 UTC:51 min 52.666 s of
Pod life. The six full processes took 4 min 47.748 s, including qualification and
diagnostics; timing phases totalled 12.511 s. Upload, setup, compilation and
infrastructure retries account for the remaining time. Estimated GPU cost is
USD 0.855983 plusUSD 0.007699 running storage (aboutUSD 0.864 total), provisional
until official billing posts. Scoped billing returned no posted records at
closeout. The owned Pod is absent, its guard is disarmed, and the pre-existing
stopped Pod/network volume were unchanged. [Closeout](prelaunch/closeout-001.json)
and [verified transfer](transfer/receipt-002.json).
