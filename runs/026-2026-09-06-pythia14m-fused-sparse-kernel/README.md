# Run 026: Pythia-14M fused sparse kernel search

Status: completed and verified; 1.8149x over qualified stock eager SDPA.
All evidence is local; the test Pod was deleted and zero Pods/endpoints remain.
This continues the user-authorized
autoresearch objective (>1.6x full-model speedup, $25 total RunPod budget).
Run 025 is retained as the immutable source/evidence predecessor.

## Fixed experiment

Target: Run 025 `14m/a7-0p5`, the retained Run 014 step-712 checkpoint.
Pythia-14M has six layers, hidden 128, FFN 512, four heads, vocabulary 50304.
Inference is BF16, B=1, T=2048, full logits, causal SDPA, no KV cache.
All weights, trained gates, thresholds, topology, seeds and pressure provenance
are imported by hash from Run 025's input manifest. No new training occurs.
The source used seed-1234 random initialization and orthogonal L1 at A7-Z-POST
sites; one-sided kappa=.5 at a/m/h/z and symmetric .5 at q_post/k_post/v.

Hypothesis: exact gate/projection fusion and shape-specialized sparse kernels
can reduce end-to-end latency while satisfying the sealed Run 025 numerical
gates. Search varies only implementation, tiling, site selection, and legal
fusion. A correct full-model paired speedup >1.6x supports the objective;
failure to reach it is a bounded negative result, not a universal limit.

Primary comparator is the fastest correct screened stock dense mode with equal input
staging/capture treatment. Eager speedup is also recorded. Dense configurations
are selected on development inputs before candidate promotion. Canonical
R_model is inherited with integer counts and provenance for every variant;
actual BF16 diagnostics are collected separately. Product counts do not imply
a wall-clock ceiling or speedup.

Development: 64 training blocks, seed 2500; timing initially uses their first
16 with seed 2504. Finalists freeze before validation timing (64 validation
blocks selected by seed 2504), seven paired passes and three fresh processes.
Quality covers all 500 documents / 338 complete blocks / 692224 input tokens;
the 1444-token tail is excluded. Logit bounds are atol=.25, rtol=.02,
relative L2=.02; absolute pooled loss difference <=.001 nat/token.

Every attempted variant has a row in `autoresearch/artifacts/*/result.json`,
raw paired samples, code/config hashes, an idea and candidate ID. Compilation
or numerical failures retain their available latency and counts, with explicit
qualification. `autoresearch/progress.csv` tracks every iteration and the best
qualified speedup so far. No failed variant is promoted or omitted.

## Budget and evidence

The new $25 cap includes all new Pod attempts, startup, compilation, transfer,
idle development time and storage. Prefer community RTX 5090; use a measured
live-price fallback when capacity is absent. Protect final validation/retrieval
funds; arm an independent stop guard for each lease. Durable work lives under
/workspace. Retrieve and verify hashes before deleting a Pod.

Collect/retain: exact and near-zero counts at 0/.001/.01, activation RMS/L2,
weight norms, logical products and architecture-ceiling counts, occupancy,
kernel/operator timings, workspace/memory, branch coverage, all errors, and
raw paired timings. The original final checkpoint and full cache identity stay
local. Gradient interaction is training-only; this inference run links the
existing training record. No extra clipping or gradient experiment is added.

This run addresses the runtime qualification of manuscript methodology
`eq:r-model-measured` and `eq:r-model-max`; no TeX change is authorized here.
Matched dense controls distinguish sparse-kernel contributions from capture
and other implementation improvements. Monitor at phase completion or every
10 minutes during long work, reporting loss, throughput, progress and ETC.

## Authorization

The user reiterated the active objective with full access and approval for all
RunPod runs, deployments and transfers on 2026-09-06 after the detailed design
proposal. Proceed under that standing authorization, preserving the inherited
strongest-matched-dense comparator rather than adding another approval wait.

## Iterations

- K017: fuse the existing one-sided gate into exact warp-compacted projection;
  specialize output elements per lane and row warps for narrow Pythia shapes.
  The gate consumes each new input inside the timer; equality survives.
- K018: jointly execute exact sparse W2 and Wo, their biases, and the two
  parallel-residual additions. Explicit BF16 rounding preserves both linear
  outputs and the intermediate branch sum. GPU qualification passed.
- K019: fuse partial RoPE, symmetric Q/K/V gates, and contiguous head layout;
  preserve BF16 rounding after each product and sum. Compose with K018.

## Launch record (historical)

Local: six focused tests and all 242 bootstrap tests pass. CUDA tests remain
remote-only: 120 primitive configuration/shape/input/replay cases, followed by
dense eager/graph/compile screening and complete validation. Local GPU has
about 9 GB free; the rented CUDA development environment is used under the
standing cloud authorization.

First lease: `gc38kxs85gwxra`, RTX 5090 community, confirmed $0.69/hour,
40 GB container + 40 GB Pod volume, pinned RunPod PyTorch image digest.
Independent stop guard PID 28976 is armed for 2026-09-06 15:52:47 UTC.
Maximum first lease: four hours (~$2.80 including Pod disks); full study cap
remains $25. Input archive is 53.4 MB. Exact lease, deadline and guard logs
live in `launch-control/rtx5090-001/`. Verify/retrieve before deletion.

Infrastructure chronology: the first SSH setup stopped before installation
because CUDA's bin directory was absent from PATH. The corrected setup
completed at the pinned Torch 2.11.0+cu128 / Transformers 5.12.1 runtime.
The direct SCP upload was slow; RunPod's encrypted relay completed and its
53,371,675-byte archive SHA-256 matched locally and remotely. All 46 input/code
files (60,002,752 bytes) then passed remote verification. The first relay
receiver used the requested code without the sender-emitted relay suffix;
the second used the actual emitted code. No transfer key is in this record.

The first CUDA preflight stopped at the extension loader because the venv's
Ninja executable was not on PATH. No kernel compiled and no latency was
measured. The unchanged K017 source is retried with the venv bin directory
added. Preserve `runtime/calibration.log` (failure) and
`runtime/calibration2.log` (retry). The retry timeout PID is 811; the first
setup/calibration handles were verified terminal before retrying. These are
infrastructure retries inside the same scientific run.

## Final result and scope

The frozen K019+K018 bundle achieves **1.8149029x** paired geometric-mean
speedup (95% crossed process/input bootstrap interval **1.8098012-1.8192157x**).
Pooled median synchronized full-forward host latency is **1.4846455 ms** versus
**2.6929291 ms** for canonical stock eager SDPA. This corresponds to 1,379,454
versus 760,510 input tokens/s at those medians, not autoregressive decode speed.
Per-process speedups: 1.8179186x, 1.8094127x, 1.8173899x.

All three fresh processes evaluated all 338 validation blocks: **bitwise-identical
full logits**, maximum absolute/relative-L2 difference zero, and identical pooled
loss **5.83130066301001 nat/token**. This is native-BF16 inference loss, not the
source training/FP32-autocast loss. 64 held-out input identities x 7 randomized
paired passes x 3 processes produce 1,344 pairs. Input staging is equal and
excluded; all full-vocabulary logits and synchronization are inside timing.
The model/checkpoint/candidate source hashes are audited by `reduce_final.py`.

Comparator qualification is important: stock CUDA graphs and both screened
torch.compile modes failed the fixed numerical gates. Hoisting the lazy import,
isolating compiler models, fixing the cuBLAS workspace, and preserving Inductor
precision casts did not qualify them. Their faster timings remain recorded but
are not promoted. The untimed graph diagnostic first differs at attention
context `z`; threshold crossings amplify small attention differences. This
localizes the failure, not a proof of its complete backend cause.

The winner uses the original canonical attention wrapper on its reference and
the default cuBLAS workspace, avoiding the slower fixed-workspace control.
The >1.6x result is **against the qualified stock reference**, not every possible
optimized dense implementation. The frozen dense-projection/QKV-fusion ablation
already reaches **1.4573163x**; joint sparse projection/residual fusion alone
reaches **1.1760117x**. Each ablation has one independent process and its own
paired stock reference. Their speedup ratio is not a directly paired incremental
measurement. Do not attribute the complete 1.815x to zero skipping alone.

Canonical `R_model` is **0.27482684296304843** for every variant: pooled integer
zero products **1,748,738,568,719** / model products **6,363,055,915,008**.
It is inherited from hash-checked source full-validation counts, not recounted
from each implementation. The full integer architecture-ceiling record is also
retained. Separate native-BF16 SDPA diagnostics cover all 338 blocks, 42 site/layer
pairs, thresholds 0/.001/.01, RMS/L2, every parameter's norms, and h/z active-row
histograms. Native-BF16 exact zeros at h/z are 99.87547% / 99.91704%, respectively.
No diagnostic attention substitution is used to claim an actual-BF16 R_model.

## Evidence and reproduction

- `autoresearch/candidates/k019/` plus `k018/`: frozen winner CUDA and adapters.
- `autoresearch/frozen.json`: source/config freeze before final timing; its
  timestamp correction uses actual local file creation time, without changing
  candidate code. Final process 1 first logged at 12:23:44 UTC, after the freeze.
- `autoresearch/progress.csv`: all 19 iterations and best-qualified-so-far;
  **these are development timings**, including development checks in final
  processes. The historical 1.8482x maximum uses a different workspace and is
  not the final claim.
- `autoresearch/all-variants.csv`: 69 mode/scope rows, including every dense
  control, unsupported compilation and numerically invalid timing. Missing
  latency means setup failed; it is not zero. Primitive checks are correctness
  probes, not timed full-model variants.
- `autoresearch/artifacts/*/`: immutable raw timings, complete validation gates,
  pooled losses, setup failures, code/input hashes and progress events.
- `autoresearch/final-summary.json`: final timing, uncertainty and ablations.
- `runtime/`: committed terminal primitive evidence (120 K017 + 16 K018 + 10
  exact K019 GPU cases, all passed), diagnostics, profiles, runtime versions and
  phase logs. Runtime environments/caches are ignored and not committed.
- `observations/01-speedup-progress.md` and `figures/01-speedup-progress.pdf`:
  numerical method, figure, result and limitations.
- `launch-control/rtx5090-001/closeout.json` and `transfer-inventory.json`:
  archive verification, teardown, billing snapshot and budget estimate.

GPU phases 001-019 and diagnostics completed normally. CUDA graph replay input
mutation is covered by primitive tests. Final timed-process peak allocated
memory was 2,970,664,960 bytes, including validation buffers; diagnostic peak
was 277,897,728 bytes. CPU tests: 12 focused run-local tests and all 242 bootstrap
tests passed at closeout. No shared scientific source or manuscript was edited.

To reproduce reduction and figure locally, from repository root:

```powershell
.venv\Scripts\python.exe runs/026-2026-09-06-pythia14m-fused-sparse-kernel/autoresearch/reduce_final.py
.venv\Scripts\python.exe runs/026-2026-09-06-pythia14m-fused-sparse-kernel/17_plot_progress.py
```

GPU runtime is pinned by `01_setup.sh`; `11_...sh` through `16_...sh` retain
the exact executed phase definitions. A new execution must use a new numbered
run or approved unchanged infrastructure-attempt directory, never overwrite
these terminal attempt names. Input weights/caches remain in their original
local source locations; they are intentionally not committed. Ignored source
transfer bundles retain intermediate harness versions for this local search.

## Teardown and budget closeout

At 12:29 UTC, all known worker PIDs were terminal and nvidia-smi showed no GPU
processes. The final 884,920-byte evidence archive had SHA-256
`047d17fa1c2cfc3f2f963b31fd2fe63c4c2cdac155f09c98e63d814df40fabd7`.
All **196 files / 6,200,054 bytes** passed local inventory verification before
deleting Pod `gc38kxs85gwxra` at approximately **2026-09-06 12:29:49 UTC**.
Its temporary container and Pod-volume copies were removed; source checkpoint
and all agreed evidence are retained locally. The independent local guard was
then cancelled. Account-wide audit confirmed zero Pods, zero endpoints, and only
the unchanged pre-existing 100 GB volume `9luykg5yc3`.

Conservatively measuring from launch request through deletion gives 0.61714 h:
GPU $0.42583 at the returned $0.69/h, new Pod disks about $0.00676, and an
additional $0.00592 prorating the existing volume. Total estimate **$0.43851**,
well within $25. Disk estimates use the [official storage prices](https://docs.runpod.io/pods/storage/types)
checked 2026-09-06 and 730 h/month. This is **not a posted invoice**: the final
scoped billing API response still contained no records. That lag is preserved,
not interpreted as free compute. No new ongoing billable resource remains.
