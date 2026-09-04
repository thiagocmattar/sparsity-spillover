# Run 022 post-mortem and revised sparse-kernel plan

## Executive decision

Do not promote A1-H or any 14M frontier yet. The official SparseLM control
works, but the released raw ELL kernel is incorrect at Pythia-14M's `N=128`
down-projection width. The next scientifically defensible step is a separate,
explicitly derived kernel-qualification run, followed by a fresh A0 14M
negative-control run. Padding to 256 may be retained as a labeled upper-bound
adapter, but must not replace the exact-shape result.

The 70M path is more promising: its `h -> W2` output width is 512, and the
unchanged raw operator passed the synthetic `N=512` check. It still requires a
checkpoint-level gate at the actual `M`, `K=2048`, sparsity distribution, and
packing cost before any speedup claim.

## What the run established

| Item | Result | Consequence |
|---|---|---|
| Official SparseLM0.5B control | TwELL 476.9542 ms; Torch 620.0687 ms; 1.3001x | Hopper build and upstream high-level path are reproducible. |
| Pythia-14M exact raw shape | `N=128`, relative L2 0.69055 | Released raw kernel cannot be timed as a valid 14M operation. |
| Width boundary | `N=256/512/2048`, relative L2 about 0.00165 | The failure is width-specific, not a general packing failure. |
| Padded adapter | 128 through 256, relative L2 0.00165 | Correct but computes twice the output columns; report only as an adapter bound. |
| A0 model evidence | Not collected after the fail-closed gate | No A0 speedup or `R_model` mapping exists from Run 022. |

## Recommended next design

This is a proposed design and is not launch authorization for a new numbered
run.

1. **Derived-kernel qualification.** Fork the pinned upstream commit with one
   minimal, separately hashed change that makes the warp reduction uniform for
   output widths below 256. Qualify every multiple of eight from `N=8` through
   `N=256`, plus `N=512` and `N=2048`; use representative `M` values including
   small/partial grids and 2,048-token rows, and `K` in
   `{32, 64, 128, 512, 2048}` to cover 14M, 70M, and attention head shapes.
   Include signed/empty/dense/random rows and overflow guards.
   Require exact ELL round-trip and relative L2 at most `0.02` everywhere.
2. **A0 14M negative control.** Re-run the full approved Run-022 model protocol
   with the qualified derived kernel: all 338 complete validation blocks,
   archived-loss identity, count-first occupancy, dense/identity timing, and
   raw kernel-only/pack/end-to-end timing. Also measure the unchanged padded
   adapter, clearly labeled, if it remains useful.
3. **A1-H 14M sentinel.** Only after A0 passes, measure the naturally sparse
   ReLU checkpoint with exactly the same hardware and timing protocol.
4. **A0/A1-H TEAL ladder.** Run one high-sparsity sentinel per checkpoint, then
   all stored post-hoc thresholds if correctness and timing dynamic range hold.
   Keep validation loss, canonical `R_model`, kernel-covered integer counts,
   capacity/overflow distributions, and measured latency separate.
5. **A4-OL1 and A7-OL1 projections.** Run `kappa=0` and `0.5` first, then fill
   the remaining three kappas only after the sentinels pass.
6. **A7 attention sentinel.** The high-level TwELL module does not implement
   attention. Test only a raw signed composition on one layer/head/configuration.
   QK can pack one of Q or K, never credit both operands' zeros; PV can test
   `(V^T P^T)^T`. Include packing, transposes, causal mask, scaling, softmax,
   and per-head dispatch. The `N=2048` width passed Run 022's synthetic check,
   but no attention speedup is implied.
7. **70M ladder.** Qualify actual `h -> W2` at `K=2048, N=512`, then repeat the
   gated sequence. Do not use the synthetic width result as checkpoint-level
   evidence.
8. **Final fixed-SKU sweep.** Use one exact GPU SKU and randomized/interleaved
   timing for retained configurations. Relate speed to both canonical
   `R_model` and kernel-covered opportunity; do not fit a universal conversion
   from logical opportunity to runtime gain.

## ETC and RunPod cost model

The live H100 NVL rate was `$2.59/hour`. Run 022 used approximately 34 minutes;
the observed balance change was `$1.47576`, consistent with `$1.46767` of GPU
time plus transient and pre-existing storage charges. H100 SXM allocation
failed before creation and incurred no Pod charge.

Measured warm-stage timings were: reused setup 15–30 seconds, the full upstream
positive control 98 seconds, and raw extension build plus fail-fast preflight
35 seconds. The first cold environment install reached its static check in
3 minutes 7 seconds. A clean future bootstrap should therefore budget 7–10
minutes including provisioning and transfer, rather than the original 60–100
minute conservative estimate.

The model benchmark itself never started, so its ETCs below remain engineering
estimates based on the configured fixed-duration timing blocks, not measured
calibrations.

| Test | GPU wall time | H100 NVL cost at $2.59/h | Playbook |
|---|---:|---:|---|
| Derived-kernel qualification | 10–20 min after code is ready | $0.43–$0.86 | Build once; run width/shape/pattern matrix; archive errors and timing blocks. |
| A0 14M, clean cold run | 15–25 min | $0.65–$1.08 | Bootstrap/control; full validation; occupancy; dense/raw/pack timings; verify. |
| One additional 14M checkpoint/config, warm Pod | 5–12 min | $0.22–$0.52 | Reuse environment; validate identity; collect counts; interleave timings. |
| One 70M checkpoint/config, warm Pod | 7–15 min | $0.30–$0.65 | Same, with actual `K=2048, N=512` correctness gate first. |
| A0+A1-H 14M TEAL set, 20 stored points | 1.7–4.0 h | $4.40–$10.36 | Run sentinels first; then batch sequentially on one fixed Pod/SKU. |
| One A7 attention sentinel after implementation | 20–40 min | $0.86–$1.73 | One layer/head; QK and PV separately; include all layout/dispatch costs. |

Source-level kernel correction and review is estimated at 1–3 engineering
hours before the 10–20 GPU minutes. A complete 60-configuration sweep should
not be budgeted until A0, A1-H, and one attention sentinel establish actual
per-point ETCs. Using the current warm estimates, a serial fixed-H100-NVL sweep
would be roughly 6–15 GPU-hours (`$15.54–$38.85`); this is deliberately a broad
planning envelope, not a launch request.

## Per-test operational playbooks

### Kernel qualification

Pin upstream commit and derived patch hashes; compile for `sm_90a`; prove
correctness before timing; test empty, full, signed, nonuniform, and overflow
rows; retain raw timing blocks and compiler/runtime identity. Stop on any
shape-specific error.

### Projection or MLP checkpoint

Hash-lock the checkpoint/cache; reproduce complete-validation loss; capture
integer-pooled exact/near-zero counts and row/tile occupancy; pack without
truncation; compare sparse BF16 with dense BF16 and FP32; interleave dense,
kernel-only, packing, and end-to-end measurements. Preserve bias and gate order.

### TEAL operating point

Load one common checkpoint, apply only the archived threshold at its declared
site, re-evaluate all 338 complete blocks, and repeat the matched timing
protocol. Do not relabel thresholds as trained models or retune them on these
runtime results.

### Attention sentinel

Keep hook placement, scaling, causal masking, softmax, and head layout exact.
Measure QK and PV separately; credit only the operand actually packed; include
packing/transposes/head dispatch in end-to-end timing; compare with eager and
Flash SDPA. A projection result is not an attention result.

### Final sweep

Use one exact SKU/software stack, warm and randomize/interleave retained points,
record integer `R_model` and kernel-covered numerators separately, and report
kernel-only plus model-level latency with uncertainty. Preserve negative and
non-break-even controls.
