# Matched 70M implementation transfer

## Question

Does the sparse structure induced by the approved interventions remain
exploitable at 70M, and how much does the original 14M-derived implementation
benefit from the additional, frozen 70M optimization step?

## Method and coverage

Run045 compares native PyTorch/SDPA, the original Run035 port and Run042's frozen
opt073 on 26 retained final checkpoints. These are Base/ReLU, T4/T7 with Ph or
Pall at kappa=0,.01,.05,.1,.5, and four T2/Ph conditions at kappa=0,.01,.05,.1.
Run046 is outside the approved cohort. Weights and thresholds are unchanged;
there is no new training, clipping or kernel search.

One physical RTX5090; BF16 operands/output, FP32 accumulation; batch one,
2,048-token causal full forward, all 50,304 output logits. Three fresh processes
per checkpoint each measure all three implementations on the same 64 fixed
validation inputs, seven randomized paired passes: 1,344 host observations per
implementation/checkpoint. Publication latency is their geometric mean.
Compilation, static layouts, graph capture and equal input staging are excluded;
all recurring model work and activation-dependent inspection are included.

Every final process checks all 338 complete validation blocks from all 500 MiniPile
documents (692,224 input tokens; 691,886 prediction tokens; 1,444-token tail
excluded). Finite logits must meet the fixed elementwise, relative-L2 and
pooled-loss gates. Unqualified measurements remain retained and receive no
published speedup. Full diagnostics are collected in the first replicate of
each checkpoint for both custom implementations. The retained checkpoints,
source inventory and independent dense/component controls remain available.

Canonical loss is ordinary final-checkpoint validation, unchanged for the 79
historical endpoints. BF16 qualification losses and logical-pass losses are
separate fields. Sparsity pools integer logical-product counts before dividing.
Adding four Run043 T2/Ph checkpoints and the separate Run046 kappa=.5 endpoint
gives 84 complete endpoints (45/27/12 at 14M/70M/410M). The main figures retain
41 14M and 26 matched 70M checkpoints. Run046 is appendix-only, as explicitly
requested by the user; its original-port timing does not enter the optimized plot.

## Result

All 78 final processes qualify across all 26 checkpoints. The 819 returned
files pass size/SHA256 checks. Local reduction exactly reproduces the remote
summary (SHA256 d7838110e86a08bb098b38fb9716248005d8de70ff0991a0688fe512240ba582).
Both owned Pods are deleted; the conservative total estimate is below USD2.60.

The native Base reference is 1.611048ms. The original port's best endpoint is
T7/Ph kappa=.5 at 1.589621ms (1.013479x native Base). The optimized implementation
takes 1.088146ms there (1.480544x); all 26 checkpoints improve over their own
original-port latency. Only six optimized checkpoints beat the fixed native
Base reference. This is an implementation-specific gain, not a scaling law.
T2/Ph at kappa<=.1 stays close in loss (+.028902 to +.092734 above Base) but
requires 1.955582--2.504098ms, remaining slower than native Base.
Local pressure is faster in 18/20 paired comparisons: 8/10 at 14M and 10/10
at 70M. All quality and logical-sparsity coordinates are retained.

The additional Run046 T2/Ph kappa=.5 point has loss4.838034 and S_model15.426970%.
Its separate-session original port takes1.690154ms versus1.691950ms for native
execution of the same checkpoint. Optimized latency is unmeasured. This point
is recorded in its own appendix table and excluded from both main panels.

## Retained controls and interpretation

`04_controls.py` verifies and recomputes nine groups of three Run042 processes
using the same geometric-mean estimand as the manuscript. These are a separate
GPU session, not additional replicates of Run045. Native Base takes 1.701483ms;
an alternative dense Base with native h/z and the other selected optimized
components takes 1.297441ms. That is a dense implementation improvement without
activation sparsification. Optimized T7/Ph kappa=.5 takes 1.127766ms, 13.078%
below the alternative dense Base, with a different, higher-loss checkpoint.

Native-h/z replacement on the same sparse checkpoint takes 1.365127ms. This
0.237361ms conditional difference includes fusion, layout and inspection as well
as sparse execution; it is not a pure instruction-toggle effect. The custom
no-skip fallback takes 2.704424ms and is not an efficient practical dense
reference. Other replacements are retained in `data/retained-controls.json`.
Their effects are conditional and nonadditive; they must not be summed.

The measured complete-model gain therefore includes both dense optimization
and exploitation of sparse h/z. Cross-recipe timings are not quality matched.
One workload, hardware type and training seed cannot establish that speedup
grows with model scale. The 14M/70M measurements use different sessions, and
the selected implementations differ.

## Figures and tables

`02_figures.py` produces the updated 70M quality/sparsity/latency figure and the
20 matched pressure contrasts. The 70M Base marker and horizontal line use the
new native Base; all other latency points use frozen opt073. Historical
Base/ReLU clipping curves appear only in the quality panel and are unchanged.
Ceilings describe logical opportunity, not a runtime limit. The paired figure
uses Pall minus Ph, so positive latency/loss differences favor local pressure.
Its 14M measurements are unchanged. The author-commented paired-pressure
section is not automatically reinstated.

The compact tables separate additional 14M ablations, matched 14M/70M endpoints
and the unchanged 410M stress test. The matched table gives both original-port
and optimized 70M latency; the unmatched T2/Ph kappa=.5 entry is a dash, linked to its separate appendix table.
A shorter kernel-appendix table selects Base, T2/Ph kappa=.1 and T4/T7/Ph
kappa=.5, with native, port and optimized latency under the same workload.

Historical 14M instruction diagnostics are retained separately in
[Observation 002](002-retained-14m-diagnostics.md). Old 70M bypass counters are
not combined with new-kernel timings.

## Sources and reproduction

- `runs/045-2026-09-20-pythia70m-kernel-grid/` owns the new raw measurements.
- Run043 owns the four retained T2/Ph training checkpoints and ordinary losses.
- Run042 owns the frozen implementation and retained component/dense controls.
- `01_collect.py` joins by checkpoint content hash and retains provenance.
- `02_figures.py`, `03_tables.py` generate the publication artifacts.
- `04_controls.py` reaggregates the retained control timings.
- `05_copy_artifacts.py` copies only approved artifacts and updates source hashes.

`06_verify.py` passes. All manuscript pages were inspected at contact-sheet
scale and changed figure/table pages at higher resolution. Final build details
and the installed PDF hash are retained in `data/build-verification.json`.
