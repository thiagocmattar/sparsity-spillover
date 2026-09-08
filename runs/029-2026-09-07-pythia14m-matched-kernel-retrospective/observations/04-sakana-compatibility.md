# Sakana reference and the matched Pythia comparator

## Question

Can the published Sakana implementation serve as an unchanged, matched
Pythia14M/RTX5090 full-model baseline? What exactly does the P0 comparison mean?

## Method and coverage

Script09 downloads eight exact Git blobs from upstream commit
661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5, verifies Git blob SHA1 against the pinned
tree, and records byte counts/SHA256 in `provenance/sakana-upstream.json`.
The retained T2D source is byte-identical to Run025's reference. The MIT license
is included. This is a source-level compatibility audit, not a new upstream
hardware benchmark or an assertion that all Sakana implementations are unusable.

## Findings

The [official reference](https://github.com/SakanaAI/sparser-faster-llms/tree/661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5)
describes H100-targeted TwELL implementations and SparseLM checkpoints. Its
[D2T producer](https://github.com/SakanaAI/sparser-faster-llms/blob/661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5/custom_models/twell_modules/matmul_d2t.cu)
uses Hopper WGMMA and positive-output sparse packing. The non-gated
[T2D dispatch](https://github.com/SakanaAI/sparser-faster-llms/blob/661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5/custom_models/twell_modules/matmul_t2d.cu)
restricts output width to2048 and input feature widths to5632 or8192. These
interfaces do not implement Pythia14M's signed activations and128/512-wide
projections unchanged. A shape/sign/hardware port would be a different artifact.

The upstream [benchmark helper](https://github.com/SakanaAI/sparser-faster-llms/blob/661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5/benchmark_base.py)
selects a core model without the LM head, unlike Run029's full50304-logit timer.
Therefore neither an upstream headline ratio nor the earlier H100 positive
control is substituted for a matched full-model speedup here.

Run029 benchmarks the **existing Sakana-derived Pythia adapter P0**, frozen
from Run025. Its signed-exact packing, full-capacity storage, predicated
shape tails, current-stream launch and accumulation adaptations are retained,
not credited as new retrospective discoveries. Its topology-active a/m/h/z
projection filter follows the historical14M probes; inactive sites are native,
and QK/PV remain dense SDPA. Thus A0 has native projection fallback. Every
recurring packing operation at enabled sites is inside the timed graph replay.

## Caption terminology and limits

Use "Sakana-derived Pythia adapter (P0)", never "unchanged Sakana kernel" or
"published Sakana full-model speedup". P0 performance and numerical
qualification are measured by the same native-graph denominator, inputs,
hardware, precision and full-validation gates as the other candidates; results
belong to the final matched observation. This study does not rank the original
SparseLM/H100 system against Pythia14M/RTX5090 and does not develop a new
optimized Blackwell port. No manuscript text changes are made.

Source script: `09_sakana_reference.py`. Supporting historical audit:
`archive/root/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/SOURCE_AUDIT.md`.

## Completed matched evaluation

P0 was evaluated on all35 checkpoints in three fresh processes each. It
qualifies on5/35 under the declared full-validation logit gates, with a0.867570x
geomean on that qualified subset;30 fail elementwise logit bounds despite finite
outputs and passing relative-L2/pooled-loss checks. Its qualified c30 speedup is
0.817961x. [Observation03](03-matched-rmodel-speedup.md) retains the exact
qualification, checkpoint identities and comparison limits. This does not
constitute a negative benchmark of the unchanged published SparseLM system.
