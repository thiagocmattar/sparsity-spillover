# Argument chain: hypotheses, not predetermined conclusions

## Paper position

The central paper remains about how training interventions change the
quality/logical-opportunity frontier. This experiment can add a systems case
study: whether a controlled implementation converts some of that opportunity
into measured full-model latency improvements. It must not redefine
`R_model` as speedup or repair the 410M quality result through kernel tuning.

The living manuscript's runtime subsection currently reports a negative 70M
calibration. A successful specialization would qualify its scope, not erase
the historical measurement. The kernel study should remain a bounded results
subsection/appendix unless its evidence justifies a separate systems paper.
Describe agent involvement and human interventions transparently.

The user's revised total ceiling is $40. Preserve correctness, fair baselines,
untuned checkpoints, and reproducibility; narrow the inference about the agent
rather than presenting a cheaper experiment as the original replicated study.
The minimum paper evidence is specified in the experiment plan. Publication
quality refers to measurement and transparent scope, not a guaranteed positive
outcome or a guarantee of acceptance.

## 1. Speedup depends on hardware and model shape

**Testable wording:** under matched precision, checkpoint, workload, and timing
protocol, the sparse/dense speedup and best dispatch configuration can change
between the three Pythia shapes and two GPU architectures.

Evidence: measure the same portable implementation/configuration on RTX 5090
Blackwell and H100 Hopper. There is no funded H100-retuning search.
Evaluate generic versus size-specialized dispatch across all three sizes.
Report latency, packing cost, memory, and executed sparse coverage, not just
different best-case points on different GPUs.

Support: reproducible hardware/shape interactions or changed break-even points.
Refutation/weakening: comparable speedups and a shared configuration throughout
the tested domain. Two GPUs and three sizes cannot establish universality.
Pythia sizes vary width, depth, and head shape together; this is a shape/size
comparison, not an isolated causal experiment on parameter count.

Feasibility: high for measuring dependence; the magnitude is unknown.

## 2. Published kernels need adaptation outside their tested use case

**Testable wording:** these particular Pythia full-sequence workloads require
integration and possibly specialization beyond the upstream demonstrated path.

Sakana's upstream optimized path and our current exact raw-ELL adapter are
distinct baselines. Run 023 calls `dense_to_ell_exact_out` followed by
`ell_spmm_raw_out`; it does not establish the performance of every TwELL
configuration. Begin from pinned official Sakana source and preserve a minimal
Pythia-adaptation baseline before search. Audit the optimized upstream path
before calling the existing adapter the best available Sakana implementation. Record unsupported shapes,
precision, signs, gate behavior, and attention semantics explicitly.

TEAL's reported acceleration targets single-batch decoding; its reference
integration is not a full-sequence Pythia prefill benchmark. An incompatible
drop-in implementation is an engineering observation, not a refutation of its
published speedup. The core study includes a compatibility audit, not a claim
to reproduce TEAL's decoding numbers. A matched decoding/TEAL runtime study
would need a separate design and quality-matched clipping protocol.

Support: documented compatibility gaps plus an ablated, correct adaptation
that changes performance. Weakening: a supported upstream configuration works
well without specialization. Avoid the blanket phrase "published kernels do
not extend"; it is stronger than this study can test.

Primary sources: [Sakana repository](https://github.com/SakanaAI/sparser-faster-llms),
[TEAL repository](https://github.com/FasterDecoding/TEAL).

Feasibility: high for a precise compatibility account, lower for an exhaustive
comparison with all upstream deployment modes; the latter is out of scope.

## 3. Specialization can reduce overhead, but can also hurt

**Testable wording:** packing, indexing, dispatch, and fragmented execution can
outweigh skipped products; specialized kernels may move that break-even point.

Compare minimally adapted official Sakana, strongest matched dense, and the
agent-specialized Sakana descendant. Historical raw-ELL is an additional
diagnostic, not the sole starting point. Include a dense-only sibling with the same
non-sparse fusions so that generic compiler improvements are not credited to
sparsity. Time packing and metadata updates inside the full-model boundary.

Support: improvements survive fresh-process tests, and profiling identifies
reduced costs on the measured critical path. Refutation: isolated kernel wins
disappear in the model, or all sparse routes remain slower than strong dense.
Dense fallback is valid deployment behavior but is not evidence of a sparse
kernel gain when it handles the entire workload.

Feasibility: high for diagnosing overhead. Achieving a meaningful full-model
gain is plausible but unproven, especially for 14M and attention.

## 4. A feedback-driven coding agent can improve this kernel family

**Testable wording:** in this reproducible case study, an agent used measured
feedback to specialize a Sakana-derived implementation and obtained verified
improvements on the stated workload, if the final results support that claim.

Use one preregistered trajectory with at most 40 attempts / 20 RTX-5090 GPU-hours.
Retain the full history, including failures and human assistance. Rebuild the
starting and final versions, repeat timings in fresh processes, ablate winning
changes, and freeze before testing interior thresholds. Do not choose the
nicest trajectory from unreported attempts.

Support: gains over the minimal Sakana adaptation and strong dense references
survive correctness, repeated measurements, and untuned-checkpoint tests.
Weakening/refutation: gains are timer artifacts, generic dense optimizations,
correctness repairs alone, or confined to tuning inputs.

The revised budget removes independent search replicates and the no-feedback
comparator. It therefore does not isolate the causal benefit of the feedback
loop, prove reliable search success, or compare agents with alternative methods.

This does **not** test "new agentic models are extremely good" in general, or
superiority to expert CUDA engineers, earlier coding models, or all autotuners.
Those require additional comparators. The loop combines code changes with
parameter tuning; log which produced the gain. If only tile sweeps help, do
not describe the result as algorithm discovery.

Borrow the small mutable surface, fixed evaluator, and bounded experiment
pattern from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).
That project optimizes training; this plan adapts the pattern to kernel
engineering and does not run its training experiment or indefinite loop.

Feasibility: moderate for the bounded case study; no success guarantee. Pinning
the agent execution route and accounting is an implementation prerequisite.

## 5. A good kernel can make training-induced opportunity useful

**Testable wording:** for a fixed implementation, GPU, size, and workload,
some intervention-generated sparsity patterns may yield greater full-model
speedup, with a positive association between `R_model` and speedup.

Freeze kernels before the interior-kappa evaluation. Show all 36 realizations
on the development GPU and all 18 prespecified H100 sentinels,
including A0 and failures, with loss, canonical `R_model`, executed-kernel
coverage, sparsity distribution, and latency. Analyze A4-OL1 and A7-OL1
threshold series separately within size/GPU/batch before presenting pooled
descriptive regressions. Do not optimize the kernel-selection score for R2.

Support: gains over strong dense baselines survive on untouched checkpoints;
positive within-topology associations are not explained solely by an A0
outlier or a dense-fallback switch; attention contributes in the combined path.
Refutation: high opportunity remains scattered and expensive to execute,
correlation disappears within strata, or the winner helps only tuned endpoints.

The 36-checkpoint core compares trained recipes, not an isolated causal effect
of OL1. A matched no-OL1 comparison would require adding the corresponding
14M controls from Runs 011/013 under a separately confirmed extension. Larger
thresholds may worsen validation loss; a faster but much worse model is not
automatically a useful operating point. The 410M fixed-token quality deficit
remains visible and is not explained away by inference results.

Feasibility: uncertain. This is the most important hypothesis and should be
allowed to fail. Existing six-point regressions are not a scaling law.

## Conditional narrative for later writing

If supported: "Logical sparsity did not translate automatically into runtime
gains. Within a fixed exact-inference workload, a bounded feedback-driven
specialization procedure reduced implementation overhead and improved selected
Pythia operating points. Gains depended on hardware, shape, and the executable
structure of the zeros; `R_model` described opportunity but did not determine
latency. Untuned checkpoints tested whether the gains generalized beyond the
search set."

If unsupported: "Even after a bounded, audited specialization effort, these
unstructured sparsity patterns did not beat optimized dense execution on the
tested workloads. The study separates training-induced logical opportunity
from the stronger requirement of efficiently executable sparsity."

Neither paragraph is a result yet. No result-bearing TeX is authorized here.
