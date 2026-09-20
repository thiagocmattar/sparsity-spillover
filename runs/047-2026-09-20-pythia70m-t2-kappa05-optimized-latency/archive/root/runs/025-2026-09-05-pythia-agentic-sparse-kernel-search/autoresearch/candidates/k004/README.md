# K004: exact tile-skip tensor-core linear

K004 is the first candidate specialized from the measured Pythia activation
layout rather than marginal scalar sparsity alone.  It preserves every signed
nonzero.  A 16x32 activation tile is bypassed only when every BF16 value in
that tile is exactly zero; active tiles use Triton `tl.dot` with FP32
accumulation and the original BF16 bias.

The initial shape is BM=16, BN=128, BK=32.  h/z use inline activity detection
because their Pythia-14M output width needs one N tile.  Wider a/m projections
materialize one byte per M/K tile so all N programs reuse the decision.  The
flag kernel, weight transpose, and all dispatch overhead remain in model
timings; only shape-static allocation and weight preparation are setup.

This is an inference-only development candidate.  CPU tests establish
dispatch and exact activity semantics.  CUDA primitive gates, changed-input
tests, real-model full-logit gates, complete validation, and paired speed
measurements are all still required before it can support a claim.
