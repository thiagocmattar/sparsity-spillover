# K003: compensated FP32 accumulation diagnostic

Status: CPU/static implementation only; no CUDA qualification or performance
claim. Direct parent is immutable K001, which descends from Run 025 P0 and
official Sakana T2D commit `661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5`.
The upstream MIT license and exact signed-compaction lineage are retained.

## One hypothesis

K001/P0 visit nonzeros in ascending K order and accumulate into one FP32 sum
per output. BF16 multiplication has at most sixteen significant product bits,
so ordinary-range products fit exactly in FP32, but thousands of signed
additions need not. Small products can vanish beside a large partial sum;
cancellation can magnify the remaining absolute error. Subsequent BF16
rounding can move an output to a different representable value, potentially
amplified by model layers or threshold crossings.

K003 changes only accumulation: keep eight FP32 compensation registers per
lane and use ordered Kahan updates. Explicit CUDA round-to-nearest multiply,
subtract and add intrinsics prevent compiler contraction/reassociation from
erasing the correction. Bias is still added in FP32 before final BF16
round-to-nearest. There is no new threshold, discarded nonzero, lower-precision
operand, checkpoint dispatch, weight modification or tolerance relaxation.

This improves accuracy toward the real-valued BF16-operand dot product, **not
necessarily agreement with a particular cuBLAS tensor-core accumulation**.
Kahan can still differ from the dense reference and must pass the same fixed
gates. It is not a proof that accumulation caused the retained-model failure.
The added arithmetic/register state will probably slow SIMT execution; this
candidate first isolates correctness and is not presumed to win a speed search.

## Interface and scope

Same K001 CUDA interface: `extension().linear(x, weight_t, bias_or_empty, out)`.
`Adapter(model, sites=("a", "m", "h", "z"))` supports `native`,
`adapter_dense`, and `k003`. Explicit h-only or other site subsets retain and
report the untouched native dense modules. Coverage reports compensated
accumulation. QK/PV and LM head are unchanged dense operations.

`gpu_probe.py` retains K001's case definitions and imports the unchanged root
numerical gates/timer. Default runs the original 48 shapes/patterns plus 32
scale-1/10 signed-sparse stress cases; every case uses a non-default stream.
Optional `--timing` measures 48 synthetic linear configurations and preserves
native/P0/K003 samples. It does not silently compare K003 against K001: the
parent's matched candidate evaluator owns that direct comparison.

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 python -u gpu_probe.py --output NEW_DIRECTORY
python -u gpu_probe.py --output ANOTHER_DIRECTORY --timing --seconds 600
```

Outputs, failed operands, source identities, numerical metrics and errors
remain durable, with fixed thresholds and a bounded worker. No token cache
or checkpoint is needed for this primitive probe. Full model qualification
is still mandatory, and failures remain failures even if primitive agreement
or the mathematical reference improves.

## CPU tests and interpretation

The inherited compaction/interface tests remain, with a separate arithmetic
test modeling explicit FP32 rounding. A deterministic BF16 cancellation case
loses eight units under naive serial addition and recovers them with Kahan.
Twenty-four signed-sparse synthetic dot products at scales 0.1, 1 and 10
compare aggregate absolute error with an FP64 reference. These tests prove a
possible mechanism and the CPU reference behavior, not compiled CUDA behavior
or the origin of the observed model failure.

No GPU, remote command, paid API or cloud resource was used by this implementation.
Parent agent owns proposal registration, source hashing, fixed-evaluator
qualification and the eventual search decision. K001 remains unchanged.
