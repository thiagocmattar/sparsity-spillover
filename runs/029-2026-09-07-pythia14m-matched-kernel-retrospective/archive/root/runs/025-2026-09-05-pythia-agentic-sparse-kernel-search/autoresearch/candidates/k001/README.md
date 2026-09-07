# K001: fuse exact register compaction into Sakana-derived T2D

Status: proposed performance candidate, CPU/static tests only. No GPU compile,
speedup, numerical qualification, or paper-level conclusion is claimed.

Parent: Run 025 P0 (`kernels/twell_pythia.cu`), itself the documented exact
signed Pythia adaptation of official Sakana commit
`661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5`. Upstream MIT license continues to apply.
Historical P0 and the evaluator remain unchanged. This source is not a newly
qualified P0 or an unchanged official kernel.

## Hypothesis and surgical change

P0 materializes a full-capacity padded 288-word representation per 256 inputs,
then launches a second kernel to consume it. At small projection widths this
extra launch and write/read traffic can dominate sparse arithmetic. K001
eliminates that intermediate: each warp ballots 32 signed BF16 inputs, visits
every surviving lane in ascending order, and broadcasts its value directly
into the same eight-output FP32 FMA loop. No nonzero is dropped and no row cap
exists. Four independent row warps share a thread block to reduce the
one-warp-block scheduling overhead. Aligned widths use explicit 128-bit BF16
weight loads; ragged widths use predicated scalar loads.

The trade-off is repeated input scanning for different 256-output tiles. Large
output widths and moderate sparsity may lose to P0 or dense GEMM. Dense A0 is
not expected to benefit. This is a measured hypothesis, not a promised gain.

## Mathematical and integration contract

- Same BF16 inputs, static transposed BF16 weights, FP32 FMA accumulation in
  ascending K order, fused FP32 bias addition and final BF16 round-to-nearest.
- No threshold/mask, nonlinear gate, residual, model weight or checkpoint changes.
- Every lane participates in full-mask shuffles, including N=128 tails.
- Same-device checks, current CUDA stream, device guard and launch error checks.
- `extension().linear(x, weight_t, bias_or_empty, out)` takes contiguous
  BF16 `[M,K]`, `[K,N]`, `[N]` or `[0]`, and `[M,N]` tensors.
- `Adapter(model, sites=("a", "m", "h", "z"))` accepts `native`,
  `adapter_dense`, and `k001` modes. For an h-only component test use
  `sites=("h",)`. Unselected sites use the actual original Linear module and
  are listed in `coverage()` as dense fallbacks. Dispatch uses only explicit
  architecture-site flags, never checkpoint identity or tuned validation scores.
- QK, softmax, PV and LM head remain dense. This candidate addresses only the
  four linear operations, not the overall study's attention obligation.

There is no materialized packed tensor. Legacy P0 wire-format equality is
therefore inapplicable, not a fake pass. `compaction_reference()` independently
tests exact signed values and sorted indices for dense, zero, mixed and ragged
inputs. The unchanged numerical and memory gates still apply to real CUDA.

## Numerical risk and first GPU test

P0 failed A1-H's full-logit tolerance despite passing its primitive cases. K001
preserves P0's mathematical ordering and fused bias; it is **not a proposed
correctness repair**. First compare K001 directly with P0 on development
activations to detect any implementation divergence, then apply the unchanged
primitive gates and sanitizer, followed by the same retained-model numerical
gates. Never spend a full timing search on a configuration that fails quality.
If all-four-site integration repeats P0's failure, a predeclared h-only
component variant may be tested and must be labeled with its reduced sparse
coverage; it does not qualify an all-four-site result.

Compaction tests and CPU mathematical wrapper tests do not exercise compiled
CUDA, vector alignment, synchronization, tensor-core-vs-SIMT round-off, or
real model loss. They are preflight checks, not substitution for those gates.

## Local verification command

```powershell
.venv/Scripts/python.exe -m pytest runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/autoresearch/candidates/k001/test_candidate.py -q
```

No cloud resource, local GPU workload, model inference run or paid API was
launched during implementation. Parent agent owns source hashing, proposal
registration, evaluator sealing, trial execution, cost accounting and decision.

Local outcome: **15 CPU/static tests passed in 0.99 seconds**. CUDA has not been
compiled or executed for this candidate.

## GPU probe (execution delegated to the parent agent)

`gpu_probe.py --output NEW_DIRECTORY` compiles K001 and runs the original
48 primitive cases in their original seed/order, all on non-default streams,
plus 32 signed-sparse activation stress cases at scales 1 and 10. The original
cases use activation scale 0.1; weights/bias remain scale 0.1. Numerical gates
are imported unchanged from the root evaluator/config. Default duration cap
is 600 seconds and needs the external process watchdog for hung CUDA calls.

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 python -u gpu_probe.py --output NEW_DIRECTORY
python -u gpu_probe.py --output ANOTHER_NEW_DIRECTORY --timing --seconds 600
```

Optional `--timing` adds 48 synthetic M=2048 shape/zero-fraction cases across
14M/70M/410M's four projection sites. Zero fractions are explicitly 0, 0.5,
0.9 and 0.99 (not ambiguous GPU occupancy); realized integer counts are saved.
The dense PyTorch, original P0 and K001 calls use rotating, paired timing with
four inputs, three passes and two warmups by default, preserving raw host/CUDA
samples. A configuration failing a numerical gate is not timed. These are
linear-operation measurements, not full-model speedups or measured `R_model`.

Each primitive saves actual/reference tensor outputs; failures also retain
input/weight/bias tensors. JSON summaries, source hashes, durable event logs,
fixed gates, environment, and tracebacks preserve failures and incomplete work.
No checkpoint or token cache is required.
