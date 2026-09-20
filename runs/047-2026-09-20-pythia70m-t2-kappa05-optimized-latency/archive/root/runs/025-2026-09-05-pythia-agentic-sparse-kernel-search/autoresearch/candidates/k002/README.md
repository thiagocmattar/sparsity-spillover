# K002: exact zero-query attention shortcut

Status: **implemented, CPU mathematical tests passed; CUDA unqualified**.
This is a proposed attention component of the Sakana-derived model candidate,
not an upstream Sakana attention kernel and not an independently demonstrated
speedup. It changes no weights, gates, thresholds, RoPE, or causal semantics.

## Hypothesis and implementation

Actual retained 410M A7-OL1(.5) native-BF16 histograms motivate a whole-query-row
shortcut: layer 7 has 11,063,517 / 11,075,584 exactly zero Q head-rows (99.891%).
Its V rows are 99.966% exactly zero. These are old H100 full-validation counts,
not measurements of this candidate or proof of temporal query-tile occupancy.
The 70M counts have weaker row sparsity and were collected at batch 32, so they
do not establish the new batch-one opportunity. Measure actual tile flags.

For a zero Q row at position i, every causally valid score is zero. Consequently
the output is exactly `sum(V[0:i+1]) / (i+1)` in real arithmetic, independent
of K. Zero K/V positions still contribute to the denominator. The code never
drops keys from normalization or treats zero Q as zero context.

`kernels.py` implements two launches:

1. Gated V's FP32 chunk-local inclusive prefixes and chunk totals, chunk size 128.
2. Query tiles of 16 (optional 32) rows. A GPU scalar test checks every actual Q
   coordinate. All-zero tiles use the stored prefix plus preceding chunk totals
   to return causal prefix means. Mixed/nonzero tiles run causal online-softmax
   attention, BF16 Q/K/V and softmax multiplicands, FP32 score/normalization/output
   accumulators, and BF16 output. Every causally valid key remains present.

All prefix construction, branch detection, and output writes happen inside the
candidate call. There is no CPU `.item()` decision, cached activation, dropped
nonzero, or full dense attention computed then overwritten. Shape-static scratch
may be allocated before timing, as for other cached workspaces, but every value
is recomputed. The candidate always writes one uint8 fast-path flag per query
tile; reading those flags to the host is diagnostic-only and outside timing.

`attention.py` exposes `attention(q,k,v,scale=...,workspace=...)`, `Workspace`,
and a Transformers-compatible `interface`. The direct output aliases the
workspace and must be consumed/cloned before its next use. Shapes are
`[B,H,T,D]`, D32/D64, T<=2048, BF16, positive strides. The interface returns
`[B,T,H,D]` context and `None` attention weights. It must be registered as a
distinct attention interface by the parent evaluator and called **after** the
existing canonical post-RoPE Q/K/V gates. Its parent attention forward still
applies the z gate and W_o, including bias. Unsupported masks/cache shapes or
dropout use the correct dense SDPA fallback; requesting materialized attention
weights raises. The approved full-sequence workload uses the sparse path.

No per-checkpoint or validation-input dispatch rule is built in. A future
selection to use K002 only for architectures/topologies with measured development
benefit must be frozen and recorded by the parent search, with dense fallback
coverage and a dense-only sibling. Prefix overhead makes a regression quite
plausible when zero query tiles are rare. A high zero-row fraction does not
guarantee all-zero temporal tiles or end-to-end gains.

## Qualification and measurements

The original Run 025 numerical bounds remain unchanged. Mathematical exactness
does not imply identical BF16 rounding: prefix summation differs from fused SDPA
and online-softmax reduction order. The candidate needs primitive tests against
both native BF16 SDPA and FP32 math SDPA, then unchanged full-logit and complete
338-block loss gates for every supported model claim. A positive timing against
a different attention backend is not a matched sparse-speedup result.

On the already provisioned GPU, from the repository root:

```bash
timeout --signal=TERM --kill-after=30s 900s "$VENV/bin/python" -u \
  runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/autoresearch/candidates/k002/qualify.py \
  --attempt cuda-001 --max-length 2048 --block-m 16
```

Run with compute-sanitizer as the owner deems appropriate. The script covers
36 cases (D32/64 x T17/129/2048 x six patterns): signed dense, all-zero Q/K/V,
mixed zero-query tiles, and one active row among zero queries. Operands are
strided along D; a non-default stream is exercised. Each case then reuses
scratch with changed Q/V. T129 tests also capture and replay a CUDA Graph after
changing the captured zero-query pattern into nonzero queries. It preserves
per-case flags, fixed numerical results, source hashes, and failures. These
elapsed times include reference work/compilation and are **not kernel latency**.

CPU tests (`test_attention.py`) cover the real-arithmetic identity with signed
inputs; zero-Q/K/V; preservation of zero-V normalization mass; causality; dense
fallback masks/layout; contract rejection; and scratch byte accounting.
**10 tests passed** in 1.00 second. No CUDA/GPU result is asserted here.

The standard online-softmax algorithm and Triton scan API were checked against
the primary [Triton attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
and [inclusive cumsum API](https://triton-lang.org/main/python-api/generated/triton.language.cumsum.html).
The implementation is new run-local code, not copied tutorial benchmark results.
