"""K007 exact-zero inline tensor-core geometry variants."""

import triton
import triton.language as tl


@triton.jit
def inline_tile_skip_linear(
    X,
    W,
    Bias,
    Output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_m = tl.program_id(0)
    tile_n = tl.program_id(1)
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    outputs = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start_k in range(0, K, BLOCK_K):
        inputs = start_k + tl.arange(0, BLOCK_K)
        activations = tl.load(
            X + rows[:, None] * K + inputs[None, :],
            mask=(rows[:, None] < M) & (inputs[None, :] < K),
            other=0,
        )
        active = tl.sum(
            tl.sum((activations != 0).to(tl.int32), axis=1), axis=0
        ) != 0
        if active:
            weights = tl.load(
                W + inputs[:, None] * N + outputs[None, :],
                mask=(inputs[:, None] < K) & (outputs[None, :] < N),
                other=0,
            )
            accumulator += tl.dot(activations, weights)
    if HAS_BIAS:
        bias = tl.load(Bias + outputs, mask=outputs < N, other=0.0)
        accumulator += bias[None, :]
    tl.store(
        Output + rows[:, None] * N + outputs[None, :],
        accumulator.to(tl.bfloat16),
        mask=(rows[:, None] < M) & (outputs[None, :] < N),
    )
