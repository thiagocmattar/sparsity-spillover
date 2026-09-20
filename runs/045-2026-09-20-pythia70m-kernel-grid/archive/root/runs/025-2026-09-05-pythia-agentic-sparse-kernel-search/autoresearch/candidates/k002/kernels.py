"""K002: exact-zero query-tile bypass; dense online-softmax fallback.

New attention code integrated into the Sakana-derived Pythia path. These are
not upstream Sakana attention kernels. No nonzero operand is discarded.
"""
import triton
import triton.language as tl


@triton.jit
def value_prefix_chunks(V, Prefix, Totals,
                        VS0: tl.constexpr, VS1: tl.constexpr,
                        VS2: tl.constexpr, VS3: tl.constexpr,
                        H: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
                        CHUNK: tl.constexpr, NCHUNKS: tl.constexpr):
    chunk = tl.program_id(0)
    bh = tl.program_id(1)
    rows = chunk * CHUNK + tl.arange(0, CHUNK)
    dims = tl.arange(0, D)
    v = tl.load(V + (bh // H) * VS0 + (bh % H) * VS1
                + rows[:, None] * VS2 + dims[None, :] * VS3,
                mask=rows[:, None] < T, other=0).to(tl.float32)
    local_prefix = tl.cumsum(v, axis=0)
    tl.store(Prefix + (bh * T + rows[:, None]) * D + dims[None, :],
             local_prefix, mask=rows[:, None] < T)
    total = tl.sum(v, axis=0)
    tl.store(Totals + (bh * NCHUNKS + chunk) * D + dims, total)


@triton.jit
def zero_query_or_attention(Q, K, V, Prefix, Totals, Output, FastTiles,
                            QS0: tl.constexpr, QS1: tl.constexpr,
                            QS2: tl.constexpr, QS3: tl.constexpr,
                            KS0: tl.constexpr, KS1: tl.constexpr,
                            KS2: tl.constexpr, KS3: tl.constexpr,
                            VS0: tl.constexpr, VS1: tl.constexpr,
                            VS2: tl.constexpr, VS3: tl.constexpr,
                            H: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
                            SCALE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                            CHUNK: tl.constexpr, NCHUNKS: tl.constexpr,
                            CHUNK_PAD: tl.constexpr, QUERY_TILES: tl.constexpr):
    tile = tl.program_id(0)
    bh = tl.program_id(1)
    rows = tile * BM + tl.arange(0, BM)
    dims = tl.arange(0, D)
    q_base = Q + (bh // H) * QS0 + (bh % H) * QS1
    k_base = K + (bh // H) * KS0 + (bh % H) * KS1
    v_base = V + (bh // H) * VS0 + (bh % H) * VS1
    q = tl.load(q_base + rows[:, None] * QS2 + dims[None, :] * QS3,
                mask=rows[:, None] < T, other=0)
    # Scalar condition on the GPU, not a host .item() or a sparsity estimate.
    zero_tile = tl.sum(tl.sum((q != 0).to(tl.int32), axis=1), axis=0) == 0
    tl.store(FastTiles + bh * QUERY_TILES + tile, zero_tile.to(tl.uint8))
    if zero_tile:
        # BM divides CHUNK, so a query tile never crosses a prefix chunk.
        chunk = tile * BM // CHUNK
        earlier = tl.arange(0, CHUNK_PAD)
        chunk_values = tl.load(Totals + (bh * NCHUNKS + earlier[:, None]) * D
                                + dims[None, :],
                                mask=(earlier[:, None] < chunk) & (earlier[:, None] < NCHUNKS),
                                other=0)
        preceding = tl.sum(chunk_values, axis=0)
        local = tl.load(Prefix + (bh * T + rows[:, None]) * D + dims[None, :],
                        mask=rows[:, None] < T, other=0)
        result = (local + preceding[None, :]) / (rows[:, None] + 1).to(tl.float32)
    else:
        maximum = tl.full((BM,), float("-inf"), tl.float32)
        normalizer = tl.zeros((BM,), tl.float32)
        accumulator = tl.zeros((BM, D), tl.float32)
        key_offsets = tl.arange(0, BN)
        # Include every causally valid key, including zero K and zero V rows.
        for start in range(0, tl.minimum((tile + 1) * BM, T), BN):
            keys = start + key_offsets
            k = tl.load(k_base + keys[:, None] * KS2 + dims[None, :] * KS3,
                        mask=keys[:, None] < T, other=0)
            scores = tl.dot(q, tl.trans(k)).to(tl.float32) * SCALE
            allowed = (keys[None, :] <= rows[:, None]) & (keys[None, :] < T)
            scores = tl.where(allowed, scores, float("-inf"))
            next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
            correction = tl.exp(maximum - next_maximum)
            probability = tl.exp(scores - next_maximum[:, None])
            v = tl.load(v_base + keys[:, None] * VS2 + dims[None, :] * VS3,
                        mask=keys[:, None] < T, other=0)
            accumulator = accumulator * correction[:, None]
            accumulator += tl.dot(probability.to(tl.bfloat16), v)
            normalizer = normalizer * correction + tl.sum(probability, axis=1)
            maximum = next_maximum
        result = accumulator / tl.maximum(normalizer[:, None], 1.0e-20)
    tl.store(Output + (bh * T + rows[:, None]) * D + dims[None, :],
             result.to(tl.bfloat16), mask=rows[:, None] < T)
