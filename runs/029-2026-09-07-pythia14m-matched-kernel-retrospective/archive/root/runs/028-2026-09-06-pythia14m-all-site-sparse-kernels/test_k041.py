"""Pinned BF16 MMA layout and exact structural-zero product-mask proof."""
import numpy as np


def support_a(matrix):
    mask = 0
    for lane in range(32):
        fragment = [int(matrix[lane // 4 + ((element // 2) % 2) * 8,
            (lane % 4) * 2 + element % 2 + (element // 4) * 8]) for element in range(8)]
        bits = sum(int(((fragment[left] | fragment[right]) & 0x7fff) != 0) << bit
            for left, right, bit in [(0, 2, 0), (1, 3, 1), (4, 6, 8), (5, 7, 9)])
        mask |= bits << ((lane % 4) * 2)
    return mask


def support_b(matrix):
    mask = 0
    for lane in range(32):
        fragment = [int(matrix[(lane % 4) * 2 + element % 2 + (element // 2) * 8,
            lane // 4]) for element in range(4)]
        bits = sum(int((fragment[element] & 0x7fff) != 0) << bit
            for element, bit in enumerate([0, 1, 8, 9]))
        mask |= bits << ((lane % 4) * 2)
    return mask


def test_every_singleton_pair_and_signed_zero():
    a = np.zeros((16, 16), dtype=np.uint16)
    b = np.zeros((16, 8), dtype=np.uint16)
    a_masks, b_masks = [], []
    for row in range(16):
        for inner in range(16):
            a.fill(0x8000); a[row, inner] = 0xbf00
            mask = support_a(a)
            assert mask == 1 << inner
            a_masks.append((mask, inner))
    for inner in range(16):
        for col in range(8):
            b.fill(0x8000); b[inner, col] = 0x3f00
            mask = support_b(b)
            assert mask == 1 << inner
            b_masks.append((mask, inner))
    for ma, ka in a_masks:
        for mb, kb in b_masks:
            assert bool(ma & mb) == (ka == kb)


def test_mixed_supports_exactly_detect_zero_product_tiles():
    rng = np.random.default_rng(2841)
    for probability in [0., .005, .03, .2, 1.]:
        for _ in range(20):
            a = (rng.random((16, 16)) < probability).astype(np.uint16)
            b = (rng.random((16, 8)) < probability).astype(np.uint16)
            expected = bool(np.any(a[:, :, None] * b[None, :, :]))
            assert bool(support_a(a) & support_b(b)) == expected
    a = np.zeros((16, 16), dtype=np.uint16); a[:, ::2] = 1
    b = np.zeros((16, 8), dtype=np.uint16); b[1::2, :] = 1
    assert support_a(a) and support_b(b)
    assert not (support_a(a) & support_b(b))
