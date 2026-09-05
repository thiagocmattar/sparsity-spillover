#!/usr/bin/env python3
"""Run the frozen exact-ELL preflight over every Pythia-410M shape."""

from _reuse_run023 import load_run023_module


_impl = load_run023_module("_run023_remote_preflight", "02_remote_preflight.py")
_impl.SHAPES = (
    ("run022_n128_regression", 256, 512, 128),
    ("attention_output_projection", 256, 1024, 1024),
    ("qkv_projection", 256, 1024, 3072),
    ("mlp_w1", 256, 1024, 4096),
    ("mlp_w2", 256, 4096, 1024),
    ("attention_qk_q_left", 2048, 64, 2048),
    ("attention_pv_vt_left", 64, 2048, 2048),
)


if __name__ == "__main__":
    _impl.main()

