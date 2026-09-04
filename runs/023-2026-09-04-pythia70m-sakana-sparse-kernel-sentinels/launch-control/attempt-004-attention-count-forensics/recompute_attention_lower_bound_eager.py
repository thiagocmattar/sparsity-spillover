#!/usr/bin/env python3
"""Run the q/v lower-bound forensic pass in Run-018's eager context."""

from __future__ import annotations

import torch

import recompute_attention_lower_bound as diagnostic


_configure_sdpa = diagnostic.BENCHMARK.configure_model


def configure_eager(model):
    model = _configure_sdpa(model)
    model.config._attn_implementation = "eager"
    return model


diagnostic.BENCHMARK.configure_model = configure_eager


if __name__ == "__main__":
    with torch.inference_mode():
        diagnostic.main()
