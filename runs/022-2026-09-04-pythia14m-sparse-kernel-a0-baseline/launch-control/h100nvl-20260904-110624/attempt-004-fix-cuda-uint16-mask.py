"""Attempt-local CUDA portability fix for Run 022's ELL contract guard.

CUDA does not implement boolean advanced indexing directly on UInt16 tensors.
Casting the whole index tensor to Int32 before applying the same boolean mask
preserves the checked values and the out-of-range rejection invariant.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


TARGET = Path(
    "/workspace/sparsity-spillover/"
    "runs/022-2026-09-04-pythia14m-sparse-kernel-a0-baseline/benchmark_core.py"
)
REPLACEMENTS = {
    "indices[valid].to(torch.int32).max().item()":
        "indices.to(torch.int32)[valid].max().item()",
    "indices[valid].to(torch.long)": "indices.to(torch.long)[valid]",
}

text = TARGET.read_text(encoding="utf-8")
for unsupported, supported in REPLACEMENTS.items():
    if text.count(unsupported) == 1:
        text = text.replace(unsupported, supported)
    elif text.count(supported) != 1:
        raise RuntimeError(f"The expected UInt16 expression was not found: {unsupported}")

with TARGET.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"updated={TARGET}")
print(f"benchmark_core_sha256={hashlib.sha256(TARGET.read_bytes()).hexdigest()}")
