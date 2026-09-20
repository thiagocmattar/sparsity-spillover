"""Deterministic complete-block causal-LM evaluation."""

from __future__ import annotations

from contextlib import nullcontext
from collections.abc import Callable
from typing import Any

from .data import complete_block_starts


def evaluate_complete_blocks(
    *,
    model: Any,
    tokens: Any,
    block_size: int,
    batch_size: int,
    device: Any,
    torch: Any,
    np: Any,
    autocast_dtype: Any | None,
    after_batch: Callable[[Any, int], None] | None = None,
) -> dict[str, Any]:
    starts = complete_block_starts(len(tokens), block_size)
    if not starts or batch_size <= 0:
        raise ValueError("Evaluation requires complete blocks and a positive batch size.")
    weighted_loss = 0.0
    sequences = batches = 0
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            selected = starts[offset : offset + batch_size]
            array = np.stack([tokens[start : start + block_size] for start in selected])
            input_ids = torch.as_tensor(array, dtype=torch.long, device=device)
            context = (
                torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype is not None and device.type == "cuda"
                else nullcontext()
            )
            with context:
                output = model(input_ids=input_ids, labels=input_ids)
            if not bool(torch.isfinite(output.loss.detach()).item()):
                raise RuntimeError("Non-finite validation loss.")
            if after_batch is not None:
                after_batch(output, len(selected))
            weighted_loss += float(output.loss.detach().cpu()) * len(selected)
            sequences += len(selected)
            batches += 1
    tail = len(tokens) - len(starts) * block_size
    return {
        "loss": weighted_loss / sequences,
        "batches": batches,
        "sequences": sequences,
        "input_tokens": sequences * block_size,
        "source_tokens": len(tokens),
        "excluded_tail_tokens": tail,
        "complete_block_coverage": True,
    }
