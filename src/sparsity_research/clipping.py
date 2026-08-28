"""One complete-validation point for a post-hoc clipping frontier."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from .capture import ActivationCapture
from .evaluation import evaluate_complete_blocks
from .logical_capture import LogicalProductAccumulator, capture_logical_products
from .metrics import ActivationAccumulator


def evaluate_clipping_point(
    *,
    model: Any,
    tokens: Any,
    block_size: int,
    batch_size: int,
    device: Any,
    torch: Any,
    np: Any,
    autocast_dtype: Any | None,
    clipping: dict[str, Any],
    measure_logical_products: bool = False,
    modeling_gpt_neox: Any | None = None,
) -> dict[str, Any]:
    """Evaluate one cutoff and return count-first activation/logical evidence."""

    sites = list(clipping.get("sites", []))
    if not clipping.get("enabled", False) or not sites:
        raise ValueError("A clipping point requires enabled clipping and explicit sites.")
    activation = ActivationAccumulator(thresholds=(0.0,))
    logical = LogicalProductAccumulator()
    logical_context = (
        capture_logical_products(
            model,
            accumulator=logical,
            torch=torch,
            modeling_gpt_neox=modeling_gpt_neox,
        )
        if measure_logical_products
        else nullcontext()
    )

    with ActivationCapture(
        model,
        sites,
        torch=torch,
        clipping=clipping,
    ) as capture:
        def consume_batch(_output: Any, _sequences: int) -> None:
            activation.update(capture.activations, torch=torch)
            capture.clear()

        with logical_context:
            validation = evaluate_complete_blocks(
                model=model,
                tokens=tokens,
                block_size=block_size,
                batch_size=batch_size,
                device=device,
                torch=torch,
                np=np,
                autocast_dtype=autocast_dtype,
                after_batch=consume_batch,
            )

    result: dict[str, Any] = {
        "clipping": dict(clipping),
        "validation": validation,
        "activations": activation.rows(),
        "activations_by_site": activation.pooled_by_site(),
    }
    if measure_logical_products:
        result["logical_products"] = logical.summary(
            model=model,
            total_input_tokens=int(validation["input_tokens"]),
        )
    return result
