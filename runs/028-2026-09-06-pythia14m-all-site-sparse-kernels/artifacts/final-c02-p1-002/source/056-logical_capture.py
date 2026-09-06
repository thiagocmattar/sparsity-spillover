"""Observe logical zero-product opportunities at the six declared Pythia operations."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from .metrics import (
    LOGICAL_OPERATIONS,
    linear_zero_product_counts,
    pv_zero_product_counts,
    qk_zero_product_counts,
    summarize_logical_products,
)


class LogicalProductAccumulator:
    """Pool integer numerators and denominators across layers and batches."""

    def __init__(self) -> None:
        self.zero_counts = {name: 0 for name in LOGICAL_OPERATIONS}
        self.product_counts = {name: 0 for name in LOGICAL_OPERATIONS}

    def add(self, name: str, zero_count: int, product_count: int) -> None:
        if name not in self.zero_counts:
            raise KeyError(f"Unknown logical operation: {name}.")
        zero_count, product_count = int(zero_count), int(product_count)
        if product_count <= 0 or not 0 <= zero_count <= product_count:
            raise ValueError(f"Invalid logical counters for {name}.")
        self.zero_counts[name] += zero_count
        self.product_counts[name] += product_count

    def summary(self, *, model: Any, total_input_tokens: int) -> dict[str, Any]:
        missing = [name for name, total in self.product_counts.items() if total <= 0]
        if missing:
            raise RuntimeError("Logical capture missed operations: " + ", ".join(missing))
        output = model.get_output_embeddings()
        weight = getattr(output, "weight", None)
        if weight is None or getattr(weight, "ndim", None) != 2:
            raise ValueError("R_model requires a two-dimensional LM-head weight.")
        vocab_size, hidden_size = (int(value) for value in weight.shape)
        lm_head_products = int(total_input_tokens) * vocab_size * hidden_size
        return summarize_logical_products(
            self.zero_counts,
            self.product_counts,
            lm_head_product_count=lm_head_products,
        )


@contextmanager
def capture_logical_products(
    model: Any,
    *,
    accumulator: LogicalProductAccumulator,
    torch: Any,
    modeling_gpt_neox: Any | None = None,
) -> Iterator[None]:
    """Instrument one uncached eager-attention evaluation, then restore the model.

    This temporarily patches the Transformers GPT-NeoX eager-attention function,
    so it must not overlap another model evaluation in the same process.
    """

    if modeling_gpt_neox is None:
        from transformers.models.gpt_neox import modeling_gpt_neox

    layers = getattr(getattr(model, "gpt_neox", None), "layers", None)
    config = getattr(model, "config", None)
    if layers is None or config is None or not hasattr(config, "_attn_implementation"):
        raise ValueError("Logical capture requires a GPT-NeoX model and attention config.")

    handles: list[Any] = []
    original_eager = modeling_gpt_neox.eager_attention_forward
    original_implementation = config._attn_implementation

    def linear_pre_hook(name: str) -> Any:
        def hook(module: Any, inputs: tuple[Any, ...]) -> None:
            if not inputs:
                raise RuntimeError(f"Logical capture saw no input for {name}.")
            accumulator.add(
                name,
                *linear_zero_product_counts(
                    inputs[0], output_features=int(module.out_features), torch=torch
                ),
            )

        return hook

    def eager_with_counts(
        module: Any,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Any,
        scaling: float,
        dropout: float | int = 0.0,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        context, probabilities = original_eager(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )
        accumulator.add("qk_scores", *qk_zero_product_counts(query, key, torch=torch))
        accumulator.add(
            "probability_value",
            *pv_zero_product_counts(probabilities, value, torch=torch),
        )
        return context, probabilities

    try:
        for index, layer in enumerate(layers):
            attention = getattr(layer, "attention", None)
            mlp = getattr(layer, "mlp", None)
            modules = (
                (getattr(attention, "query_key_value", None), "qkv_projection"),
                (getattr(attention, "dense", None), "attention_output_projection"),
                (getattr(mlp, "dense_h_to_4h", None), "mlp_w1"),
                (getattr(mlp, "dense_4h_to_h", None), "mlp_w2"),
            )
            if any(module is None for module, _name in modules):
                raise ValueError(f"Logical capture cannot resolve all operations in layer {index}.")
            handles.extend(
                module.register_forward_pre_hook(linear_pre_hook(name))
                for module, name in modules
            )
        modeling_gpt_neox.eager_attention_forward = eager_with_counts
        config._attn_implementation = "eager"
        yield
    finally:
        config._attn_implementation = original_implementation
        modeling_gpt_neox.eager_attention_forward = original_eager
        for handle in handles:
            handle.remove()
