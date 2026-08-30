"""Run 004 complete-validation activation and logical-product diagnostics."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from sparsity_research.capture import ActivationCapture
from sparsity_research.ceilings import architecture_ceiling
from sparsity_research.evaluation import evaluate_complete_blocks
from sparsity_research.logical_capture import LogicalProductAccumulator, capture_logical_products
from sparsity_research.metrics import ActivationAccumulator

from optimizer_boundary import recipe_attention_context
from run_config import mapping, require_validation_coverage


class AttentionOutputCapture:
    """Capture output of W_o, equal under zero dropout to pre-residual attention output."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.activations: dict[str, Any] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> "AttentionOutputCapture":
        layers = getattr(getattr(self.model, "gpt_neox", None), "layers", None)
        if layers is None:
            raise ValueError("Attention-output capture requires GPT-NeoX layers.")
        for index, layer in enumerate(layers):
            dense = getattr(getattr(layer, "attention", None), "dense", None)
            if dense is None:
                raise ValueError(f"Could not resolve attention.dense in layer {index}.")
            name = f"attention_output.layer_{index}"
            self._handles.append(dense.register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, *_args: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def clear(self) -> None:
        self.activations.clear()

    def _hook(self, name: str) -> Callable[[Any, Any, Any], None]:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not hasattr(output, "detach"):
                raise TypeError(f"Expected tensor at {name}.")
            self.activations[name] = output

        return hook


def activation_diagnostic_validation(
    *,
    model: Any,
    tokens: Any,
    config: Mapping[str, Any],
    torch: Any,
    np: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    diagnostics = mapping(config, "diagnostics")
    requested = list(diagnostics["activation_sites"])
    canonical = [site for site in requested if site != "attention_output"]
    accumulator = ActivationAccumulator(
        tuple(float(value) for value in diagnostics["near_zero_thresholds"])
    )
    device = torch.device("cuda")
    with ActivationCapture(model, canonical, torch=torch) as capture, AttentionOutputCapture(model) as output_capture:

        def observe(_output: Any, _batch_sequences: int) -> None:
            expected = len(model.gpt_neox.layers) * len(requested)
            combined = {**capture.activations, **output_capture.activations}
            if len(combined) != expected:
                raise RuntimeError(
                    f"Activation diagnostic captured {len(combined)} tensors, expected {expected}."
                )
            accumulator.update(combined, torch=torch)
            capture.clear()
            output_capture.clear()

        with recipe_attention_context(torch, device):
            coverage = evaluate_complete_blocks(
                model=model,
                tokens=tokens,
                block_size=int(mapping(config, "data")["sequence_length"]),
                batch_size=int(mapping(config, "validation")["batch_size"]),
                device=device,
                torch=torch,
                np=np,
                autocast_dtype=torch.float16,
                after_batch=observe,
            )
    require_validation_coverage(coverage, config)
    statistics = {
        "site_definition": {
            "attention_output": "output of attention.dense (W_o), immediately before zero dropout and residual addition"
        },
        "rows": accumulator.rows(),
        "pooled_by_site": accumulator.pooled_by_site(),
    }
    _require_activation_rows(statistics, requested, layers=len(model.gpt_neox.layers))
    return coverage, statistics


def logical_product_validation(
    *,
    model: Any,
    tokens: Any,
    config: Mapping[str, Any],
    torch: Any,
    np: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    accumulator = LogicalProductAccumulator()
    device = torch.device("cuda")
    with capture_logical_products(model, accumulator=accumulator, torch=torch):
        coverage = evaluate_complete_blocks(
            model=model,
            tokens=tokens,
            block_size=int(mapping(config, "data")["sequence_length"]),
            batch_size=int(mapping(config, "diagnostics")["logical_product_batch_size"]),
            device=device,
            torch=torch,
            np=np,
            autocast_dtype=torch.float16,
        )
    require_validation_coverage(coverage, config)
    logical = accumulator.summary(model=model, total_input_tokens=int(coverage["input_tokens"]))
    architecture = architecture_ceiling(
        str(model.config.topology_id),
        layers=int(model.config.num_hidden_layers),
        hidden_size=int(model.config.hidden_size),
        ffn_size=int(model.config.intermediate_size),
        sequence_length=int(mapping(config, "data")["sequence_length"]),
        vocabulary_size=int(model.config.vocab_size),
    )
    return coverage, logical, architecture


def _require_activation_rows(
    statistics: Mapping[str, Any], requested: list[str], *, layers: int
) -> None:
    rows = statistics.get("rows")
    pooled = statistics.get("pooled_by_site")
    if not isinstance(rows, list) or len(rows) != len(requested) * layers:
        raise RuntimeError("Activation diagnostic layer rows are incomplete.")
    if not isinstance(pooled, list) or {row["name"] for row in pooled} != set(requested):
        raise RuntimeError("Activation diagnostic pooled rows are incomplete.")
    if any(int(row["nonfinite"]) != 0 for row in rows):
        raise RuntimeError("Activation diagnostic contains non-finite values.")

