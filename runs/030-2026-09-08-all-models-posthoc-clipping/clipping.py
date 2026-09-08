"""Run-local retained TEAL helpers, copied from Run 019; see helper-provenance.json.

Only passive near-zero diagnostics are extended to 0.001 and 0.01.
The calibration, clipping, loss and count algorithms are unchanged.
"""
from contextlib import contextmanager
import math
from time import perf_counter
from typing import Any, Callable, Iterator, Mapping
SITES = ('a','m','h','z')
DIAGNOSTIC_SITES = ('a','m','h','q_post','k_post','v','z','attention_output')
EXPECTED_VALIDATION = {'sequences':338,'input_tokens':692224,'excluded_tail_tokens':1444,'complete_block_coverage':True}

def empirical_thresholds(values: Any, targets: tuple[float, ...], *, np: Any) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size <= 0 or not bool(np.isfinite(flat).all()) or bool((flat < 0).any()):
        raise ValueError("Calibration requires finite absolute activation values.")
    target_indices = {
        target: None if target == 0.0 else max(0, math.ceil(target * flat.size) - 1)
        for target in targets
    }
    kth = sorted({index for index in target_indices.values() if index is not None})
    partitioned = np.partition(flat, kth) if kth else flat
    rows = []
    for target in targets:
        index = target_indices[target]
        threshold = 0.0 if index is None else float(partitioned[index])
        hits = int(np.count_nonzero(flat <= threshold))
        rows.append(
            {
                "target_sparsity": target,
                "threshold": threshold,
                "calibration_zero_count": hits,
                "calibration_total": int(flat.size),
                "calibration_fraction": hits / int(flat.size),
            }
        )
    return {
        "total": int(flat.size),
        "natural_exact_zero_count": int(np.count_nonzero(flat == 0.0)),
        "targets": rows,
    }


def _module_map(model: Any) -> dict[str, tuple[Any, str]]:
    layers = getattr(getattr(model, "gpt_neox", None), "layers", None)
    if layers is None or len(layers) <= 0:
        raise ValueError("TEAL module mapping requires GPT-NeoX layers.")
    modules = {}
    for layer_index, layer in enumerate(layers):
        resolved = {
            "a": (layer.input_layernorm, "output"),
            "m": (layer.post_attention_layernorm, "output"),
            "h": (layer.mlp.act, "output"),
            "z": (layer.attention.dense, "input"),
        }
        for site, module_and_port in resolved.items():
            modules[f"{site}.layer_{layer_index}"] = module_and_port
    if len(modules) != len(layers) * len(SITES):
        raise ValueError("TEAL mapping did not cover four sites in every model layer.")
    return modules


def _calibrate(
    model: Any,
    train_tokens: Any,
    *,
    targets: tuple[float, ...],
    blocks: int,
    block_size: int,
    device: Any,
    torch: Any,
    np: Any,
) -> dict[str, Any]:
    modules = _module_map(model)
    captured: dict[str, list[Any]] = {name: [] for name in modules}
    handles = []

    def hook(name: str, port: str):
        def capture(_module: Any, inputs: Any, output: Any = None) -> None:
            value = _first_tensor(output if port == "output" else inputs)
            captured[name].append(value.detach().abs().float().cpu().reshape(-1).numpy())

        return capture

    for name, (module, port) in modules.items():
        handles.append(
            module.register_forward_hook(hook(name, port))
            if port == "output"
            else module.register_forward_pre_hook(hook(name, port))
        )
    started = perf_counter()
    model.eval()
    try:
        with torch.no_grad():
            for block_index in range(blocks):
                start = block_index * block_size
                array = np.asarray(train_tokens[start : start + block_size], dtype=np.int64)[None, :]
                input_ids = torch.as_tensor(array, dtype=torch.long, device=device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model(input_ids=input_ids)
    finally:
        for handle in handles:
            handle.remove()
    statistics = {}
    for name in sorted(captured):
        if len(captured[name]) != blocks:
            raise ValueError(f"Calibration capture is incomplete for {name}.")
        values = np.concatenate(captured[name])
        statistics[name] = empirical_thresholds(values, targets, np=np)
    return {
        "blocks": blocks,
        "input_tokens": blocks * block_size,
        "source_order_block_indices": list(range(blocks)),
        "wall_seconds": perf_counter() - started,
        "statistics": statistics,
    }


@contextmanager
def threshold_hooks(
    model: Any, thresholds: Mapping[str, float]
) -> Iterator[dict[str, Any]]:
    modules = _module_map(model)
    if set(modules) != set(thresholds):
        raise ValueError("Thresholds must exactly cover every TEAL site-layer tensor.")
    handles = []
    activations: dict[str, Any] = {}

    def output_hook(name: str):
        threshold = _valid_threshold(thresholds[name], name)

        def clip(_module: Any, _inputs: Any, output: Any) -> Any:
            value = _first_tensor(output)
            clipped = value.masked_fill(value.detach().abs() <= threshold, 0.0)
            activations[name] = clipped
            if value is not output:
                raise TypeError("TEAL output hook requires a direct tensor output.")
            return clipped

        return clip

    def input_hook(name: str):
        threshold = _valid_threshold(thresholds[name], name)

        def clip(_module: Any, inputs: Any) -> Any:
            value = _first_tensor(inputs)
            clipped = value.masked_fill(value.detach().abs() <= threshold, 0.0)
            activations[name] = clipped
            if not isinstance(inputs, tuple) or not inputs or inputs[0] is not value:
                raise TypeError("TEAL input hook requires the first tuple element to be the tensor.")
            return (clipped, *inputs[1:])

        return clip

    try:
        for name, (module, port) in modules.items():
            handles.append(
                module.register_forward_hook(output_hook(name))
                if port == "output"
                else module.register_forward_pre_hook(input_hook(name))
            )
        yield activations
    finally:
        for handle in handles:
            handle.remove()


def _evaluate_point(
    model: Any,
    validation_tokens: Any,
    thresholds: Mapping[str, float],
    *,
    block_size: int,
    batch_size: int,
    device: Any,
    torch: Any,
    np: Any,
) -> dict[str, Any]:
    from sparsity_research.capture import ActivationCapture
    from sparsity_research.evaluation import evaluate_complete_blocks
    from sparsity_research.logical_capture import LogicalProductAccumulator, capture_logical_products
    from sparsity_research.metrics import ActivationAccumulator

    activation = ActivationAccumulator((0.0, 0.001, 0.01))
    logical = LogicalProductAccumulator()
    expected_clipped = len(_module_map(model))
    expected_diagnostics = len(model.gpt_neox.layers) * len(DIAGNOSTIC_SITES)
    started = perf_counter()
    # Register clipping first. The passive downstream hooks then observe the
    # executed clipped forward rather than the source checkpoint activations.
    with threshold_hooks(model, thresholds) as clipped, ActivationCapture(
        model, ["q_post", "k_post", "v"], torch=torch
    ) as downstream, AttentionOutputCapture(model) as output_capture:

        def consume(_output: Any, _sequences: int) -> None:
            if len(clipped) != expected_clipped:
                raise ValueError("TEAL clipping did not capture every matrix input.")
            combined = {
                **clipped,
                **downstream.activations,
                **output_capture.activations,
            }
            if len(combined) != expected_diagnostics:
                raise ValueError(
                    f"TEAL diagnostics captured {len(combined)} tensors, "
                    f"expected {expected_diagnostics}."
                )
            activation.update(combined, torch=torch)
            clipped.clear()
            downstream.clear()
            output_capture.clear()

        with capture_logical_products(model, accumulator=logical, torch=torch):
            coverage = evaluate_complete_blocks(
                model=model,
                tokens=validation_tokens,
                block_size=block_size,
                batch_size=batch_size,
                device=device,
                torch=torch,
                np=np,
                autocast_dtype=torch.float16,
                after_batch=consume,
            )
    if any(coverage.get(key) != value for key, value in EXPECTED_VALIDATION.items()):
        raise ValueError("TEAL complete-validation coverage changed.")
    layer_rows = activation.rows()
    pooled = activation.pooled_by_site()
    if (
        len(layer_rows) != expected_diagnostics
        or {row["name"] for row in pooled} != set(DIAGNOSTIC_SITES)
    ):
        raise ValueError("TEAL activation rows are incomplete.")
    logical_summary = logical.summary(model=model, total_input_tokens=int(coverage["input_tokens"]))
    operations = logical_summary["per_operation"].values()
    if sum(int(row["zero_product_count"]) for row in operations) != int(logical_summary["block_zero_product_count"]):
        raise ValueError("TEAL logical zero-product counts do not reconcile.")
    if sum(int(row["product_count"]) for row in logical_summary["per_operation"].values()) != int(logical_summary["block_product_count"]):
        raise ValueError("TEAL logical product denominators do not reconcile.")
    seconds = perf_counter() - started
    return {
        "validation": coverage,
        "thresholds_by_site_layer": dict(sorted(thresholds.items())),
        "activation_rows": layer_rows,
        "activations_by_site": pooled,
        "logical_products": logical_summary,
        "evaluation_seconds": seconds,
        "input_tokens_per_second": int(coverage["input_tokens"]) / seconds,
    }


def nondominated(rows: list[Mapping[str, Any]]) -> list[bool]:
    result = []
    for index, row in enumerate(rows):
        loss = float(row["validation"]["loss"])
        opportunity = float(row["logical_products"]["R_model"])
        result.append(
            not any(
                other_index != index
                and float(other["validation"]["loss"]) <= loss
                and float(other["logical_products"]["R_model"]) >= opportunity
                and (
                    float(other["validation"]["loss"]) < loss
                    or float(other["logical_products"]["R_model"]) > opportunity
                )
                for other_index, other in enumerate(rows)
            )
        )
    return result


def _thresholds_for_target(calibration: Mapping[str, Any], target: float) -> dict[str, float]:
    thresholds = {}
    for name, statistics in calibration["statistics"].items():
        matches = [row for row in statistics["targets"] if math.isclose(float(row["target_sparsity"]), target)]
        if len(matches) != 1:
            raise ValueError(f"Calibration target {target} is missing for {name}.")
        thresholds[name] = float(matches[0]["threshold"])
    return thresholds


def _first_tensor(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                pass
    raise TypeError("Hook value did not contain a tensor.")


def _valid_threshold(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"Invalid threshold for {name}: {value}")
    return result


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
