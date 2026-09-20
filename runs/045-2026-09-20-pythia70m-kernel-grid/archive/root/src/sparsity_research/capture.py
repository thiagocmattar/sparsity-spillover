"""Activation capture and evaluation-time clipping at canonical sites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .sites import SITE_SPECS, resolve_sites


@dataclass(frozen=True)
class CapturedSite:
    alias: str
    name: str
    module_path: str
    tensor: str
    shape: str
    downstream_matmul: str


class ActivationCapture:
    """Capture exact site outputs, optionally replacing selected values."""

    def __init__(
        self,
        model: Any,
        sites: list[str],
        *,
        torch: Any,
        clipping: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.sites = resolve_sites(sites)
        self.torch = torch
        self.clipping = clipping or {"enabled": False}
        self.activations: dict[str, Any] = {}
        self.metadata: list[CapturedSite] = []
        self._handles: list[Any] = []

    def __enter__(self) -> "ActivationCapture":
        self.register()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.remove()

    def clear(self) -> None:
        self.activations.clear()

    def register(self) -> None:
        self.remove()
        self.metadata.clear()
        for alias in self.sites:
            if alias == "a":
                self._register_branch(alias, "a_gate", "input_layernorm")
            elif alias == "m":
                self._register_branch(alias, "m_gate", "post_attention_layernorm")
            elif alias == "h":
                self._register_h()
            else:
                self._register_attention(alias)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _layers(self, alias: str) -> Any:
        layers = getattr(getattr(self.model, "gpt_neox", None), "layers", None)
        if layers is None:
            raise ValueError(f"Site {alias} capture supports GPT-NeoX/Pythia models only.")
        return layers

    def _register_branch(self, alias: str, gate_name: str, layernorm_name: str) -> None:
        for index, layer in enumerate(self._layers(alias)):
            module = getattr(layer, gate_name, None)
            module_name = gate_name
            if module is None:
                module = getattr(layer, layernorm_name, None)
                module_name = layernorm_name
            if module is None:
                raise ValueError(f"Could not resolve site {alias} in layer {index}.")
            self._register(alias, index, module, f"gpt_neox.layers.{index}.{module_name}")

    def _register_h(self) -> None:
        for index, layer in enumerate(self._layers("h")):
            module = getattr(getattr(layer, "mlp", None), "act", None)
            if module is None:
                raise ValueError(f"Could not resolve site h in layer {index}.")
            self._register("h", index, module, f"gpt_neox.layers.{index}.mlp.act")

    def _register_attention(self, alias: str) -> None:
        from .pythia import expose_attention_sites

        expose_attention_sites(self.model, torch=self.torch)
        for index, layer in enumerate(self._layers(alias)):
            module = getattr(layer.attention, f"{alias}_site", None)
            if module is None:
                raise ValueError(f"Could not resolve attention site {alias} in layer {index}.")
            self._register(
                alias,
                index,
                module,
                f"gpt_neox.layers.{index}.attention.{alias}_site",
            )

    def _register(self, alias: str, index: int, module: Any, module_path: str) -> None:
        name = f"{alias}.layer_{index}"
        spec = SITE_SPECS[alias]
        self.metadata.append(
            CapturedSite(alias, name, module_path, spec.tensor, spec.shape, spec.downstream_matmul)
        )
        self._handles.append(module.register_forward_hook(self._hook(name, alias)))

    def _hook(self, name: str, alias: str) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            value = _first_tensor(output)
            if clipping_enabled(self.clipping, alias, name):
                value = clip_tensor(value, self.clipping, torch=self.torch)
                self.activations[name] = value
                return _replace_first_tensor(output, value)
            self.activations[name] = value
            return output

        return hook


def clip_tensor(value: Any, config: dict[str, Any], *, torch: Any) -> Any:
    mode = config.get("mode", "threshold")
    if mode == "threshold":
        threshold = float(config.get("threshold", 0.0))
        if threshold < 0.0:
            raise ValueError("Clipping threshold must be nonnegative.")
        return value.masked_fill(value.detach().abs() <= threshold, 0.0)
    if mode == "rms_threshold":
        multiplier = float(config["rms_multiplier"])
        if multiplier < 0.0:
            raise ValueError("RMS multiplier must be nonnegative.")
        detached = value.detach().float()
        threshold = multiplier * detached.square().mean().sqrt()
        return value.masked_fill(detached.abs() <= threshold, 0.0)
    if mode == "quantile":
        quantile = float(config["quantile"])
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("Quantile must be in [0, 1].")
        flat = value.detach().abs().reshape(-1).float()
        if flat.numel() == 0:
            return value
        k = max(1, min(flat.numel(), round(quantile * flat.numel())))
        threshold = flat.kthvalue(k).values
        return value.masked_fill(value.detach().abs() <= threshold, 0.0)
    raise ValueError(f"Unsupported clipping mode: {mode!r}.")


def clipping_enabled(config: dict[str, Any], alias: str, name: str) -> bool:
    if not config.get("enabled", False):
        return False
    sites = config.get("sites", [])
    return alias in sites or name in sites


def _first_tensor(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    raise TypeError("Could not find a tensor in the hook output.")


def _replace_first_tensor(value: Any, replacement: Any) -> Any:
    if hasattr(value, "detach"):
        return replacement
    if not isinstance(value, (tuple, list)):
        raise TypeError("Could not find a tensor in the hook output.")
    items = list(value)
    for index, item in enumerate(items):
        try:
            items[index] = _replace_first_tensor(item, replacement)
            return tuple(items) if isinstance(value, tuple) else items
        except TypeError:
            continue
    raise TypeError("Could not find a tensor in the hook output.")

