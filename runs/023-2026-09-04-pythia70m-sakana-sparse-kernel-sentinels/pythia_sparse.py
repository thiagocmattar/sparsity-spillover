"""Run-local Pythia-70M adapters for the Sakana-derived exact ELL operators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ExactEllWorkspace:
    values: Any | None = None
    columns: Any | None = None
    counts: Any | None = None
    output: Any | None = None
    shape: tuple[int, int, int] | None = None
    allocations: int = 0

    def ensure(self, rows: int, inner: int, output_columns: int, *, device: Any, torch: Any) -> None:
        shape = (rows, inner, output_columns)
        if self.shape == shape and self.values is not None and self.values.device == device:
            return
        options = {"device": device}
        self.values = torch.empty((rows, inner), dtype=torch.bfloat16, **options)
        self.columns = torch.empty((rows, inner), dtype=torch.uint16, **options)
        self.counts = torch.empty((rows,), dtype=torch.int32, **options)
        self.output = torch.empty((rows, output_columns), dtype=torch.bfloat16, **options)
        self.shape = shape
        self.allocations += 1

    def clear(self) -> None:
        self.values = self.columns = self.counts = self.output = None
        self.shape = None


def exact_sparse_mm(matrix: Any, rhs: Any, workspace: ExactEllWorkspace, *, torch: Any) -> Any:
    if matrix.ndim != 2 or rhs.ndim != 2 or matrix.shape[1] != rhs.shape[0]:
        raise ValueError("Sparse matmul expects [M,K] @ [K,N].")
    if matrix.dtype != torch.bfloat16 or rhs.dtype != torch.bfloat16:
        raise TypeError("Sparse matmul is fixed to BF16.")
    matrix = matrix.contiguous()
    rhs = rhs.contiguous()
    rows, inner = matrix.shape
    output_columns = rhs.shape[1]
    workspace.ensure(rows, inner, output_columns, device=matrix.device, torch=torch)
    torch.ops.sparse_ops.dense_to_ell_exact_out(
        matrix, workspace.values, workspace.columns, workspace.counts
    )
    torch.ops.sparse_ops.ell_spmm_raw_out(
        workspace.values,
        workspace.columns,
        workspace.counts,
        rhs,
        rows,
        inner,
        output_columns,
        inner,
        inner,
        workspace.output,
    )
    return workspace.output


class ExactEllLinear(torch.nn.Module):
    """Inference-only nn.Module-compatible linear with reusable ELL workspaces."""

    def __init__(self, linear: Any, *, operation: str, torch: Any) -> None:
        super().__init__()
        if not isinstance(linear, torch.nn.Linear):
            raise TypeError(f"{operation} adapter requires torch.nn.Linear.")
        self.operation = operation
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.register_buffer(
            "_rhs", linear.weight.detach().transpose(0, 1).contiguous(), persistent=False
        )
        self.workspace = ExactEllWorkspace()
        self.mode = "sparse"
        self._torch = torch

    def forward(self, inputs: Any) -> Any:
        torch = self._torch
        if self.mode == "dense":
            return torch.nn.functional.linear(inputs, self.weight, self.bias)
        if self.mode != "sparse":
            raise ValueError(f"Unknown adapter mode: {self.mode}")
        if torch.is_grad_enabled():
            raise RuntimeError("ExactEllLinear is inference-only.")
        if inputs.dtype != torch.bfloat16 or not inputs.is_cuda:
            raise TypeError("ExactEllLinear sparse mode requires CUDA BF16 activations.")
        flat = inputs.reshape(-1, self.in_features).contiguous()
        result = exact_sparse_mm(flat, self._rhs, self.workspace, torch=torch)
        if self.bias is not None:
            result = result + self.bias
        return result.reshape(*inputs.shape[:-1], self.out_features)

    def clear_workspace(self) -> None:
        self.workspace.clear()


def install_sparse_linears(model: Any, operations: list[str], *, torch: Any) -> dict[str, ExactEllLinear]:
    """Replace only the declared downstream linears; gate placement is untouched."""

    adapters: dict[str, ExactEllLinear] = {}
    for layer_index, layer in enumerate(model.gpt_neox.layers):
        targets = {
            "qkv_projection": (layer.attention, "query_key_value"),
            "mlp_w1": (layer.mlp, "dense_h_to_4h"),
            "mlp_w2": (layer.mlp, "dense_4h_to_h"),
            "attention_output_projection": (layer.attention, "dense"),
        }
        for operation in operations:
            parent, name = targets[operation]
            original = getattr(parent, name)
            if isinstance(original, ExactEllLinear):
                raise ValueError(f"Layer {layer_index} {operation} already has an adapter.")
            adapter = ExactEllLinear(original, operation=operation, torch=torch)
            setattr(parent, name, adapter)
            adapters[f"layer_{layer_index}.{operation}"] = adapter
    return adapters


def set_adapter_mode(adapters: dict[str, ExactEllLinear], mode: str) -> None:
    if mode not in {"dense", "sparse"}:
        raise ValueError(f"Unsupported adapter mode: {mode}")
    for adapter in adapters.values():
        adapter.mode = mode


def clear_adapter_workspaces(adapters: dict[str, ExactEllLinear]) -> None:
    for adapter in adapters.values():
        adapter.clear_workspace()


def dense_attention_composition(query: Any, key: Any, value: Any, *, torch: Any) -> Any:
    """Matched per-head dense causal attention, including scale/mask/softmax."""

    if query.shape != key.shape or query.shape != value.shape or query.ndim != 4:
        raise ValueError("Attention tensors must share [B,H,T,d].")
    batch, heads, sequence, head_size = query.shape
    scale = 1.0 / math.sqrt(head_size)
    causal = torch.ones((sequence, sequence), dtype=torch.bool, device=query.device).tril()
    outputs = []
    for batch_index in range(batch):
        head_outputs = []
        for head_index in range(heads):
            q = query[batch_index, head_index].contiguous()
            k = key[batch_index, head_index].contiguous()
            v = value[batch_index, head_index].contiguous()
            scores = (q @ k.transpose(0, 1).contiguous()) * scale
            probabilities = torch.softmax(scores.masked_fill(~causal, -torch.inf).float(), dim=-1).to(query.dtype)
            head_outputs.append(probabilities @ v)
        outputs.append(torch.stack(head_outputs))
    return torch.stack(outputs)


def sparse_attention_composition(
    query: Any,
    key: Any,
    value: Any,
    *,
    q_workspace: ExactEllWorkspace,
    v_workspace: ExactEllWorkspace,
    torch: Any,
) -> Any:
    """q-packed QK plus v-packed (V^T @ P^T)^T, dispatched per head."""

    if query.shape != key.shape or query.shape != value.shape or query.ndim != 4:
        raise ValueError("Attention tensors must share [B,H,T,d].")
    batch, heads, sequence, head_size = query.shape
    scale = 1.0 / math.sqrt(head_size)
    causal = torch.ones((sequence, sequence), dtype=torch.bool, device=query.device).tril()
    outputs = []
    for batch_index in range(batch):
        head_outputs = []
        for head_index in range(heads):
            q = query[batch_index, head_index].contiguous()
            k_t = key[batch_index, head_index].transpose(0, 1).contiguous()
            scores = exact_sparse_mm(q, k_t, q_workspace, torch=torch) * scale
            probabilities = torch.softmax(scores.masked_fill(~causal, -torch.inf).float(), dim=-1).to(query.dtype)
            v_t = value[batch_index, head_index].transpose(0, 1).contiguous()
            output_t = exact_sparse_mm(
                v_t, probabilities.transpose(0, 1).contiguous(), v_workspace, torch=torch
            )
            # The workspace is reused by the next head, so materialize the
            # current head exactly as the dense path materializes each GEMM.
            head_outputs.append(output_t.transpose(0, 1).clone())
        outputs.append(torch.stack(head_outputs))
    return torch.stack(outputs)
