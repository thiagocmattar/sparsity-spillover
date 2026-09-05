"""Reuse the architecture-independent frozen Run-023 sparse adapters."""

from _reuse_run023 import load_run023_module


_impl = load_run023_module("_run023_pythia_sparse", "pythia_sparse.py")

ExactEllLinear = _impl.ExactEllLinear
ExactEllWorkspace = _impl.ExactEllWorkspace
clear_adapter_workspaces = _impl.clear_adapter_workspaces
dense_attention_composition = _impl.dense_attention_composition
exact_sparse_mm = _impl.exact_sparse_mm
install_sparse_linears = _impl.install_sparse_linears
set_adapter_mode = _impl.set_adapter_mode
sparse_attention_composition = _impl.sparse_attention_composition

