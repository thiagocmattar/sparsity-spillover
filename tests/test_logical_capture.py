from types import SimpleNamespace

import torch

from sparsity_research.logical_capture import (
    LogicalProductAccumulator,
    capture_logical_products,
)


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.Module()
        self.attention.query_key_value = torch.nn.Linear(2, 6, bias=False)
        self.attention.dense = torch.nn.Linear(2, 2, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.dense_h_to_4h = torch.nn.Linear(2, 4, bias=False)
        self.mlp.dense_4h_to_h = torch.nn.Linear(4, 2, bias=False)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt_neox = torch.nn.Module()
        self.gpt_neox.layers = torch.nn.ModuleList([_Layer()])
        self.config = SimpleNamespace(_attn_implementation="sdpa")
        self.output = torch.nn.Linear(2, 5, bias=False)

    def get_output_embeddings(self):
        return self.output


def test_logical_capture_observes_all_operations_and_restores_attention():
    model = _Model()

    def original_eager(_module, query, _key, value, _mask, **_kwargs):
        probabilities = torch.ones(query.shape[:-1] + (query.shape[-2],))
        return value, probabilities

    modeling = SimpleNamespace(eager_attention_forward=original_eager)
    accumulator = LogicalProductAccumulator()
    layer = model.gpt_neox.layers[0]
    with capture_logical_products(
        model,
        accumulator=accumulator,
        torch=torch,
        modeling_gpt_neox=modeling,
    ):
        assert model.config._attn_implementation == "eager"
        layer.attention.query_key_value(torch.tensor([[[0.0, 1.0]]]))
        layer.attention.dense(torch.tensor([[[0.0, 1.0]]]))
        hidden = layer.mlp.dense_h_to_4h(torch.tensor([[[0.0, 1.0]]]))
        layer.mlp.dense_4h_to_h(hidden)
        q = torch.ones(1, 1, 2, 2)
        k = torch.ones_like(q)
        v = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        modeling.eager_attention_forward(None, q, k, v, None, scaling=1.0)

    assert model.config._attn_implementation == "sdpa"
    assert modeling.eager_attention_forward is original_eager
    summary = accumulator.summary(model=model, total_input_tokens=2)
    assert set(summary["per_operation"]) == set(accumulator.zero_counts)
    assert summary["lm_head_product_count"] == 20
    assert 0.0 <= summary["R_block"] <= 1.0
    assert 0.0 <= summary["R_model"] <= 1.0
