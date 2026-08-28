from types import SimpleNamespace

import numpy as np
import torch

from sparsity_research.clipping import evaluate_clipping_point
from sparsity_research.evaluation import evaluate_complete_blocks


class _Model(torch.nn.Module):
    def forward(self, *, input_ids, labels):
        assert torch.equal(input_ids, labels)
        return SimpleNamespace(loss=input_ids.float().mean())


def test_evaluation_uses_every_complete_block_once_and_reports_tail():
    tokens = np.arange(11, dtype=np.int32)
    result = evaluate_complete_blocks(
        model=_Model(),
        tokens=tokens,
        block_size=4,
        batch_size=2,
        device=torch.device("cpu"),
        torch=torch,
        np=np,
        autocast_dtype=None,
    )
    assert result["sequences"] == 2
    assert result["input_tokens"] == 8
    assert result["excluded_tail_tokens"] == 3
    assert result["complete_block_coverage"] is True
    assert result["loss"] == 3.5


class _ClipModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt_neox = torch.nn.Module()
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.act = torch.nn.Identity()
        self.gpt_neox.layers = torch.nn.ModuleList([layer])

    def forward(self, *, input_ids, labels):
        del labels
        activations = self.gpt_neox.layers[0].mlp.act(input_ids.float() - 1.0)
        return SimpleNamespace(loss=activations.square().mean())


def test_clipping_point_evaluates_and_counts_clipped_site_values():
    result = evaluate_clipping_point(
        model=_ClipModel(),
        tokens=np.array([0, 1, 2, 3, 0, 1, 2, 3, 9], dtype=np.int32),
        block_size=4,
        batch_size=1,
        device=torch.device("cpu"),
        torch=torch,
        np=np,
        autocast_dtype=None,
        clipping={"enabled": True, "mode": "threshold", "threshold": 1.0, "sites": ["h"]},
    )
    assert result["validation"]["sequences"] == 2
    assert result["validation"]["excluded_tail_tokens"] == 1
    assert result["activations"][0]["name"] == "h.layer_0"
    assert result["activations"][0]["exact_zero_count"] == 6
