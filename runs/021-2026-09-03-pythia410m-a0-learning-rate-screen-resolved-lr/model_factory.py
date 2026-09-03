"""Construct Run 021 from the vendored config without loading released weights."""

from __future__ import annotations

import json
from typing import Any

from sparsity_research.pythia import build_random_pythia

from run_config import RUN_DIR


ARCHITECTURE_CONFIG = RUN_DIR / "architecture_config.json"
ARCHITECTURE_NAME = "EleutherAI/pythia-410m-deduped"
ARCHITECTURE_REVISION = "b5e8535141902c0e985cea61fd02afe7fe86af32"
EXPECTED_PARAMETER_COUNT = 405_334_016


def build_pinned_run021_model(
    model_config: dict[str, Any], *, device: Any, torch: Any, auto_model: Any
) -> Any:
    """Construct only the checked-in architecture; released weights are never requested."""

    if (
        model_config.get("architecture") != ARCHITECTURE_NAME
        or model_config.get("revision") != ARCHITECTURE_REVISION
    ):
        raise ValueError("Run 021 model source differs from the pinned architecture identity.")

    class PinnedAutoConfig:
        @staticmethod
        def from_pretrained(name: str, *, revision: str):
            if name != ARCHITECTURE_NAME or revision != ARCHITECTURE_REVISION:
                raise ValueError("Unexpected architecture request reached the pinned config loader.")
            from transformers import GPTNeoXConfig

            value = json.loads(ARCHITECTURE_CONFIG.read_text(encoding="utf-8"))
            return GPTNeoXConfig.from_dict(value)

    model = build_random_pythia(
        model_config,
        device=device,
        torch=torch,
        auto_config=PinnedAutoConfig,
        auto_model=auto_model,
    )
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Pythia-410M parameter count changed: {parameter_count} != {EXPECTED_PARAMETER_COUNT}"
        )
    return model
