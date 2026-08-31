"""Construct Run 016 from the vendored config at the pinned Hub revision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sparsity_research.pythia import build_random_pythia

from run_config import RUN_DIR


ARCHITECTURE_CONFIG = RUN_DIR / "architecture_config.json"
ARCHITECTURE_NAME = "EleutherAI/pythia-70m-deduped"
ARCHITECTURE_REVISION = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"


def build_pinned_run016_model(
    model_config: dict[str, Any], *, device: Any, torch: Any, auto_model: Any
) -> Any:
    """Use the checked-in official config; never load released weights."""

    if (
        model_config.get("architecture") != ARCHITECTURE_NAME
        or model_config.get("revision") != ARCHITECTURE_REVISION
    ):
        raise ValueError("Run 016 model source differs from the pinned architecture identity.")

    class PinnedAutoConfig:
        @staticmethod
        def from_pretrained(name: str, *, revision: str):
            if name != ARCHITECTURE_NAME or revision != ARCHITECTURE_REVISION:
                raise ValueError("Unexpected architecture request reached the pinned config loader.")
            from transformers import GPTNeoXConfig

            value = json.loads(ARCHITECTURE_CONFIG.read_text(encoding="utf-8"))
            return GPTNeoXConfig.from_dict(value)

    return build_random_pythia(
        model_config,
        device=device,
        torch=torch,
        auto_config=PinnedAutoConfig,
        auto_model=auto_model,
    )
