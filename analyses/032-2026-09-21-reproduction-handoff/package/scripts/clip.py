"""Pinned ten-block calibration and ten full-validation post-hoc clipping points."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM
    from sparsity_research.pythia import load_checkpoint_pythia, expose_attention_sites
    from sparsity_research.data import file_sha256, FULL_VALIDATION_SHA256
    from training.config import register_hz
    from training.clipping import _calibrate, _thresholds_for_target, _evaluate_point

    register_hz()
    cache = ROOT / "data/tokenized/minipile-pythia-14m-full"
    for split, expected in [
        ("train", "da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c"),
        ("validation", FULL_VALIDATION_SHA256),
    ]:
        if file_sha256(cache / split / "tokens.int32.bin") != expected:
            raise ValueError(f"{split} cache mismatch")
    model = (
        load_checkpoint_pythia(AutoModelForCausalLM, a.checkpoint, torch=torch)
        .to("cuda")
        .eval()
    )
    model.config.use_cache = False
    model.set_attn_implementation("eager")
    expose_attention_sites(model, torch=torch)
    train = np.memmap(cache / "train/tokens.int32.bin", dtype=np.int32, mode="r")
    validation = np.memmap(
        cache / "validation/tokens.int32.bin", dtype=np.int32, mode="r"
    )
    targets = tuple(i / 10 for i in range(10))
    a.output.mkdir(parents=True, exist_ok=False)
    calibration = _calibrate(
        model,
        train,
        targets=targets,
        blocks=10,
        block_size=2048,
        device=torch.device("cuda"),
        torch=torch,
        np=np,
    )
    (a.output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    with (a.output / "points.jsonl").open("w") as f:
        for target in targets:
            thresholds = _thresholds_for_target(calibration, target)
            point = _evaluate_point(
                model,
                validation,
                thresholds,
                block_size=2048,
                batch_size=1,
                device=torch.device("cuda"),
                torch=torch,
                np=np,
            )
            point["target"] = target
            f.write(json.dumps(point, allow_nan=False) + "\n")
            f.flush()
            print(
                json.dumps(
                    {
                        "target": target,
                        "loss": point["validation"]["loss"],
                        "R_model": point["logical_products"]["R_model"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
