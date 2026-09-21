"""Separate ordinary loss, activation statistics and actual-operand logical passes."""

import argparse
import json
from pathlib import Path
from training.config import ROOT, register_hz


def evaluate(model, token_path, diagnostics=True):
    import numpy as np
    import torch
    from sparsity_research.evaluation import evaluate_complete_blocks
    from sparsity_research.capture import ActivationCapture
    from sparsity_research.metrics import ActivationAccumulator, weight_statistics
    from sparsity_research.logical_capture import (
        LogicalProductAccumulator,
        capture_logical_products,
    )
    from sparsity_research.ceilings import architecture_ceiling
    from sparsity_research.pythia import expose_attention_sites
    from training.optimizer import recipe_attention_context
    from sparsity_research.data import file_sha256, FULL_VALIDATION_SHA256

    if file_sha256(token_path) != FULL_VALIDATION_SHA256:
        raise ValueError("Full validation cache hash mismatch")
    tokens = np.memmap(token_path, dtype=np.int32, mode="r")
    device = next(model.parameters()).device
    if divmod(len(tokens), 2048) != (338, 1444):
        raise ValueError("Expected all 338 validation blocks and 1,444 excluded tokens")
    args = dict(
        model=model,
        tokens=tokens,
        block_size=2048,
        batch_size=32 if model.config.hidden_size == 128 else 4,
        device=device,
        torch=torch,
        np=np,
        autocast_dtype=torch.float16,
    )
    with recipe_attention_context(torch, device):
        ordinary = evaluate_complete_blocks(**args)
    result = {
        "ordinary_validation": ordinary,
        "precision": "FP16 autocast; FP32 parameters",
        "validation_documents": 500,
    }
    if not diagnostics:
        return result
    expose_attention_sites(model, torch=torch)
    sites = ["a", "m", "h", "q_post", "k_post", "v", "z"]
    accumulator = ActivationAccumulator((0.0, 0.001, 0.01))
    with ActivationCapture(model, sites, torch=torch) as capture:

        def observe(output, sequences):
            if len(capture.activations) != len(sites) * model.config.num_hidden_layers:
                raise RuntimeError("Incomplete site capture")
            accumulator.update(capture.activations, torch=torch)
            capture.clear()

        with recipe_attention_context(torch, device):
            activation_coverage = evaluate_complete_blocks(**args, after_batch=observe)
    logical = LogicalProductAccumulator()
    with capture_logical_products(model, accumulator=logical, torch=torch):
        logical_coverage = evaluate_complete_blocks(**{**args, "batch_size": 1})
    c = model.config
    result.update(
        activation_coverage=activation_coverage,
        activation_rows=accumulator.rows(),
        activation_pooled=accumulator.pooled_by_site(),
        weights=weight_statistics(model),
        logical_coverage=logical_coverage,
        logical=logical.summary(model=model, total_input_tokens=692224),
        ceiling=architecture_ceiling(
            c.topology_id,
            layers=c.num_hidden_layers,
            hidden_size=c.hidden_size,
            ffn_size=c.intermediate_size,
            sequence_length=2048,
            vocabulary_size=c.vocab_size,
        ),
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM
    from sparsity_research.pythia import load_checkpoint_pythia

    register_hz()
    model = load_checkpoint_pythia(AutoModelForCausalLM, a.checkpoint, torch=torch).to(
        "cuda"
    )
    model.set_attn_implementation("sdpa")
    model.config.use_cache = False
    result = evaluate(
        model,
        ROOT / "data/tokenized/minipile-pythia-14m-full/validation/tokens.int32.bin",
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
