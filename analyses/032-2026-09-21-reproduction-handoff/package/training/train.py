"""One explicit paper condition per invocation; no cloud provisioning or launcher."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time
from training.config import (
    ROOT,
    identifier,
    paper_rows,
    resolve,
    register_hz,
    parameter_hash,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true")
    p.add_argument("--condition")
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--initial-state",
        type=Path,
        help="Untrained safetensors from the canonical initialization; never released pretrained weights",
    )
    p.add_argument(
        "--fresh-initialization",
        action="store_true",
        help="Explicitly run a new seed-matched draw; not byte-exact historical reproduction",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Two tiny synthetic optimizer boundaries; no paper measurement",
    )
    a = p.parse_args()
    if a.list:
        print("\n".join(identifier(r) for r in paper_rows()))
        return
    if not a.condition or a.output is None:
        p.error("--condition and a new --output directory are required")
    if not a.smoke and bool(a.initial_state) == a.fresh_initialization:
        p.error("Choose --initial-state or explicitly --fresh-initialization")
    cfg = resolve(a.condition)
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, GPTNeoXConfig
    from sparsity_research.pythia import (
        build_random_pythia,
        apply_activation_topology,
        expose_attention_sites,
    )
    from sparsity_research.data import build_training_schedule, file_sha256
    from sparsity_research.capture import ActivationCapture
    from sparsity_research.pressure import parse_pressure_config
    from training import optimizer, ol1
    from training.initialization import apply_pythia_14m_initialization
    from training.evaluate import evaluate

    if not a.smoke and not torch.cuda.is_available():
        raise RuntimeError("Full training requires CUDA; smoke is CPU-capable")
    device = torch.device("cpu" if a.smoke else "cuda")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    register_hz()
    if a.smoke:
        c = GPTNeoXConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=32,
        )
        c.topology_id = cfg["model"]["topology_id"]
        c.site_gates = cfg["model"]["site_gates"]
        c.use_cache = False
        c._attn_implementation = "eager"
        model = apply_activation_topology(
            AutoModelForCausalLM.from_config(c), torch=torch
        )
        tokens = np.arange(1024, dtype=np.int32) % 64
        block = 16
        steps = 2
        mb = 1
        gas = 2
    else:
        model = build_random_pythia(
            cfg["model"],
            device=device,
            torch=torch,
            auto_config=AutoConfig,
            auto_model=AutoModelForCausalLM,
        )
        model.config.hidden_dropout = 0.0
        model.config.attention_dropout = 0.0
        apply_pythia_14m_initialization(model, torch=torch)
        if a.initial_state:
            from safetensors.torch import load_file

            model.load_state_dict(load_file(str(a.initial_state)), strict=True)
        block = 2048
        steps = cfg["training"]["max_steps"]
        mb = cfg["training"]["micro_batch_size"]
        gas = cfg["training"]["gradient_accumulation_steps"]
        cache = ROOT / "data/tokenized/minipile-pythia-14m-full"
        for split, expected in [
            (
                "train",
                "da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c",
            ),
            (
                "validation",
                "51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451",
            ),
        ]:
            if file_sha256(cache / split / "tokens.int32.bin") != expected:
                raise ValueError(f"{split} cache identity mismatch")
        tokens = np.memmap(cache / "train/tokens.int32.bin", dtype=np.int32, mode="r")
    initial = parameter_hash(model)
    if a.initial_state and initial != cfg["expected_initial_parameter_sha256"]:
        raise ValueError("Canonical random parameter hash mismatch")
    starts, schedule, metadata = build_training_schedule(
        np,
        token_count=len(tokens),
        block_size=block,
        max_steps=steps,
        gradient_accumulation_steps=gas,
        micro_batch_size=mb,
        seed=1234,
    )
    if not a.smoke and schedule != cfg["expected_schedule_sha256"]:
        raise ValueError("Paper schedule mismatch")
    a.output.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (a.output / name).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n"
        )

    cfg.update(
        initial_parameter_sha256=initial,
        schedule_sha256=schedule,
        schedule=metadata,
        initialization_mode="canonical_random_replay"
        if a.initial_state
        else "fresh_random_draw",
        smoke=a.smoke,
        runtime=dict(
            torch=torch.__version__, cuda=torch.version.cuda, device=str(device)
        ),
    )
    write("config.json", cfg)
    write("manifest.json", dict(status="running", condition=a.condition, smoke=a.smoke))
    expose_attention_sites(model, torch=torch)
    opt, _ = optimizer.build_recipe_adamw(model, cfg["training"], torch=torch)
    scaler = optimizer.DynamicLossScaler()
    pressure = parse_pressure_config(cfg["pressure"])
    boundary = (
        ol1.run_recipe_boundary
        if pressure.orthogonal
        else optimizer.run_recipe_boundary
    )
    context = (
        ActivationCapture(model, list(pressure.sites), torch=torch)
        if pressure.enabled
        else nullcontext(None)
    )
    start = time.perf_counter()
    try:
        if not a.smoke:
            model.save_pretrained(a.output / "initial", safe_serialization=True)
        with context as capture, (a.output / "events.jsonl").open("w") as events:
            for step in range(1, steps + 1):
                model.train()
                lr = optimizer.recipe_learning_rate(
                    step,
                    peak=cfg["training"]["peak_learning_rate"],
                    minimum=cfg["training"]["minimum_learning_rate"],
                    max_steps=steps,
                    warmup_fraction=0.01,
                )
                for group in opt.param_groups:
                    group["lr"] = lr
                batches = [
                    torch.as_tensor(
                        np.stack(
                            [
                                tokens[int(s) : int(s) + block]
                                for s in starts[step - 1, j]
                            ]
                        ),
                        dtype=torch.long,
                        device=device,
                    )
                    for j in range(gas)
                ]
                metrics = boundary(
                    model=model,
                    optimizer=opt,
                    batches=batches,
                    pressure=pressure,
                    capture=capture,
                    loss_scaler=scaler,
                    gradient_clip_norm=1.0,
                    torch=torch,
                    device=device,
                )
                metrics.update(
                    step=step,
                    learning_rate=lr,
                    elapsed_seconds=time.perf_counter() - start,
                )
                events.write(json.dumps(metrics, allow_nan=False) + "\n")
                events.flush()
                print(
                    json.dumps(
                        dict(
                            step=step,
                            loss=metrics["task_loss"],
                            elapsed=metrics["elapsed_seconds"],
                        )
                    ),
                    flush=True,
                )
                if metrics["optimizer_step_skipped"]:
                    raise RuntimeError(
                        "Overflow skipped a boundary; this run does not match the no-skip paper cohort"
                    )
                if not a.smoke and step == 1:
                    write(
                        "validation-step1.json",
                        evaluate(
                            model,
                            cache / "validation/tokens.int32.bin",
                            diagnostics=False,
                        ),
                    )
        model.save_pretrained(a.output / "final", safe_serialization=True)
        torch.save(
            {
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "step": steps,
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else [],
            },
            a.output / "recovery.pt",
        )
        if not a.smoke:
            from sparsity_research.pythia import load_checkpoint_pythia

            del model, opt
            torch.cuda.empty_cache()
            model = load_checkpoint_pythia(
                AutoModelForCausalLM, a.output / "final", torch=torch
            ).to(device)
            model.set_attn_implementation("sdpa")
            model.config.use_cache = False
            write(
                "metrics.json",
                evaluate(
                    model, cache / "validation/tokens.int32.bin", diagnostics=True
                ),
            )
        write(
            "manifest.json",
            dict(
                status="completed",
                condition=a.condition,
                smoke=a.smoke,
                steps=steps,
                elapsed_seconds=time.perf_counter() - start,
            ),
        )
    except Exception as exc:
        write(
            "manifest.json",
            dict(status="failed", error=str(exc), condition=a.condition, smoke=a.smoke),
        )
        raise


if __name__ == "__main__":
    main()
