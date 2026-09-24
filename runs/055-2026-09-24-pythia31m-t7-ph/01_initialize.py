"""Create one immutable seed1234 random 31M initialization before any training."""
import hashlib
import platform
import random
import numpy as np
import torch
import transformers
from safetensors.torch import save_file
from initialization import apply_pythia_31m_initialization
from model_factory import build_pinned_run055_model
from run_config import RUN_DIR, load_config, parameter_sha256, seed_everything, write_json


def main():
    config = load_config()
    paths = config["initialization_artifact"]
    if any((RUN_DIR/value).exists() for value in paths.values()):
        raise FileExistsError("Canonical initialization already exists")
    model = build_pinned_run055_model(config["model"], device=torch.device("cpu"),
                                    torch=torch, auto_model=transformers.AutoModelForCausalLM)
    seed_everything(torch, config["seeds"]["model"])
    recipe = apply_pythia_31m_initialization(model, torch=torch)
    meta = dict(parameter_sha256=parameter_sha256(model), recipe=recipe,
                architecture=config["model"]["architecture"], revision=config["model"]["revision"],
                seed=config["seeds"]["model"], parameter_count=sum(p.numel() for p in model.parameters()),
                origin="Locally generated random step-zero weights; never released/trained weights",
                runtime=dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__))
    save_file(model.state_dict(), str(RUN_DIR/paths["model_path"]))
    torch.save(dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state()),
               RUN_DIR/paths["rng_path"])
    for key in ("model", "rng"):
        path = RUN_DIR/paths[key+"_path"]
        meta[key] = dict(path=paths[key+"_path"], bytes=path.stat().st_size,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write_json(RUN_DIR/paths["metadata_path"], meta)
    print(meta["parameter_count"], meta["parameter_sha256"])


if __name__ == "__main__":
    main()
