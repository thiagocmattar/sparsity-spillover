"""Replay the single local random initialization; verify tensor and file identities."""
import hashlib
import json
import random
import numpy as np
from safetensors.torch import load_file
from run_config import RUN_DIR, load_config, parameter_sha256


def load_pinned_initialization(model, *, torch):
    paths = load_config()["initialization_artifact"]
    meta = json.loads((RUN_DIR/paths["metadata_path"]).read_text())
    for key in ("model", "rng"):
        path = RUN_DIR/paths[key+"_path"]
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != meta[key]["sha256"] or path.stat().st_size != meta[key]["bytes"]:
            raise RuntimeError(f"Initialization {key} file mismatch")
    if {p.device.type for p in model.parameters()} != {"cpu"}:
        raise RuntimeError("Initialization replay must happen on CPU")
    model.load_state_dict(load_file(str(RUN_DIR/paths["model_path"])), strict=True)
    if parameter_sha256(model) != meta["parameter_sha256"]:
        raise RuntimeError("Initialization parameter mismatch")
    state = torch.load(RUN_DIR/paths["rng_path"], map_location="cpu", weights_only=False)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    model.config.pythia_recipe_initialization = meta["recipe"]
    model.config.use_cache = False
    model.config._attn_implementation = "sdpa"
    return dict(**meta["recipe"], random_initialization_replay=True,
                parameter_sha256=meta["parameter_sha256"], released_weights_loaded=False)
