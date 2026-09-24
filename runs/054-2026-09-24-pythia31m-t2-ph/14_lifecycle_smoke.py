"""Two-update non-evidence exercise of the actual publication/retrieval lifecycle."""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import torch
import training
from run_config import (RUN_DIR, load_config, condition_specs, load_verified_caches,
    build_schedule, run_code_identity, git_identity, write_json)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attempt",required=True)
    args = p.parse_args()
    if not args.attempt.replace("-","").isalnum():
        raise ValueError("Simple unique attempt required")
    root = RUN_DIR/"prelaunch"/("lifecycle-"+args.attempt)
    root.mkdir(exist_ok=False)
    config = load_config()
    config["name"] += "-NON-EVIDENCE-LIFECYCLE-SMOKE"
    config["training"]["max_steps"] = 2
    config["checkpoints"].update(model_steps=[0,1,2],optimizer_steps=[2])
    training._BASE.RUN_DIR = root
    train,val,tm,vm,seconds = load_verified_caches(config,np=np)
    starts,digest,schedule = build_schedule(config,tm,np=np)
    spec = importlib.util.spec_from_file_location("run054_verify",RUN_DIR/"03_verify.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    result = dict(kind="non_evidence_lifecycle_smoke",status="running",conditions=[])
    write_json(root/"result.json",result)
    for row in (condition_specs(config)[0],condition_specs(config)[-1]):
        attempt = training.run_condition(config=config,worker_id=row["id"],condition=row,
            train_tokens=train,validation_tokens=val,train_metadata=tm,validation_metadata=vm,
            starts=starts,schedule_hash=digest,schedule_metadata=schedule,cache_verification_seconds=seconds,
            code_identity=run_code_identity(),repository_identity=dict(kind="non_evidence_staged_source_smoke"),torch=torch,np=np)
        result["conditions"].append(verifier.verify(attempt,config))
        write_json(root/"result.json",result)
    result["status"] = "passed"
    write_json(root/"result.json",result)
    print("Lifecycle smoke passed; not part of the scientific cohort",flush=True)


if __name__ == "__main__":
    main()
