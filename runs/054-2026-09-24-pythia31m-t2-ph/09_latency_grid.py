"""Measure all six verified final checkpoints in three fresh processes each."""
import argparse
import json
import subprocess
import sys
from run_config import RUN_DIR, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    if not args.tag.replace("-", "").isalnum():
        raise ValueError("Simple unique tag required")
    subprocess.run([sys.executable, str(RUN_DIR/"03_verify.py")],check=True)
    catalog = json.loads((RUN_DIR/"artifacts/verification.json").read_text())["conditions"]
    outcomes = []
    for row in catalog:
        checkpoint = RUN_DIR/"artifacts/attempts"/row["attempt"]/row["final_checkpoint"]["path"]
        for replicate in (1,2,3):
            attempt = f"{args.tag}-{row['condition']['id']}-r{replicate}"
            command = [sys.executable,str(RUN_DIR/"latency/01_benchmark.py"),"--checkpoint",str(checkpoint),
                       "--condition",row["condition"]["id"],"--attempt",attempt,"--replicate",str(replicate)]
            log = RUN_DIR/"latency/logs"/(attempt+".log")
            log.parent.mkdir(parents=True,exist_ok=True)
            with log.open("x") as handle:
                process = subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,timeout=1200)
            result_path = RUN_DIR/"latency/artifacts/attempts"/attempt/"manifest.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            outcomes.append(dict(attempt=attempt,condition=row["condition"]["id"],replicate=replicate,
                returncode=process.returncode,status=result.get("status","missing"),qualified=result.get("qualified",False)))
            write_json(RUN_DIR/"latency/artifacts"/(args.tag+"-grid.json"),dict(outcomes=outcomes))
            print(json.dumps(outcomes[-1]),flush=True)
    if not all(row["qualified"] and row["returncode"]==0 for row in outcomes):
        raise RuntimeError("At least one retained final process failed qualification")


if __name__ == "__main__":
    main()
