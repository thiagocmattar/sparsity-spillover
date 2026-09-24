"""Execute one assigned condition after launch approval and remote preflight."""
import argparse
from training import run_worker
from run_config import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=list(load_config()["runpod"]["worker_assignments"]), required=True)
    args = parser.parse_args()
    for path in run_worker(args.worker):
        print(path, flush=True)
