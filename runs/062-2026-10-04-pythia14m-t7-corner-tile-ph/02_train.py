"""Run one independent, preassigned condition on its visible CUDA device."""
import argparse
from run_config import load_config
from training import run_worker

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',required=True,choices=tuple(load_config()['runpod']['worker_assignments']))
    args = parser.parse_args()
    for path in run_worker(args.worker):
        print(path, flush=True)
