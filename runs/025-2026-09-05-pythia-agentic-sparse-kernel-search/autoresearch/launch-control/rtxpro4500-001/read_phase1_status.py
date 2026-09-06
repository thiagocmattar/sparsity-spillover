"""Emit a compact, read-only status summary for the live phase-1 worker."""

from __future__ import annotations

import json
import hashlib
import pathlib
import subprocess


RUN = pathlib.Path(
    "/workspace/run025-autoresearch-rtxpro4500-001/sparsity-spillover/"
    "runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
)
CONTROL = RUN / "artifacts/phase1-primitives-rtxpro4500-001"


result = {
    "checked_utc": subprocess.check_output(
        ["date", "-u", "+%FT%TZ"], text=True
    ).strip()
}
processes = subprocess.run(
    ["pgrep", "-af", "phase1-primitives.sh|k003/gpu_probe.py|compute-sanitizer"],
    capture_output=True,
    text=True,
).stdout.splitlines()
result["phase1_processes"] = len(
    [line for line in processes if "pgrep -af" not in line]
)
result["exit_codes"] = {
    path.name: path.read_text().strip()
    for path in sorted(CONTROL.glob("*.exit-code.txt"))
}
result["phase1_finished"] = (CONTROL / "all-checks-finished-utc.txt").exists()
phase2_script = (
    RUN / "autoresearch/launch-control/rtxpro4500-001/phase2-diagnostics.sh"
)
result["phase2_preflight"] = {
    "script_exists": phase2_script.exists(),
    "script_sha256": (
        hashlib.sha256(phase2_script.read_bytes()).hexdigest()
        if phase2_script.exists()
        else None
    ),
    "control_exists": (
        RUN / "artifacts/phase2-diagnostics-rtxpro4500-001"
    ).exists(),
    "launch_log_exists": (
        RUN / "artifacts/phase2-diagnostics-rtxpro4500-001.launch.log"
    ).exists(),
    "source_sha256": {
        str(path.relative_to(RUN)): (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        )
        for path in (
            RUN / "autoresearch/collect_development_occupancy.py",
            RUN / "autoresearch/dense_probe/probe.py",
            RUN / "autoresearch/numerics/diagnose_layers.py",
        )
    },
}
phase2_control = RUN / "artifacts/phase2-diagnostics-rtxpro4500-001"
phase2_processes = subprocess.run(
    [
        "pgrep",
        "-af",
        (
            "phase2-diagnostics.sh|collect_development_occupancy.py|"
            "dense_probe/probe.py|numerics/diagnose_layers.py"
        ),
    ],
    capture_output=True,
    text=True,
).stdout.splitlines()
result["phase2_status"] = {
    "processes": len(
        [line for line in phase2_processes if "pgrep -af" not in line]
    ),
    "exit_codes": {
        path.name: path.read_text().strip()
        for path in sorted(phase2_control.glob("*.exit-code.txt"))
    },
    "finished": (phase2_control / "all-checks-finished-utc.txt").exists(),
}
for path in sorted(phase2_control.glob("*.log")):
    lines = path.read_text(errors="replace").splitlines()
    if lines:
        result["phase2_status"][f"{path.stem}_tail"] = lines[-2:]

phase2b_control = RUN / "artifacts/phase2b-attention-and-baseline-rtxpro4500-001"
phase2b_processes = subprocess.run(
    [
        "pgrep",
        "-af",
        (
            "phase2b-attention-and-baseline.sh|"
            "collect_attention_tile_occupancy.py|"
            "rtxpro4500-dense-max-autotune-002"
        ),
    ],
    capture_output=True,
    text=True,
).stdout.splitlines()
result["phase2b_status"] = {
    "processes": len(
        [line for line in phase2b_processes if "pgrep -af" not in line]
    ),
    "exit_codes": {
        path.name: path.read_text().strip()
        for path in sorted(phase2b_control.glob("*.exit-code.txt"))
    },
    "finished": (phase2b_control / "all-checks-finished-utc.txt").exists(),
}
for path in sorted(phase2b_control.glob("*.log")):
    lines = path.read_text(errors="replace").splitlines()
    if lines:
        result["phase2b_status"][f"{path.stem}_tail"] = lines[-2:]

phase3a_control = RUN / "artifacts/phase3a-k001-model-probes-rtxpro4500-001"
phase3a_processes = subprocess.run(
    ["pgrep", "-af", "phase3a-k001-model-probes.sh|probe_models.py"],
    capture_output=True,
    text=True,
).stdout.splitlines()
result["phase3a_status"] = {
    "processes": len(
        [line for line in phase3a_processes if "pgrep -af" not in line]
    ),
    "exit_codes": {
        path.name: path.read_text().strip()
        for path in sorted(phase3a_control.glob("*.exit-code.txt"))
    },
    "finished": (phase3a_control / "all-checks-finished-utc.txt").exists(),
}
for path in sorted(phase3a_control.glob("*.log")):
    lines = path.read_text(errors="replace").splitlines()
    if lines:
        result["phase3a_status"][f"{path.stem}_tail"] = lines[-2:]

phase3b_control = RUN / "artifacts/phase3b-p0-model-probes-rtxpro4500-001"
phase3b_processes = subprocess.run(
    ["pgrep", "-af", "phase3b-p0-model-probes.sh|probe_models.py"],
    capture_output=True,
    text=True,
).stdout.splitlines()
result["phase3b_status"] = {
    "processes": len(
        [line for line in phase3b_processes if "pgrep -af" not in line]
    ),
    "exit_codes": {
        path.name: path.read_text().strip()
        for path in sorted(phase3b_control.glob("*.exit-code.txt"))
    },
    "finished": (phase3b_control / "all-checks-finished-utc.txt").exists(),
}
for path in sorted(phase3b_control.glob("*.log")):
    lines = path.read_text(errors="replace").splitlines()
    if lines:
        result["phase3b_status"][f"{path.stem}_tail"] = lines[-2:]

k004_directory = RUN / "artifacts/k004-primitives-rtxpro4500-001"
k004_processes = subprocess.run(
    ["pgrep", "-af", "candidates/k004/gpu_probe.py"],
    capture_output=True,
    text=True,
).stdout.splitlines()
result["k004_status"] = {
    "processes": len([line for line in k004_processes if "pgrep -af" not in line]),
    "status": (
        json.loads((k004_directory / "status.json").read_text())
        if (k004_directory / "status.json").exists()
        else None
    ),
    "summary": (
        json.loads((k004_directory / "summary.json").read_text())
        if (k004_directory / "summary.json").exists()
        else None
    ),
}
k004_log = RUN / "artifacts/k004-primitives-rtxpro4500-001.launch.log"
if k004_log.exists():
    lines = k004_log.read_text(errors="replace").splitlines()
    result["k004_status"]["log_tail"] = lines[-4:]

k005_directory = RUN / "artifacts/k005-primitives-rtxpro4500-002"
k005_processes = subprocess.run(
    ["pgrep", "-af", "candidates/k005/gpu_probe.py|run025_k005"],
    capture_output=True,
    text=True,
).stdout.splitlines()
result["k005_status"] = {
    "processes": len([line for line in k005_processes if "pgrep -af" not in line]),
    "status": (
        json.loads((k005_directory / "status.json").read_text())
        if (k005_directory / "status.json").exists()
        else None
    ),
    "summary": (
        json.loads((k005_directory / "summary.json").read_text())
        if (k005_directory / "summary.json").exists()
        else None
    ),
}
k005_log = RUN / "artifacts/k005-primitives-rtxpro4500-002.launch.log"
if k005_log.exists():
    lines = k005_log.read_text(errors="replace").splitlines()
    result["k005_status"]["log_tail"] = lines[-5:]

for label, path in {
    "k001": RUN / "artifacts/k001-memcheck-rtxpro4500-001/summary.json",
    "k003": RUN / "artifacts/k003-memcheck-rtxpro4500-001/summary.json",
    "k002_short": (
        RUN
        / "autoresearch/candidates/k002/artifacts/rtxpro4500-short-001/status.json"
    ),
    "k002_full": (
        RUN
        / "autoresearch/candidates/k002/artifacts/rtxpro4500-contiguous-001/status.json"
    ),
}.items():
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    result[label] = {
        key: data[key]
        for key in (
            "complete",
            "pass",
            "primitive_cases",
            "primitive_failures",
            "stage",
            "cases",
            "all_fixed_gates_passed",
            "completed_cases",
            "total_cases",
            "error",
        )
        if key in data
    }

full = (
    RUN / "autoresearch/candidates/k002/artifacts/rtxpro4500-contiguous-001"
)
timings = {}
for path in sorted(full.glob("timing-*.json")):
    summary = json.loads(path.read_text())["summary"]
    timings[path.stem] = {
        mode: {
            "median_ms": round(row["median_host_ms"], 6),
            "speedup": round(row["paired_geomean_speedup"], 6),
        }
        for mode, row in summary.items()
    }
result["k002_timings"] = timings

log = CONTROL / "k003-memcheck.log"
if log.exists():
    result["k003_log_tail"] = log.read_text(errors="replace").splitlines()[-8:]

print(json.dumps(result, sort_keys=True))
