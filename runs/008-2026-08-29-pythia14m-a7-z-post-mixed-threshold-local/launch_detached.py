"""Start Run 008 detached without PowerShell's environment enumeration."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    if len(sys.argv) != 6:
        raise SystemExit("usage: launch_detached.py PYTHON ENTRYPOINT REPO_ROOT STDOUT STDERR")
    python, entrypoint, repo_root, stdout_path, stderr_path = sys.argv[1:]
    clean_environment = dict(os.environ)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    with Path(stdout_path).open("ab", buffering=0) as stdout, Path(stderr_path).open(
        "ab", buffering=0
    ) as stderr:
        process = subprocess.Popen(
            [python, entrypoint],
            cwd=repo_root,
            env=clean_environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=flags,
            close_fds=True,
        )
    print(process.pid)


if __name__ == "__main__":
    main()
