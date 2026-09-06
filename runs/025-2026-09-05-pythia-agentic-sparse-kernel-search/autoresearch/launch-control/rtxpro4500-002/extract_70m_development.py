"""Verify and exclusively extract the sealed 70M development payload on the Pod."""

from __future__ import annotations

import datetime
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import time
import traceback


ROOT = Path("/workspace/run025-autoresearch-rtxpro4500-001/sparsity-spillover")
RUN = ROOT / "runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
ARCHIVE = Path(
    "/workspace/run025-autoresearch-rtxpro4500-001/relay-70m-development-001/"
    "run025-70m-development.tar"
)
ARCHIVE_BYTES = 1_690_552_320
ARCHIVE_SHA256 = "fa74f92130fd511f4f98206cf2ca9d2446afa7fe8ca9b20ea5d21a3f394deb6c"
INVENTORY_MEMBER = (
    "runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/"
    "prelaunch/transfer-70m-development.json"
)
RESULT = RUN / "artifacts/transfer-70m-development-002"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_target(name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive path: {name}")
    target = ROOT.joinpath(*relative.parts)
    resolved_root = ROOT.resolve()
    if not target.resolve().is_relative_to(resolved_root):
        raise ValueError(f"Archive path escapes repository: {name}")
    return target


def write_member(stream, target: Path, expected: dict) -> str:
    if target.exists():
        if (
            not target.is_file()
            or target.stat().st_size != expected["bytes"]
            or digest(target) != expected["sha256"]
        ):
            raise FileExistsError(f"Existing target has another identity: {target}")
        return "already_exact"
    target.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    written = 0
    with target.open("xb") as output:
        while chunk := stream.read(8 * 1024 * 1024):
            output.write(chunk)
            hasher.update(chunk)
            written += len(chunk)
    if written != expected["bytes"] or hasher.hexdigest() != expected["sha256"]:
        raise ValueError(f"Extracted member identity mismatch: {target}")
    return "extracted"


def main() -> None:
    if RESULT.exists():
        raise FileExistsError("Transfer verification attempt already exists")
    RESULT.mkdir(parents=True)
    started = time.monotonic()
    try:
        if ARCHIVE.stat().st_size != ARCHIVE_BYTES or digest(ARCHIVE) != ARCHIVE_SHA256:
            raise ValueError("Received archive identity mismatch")
        with tarfile.open(ARCHIVE, "r:") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)) or INVENTORY_MEMBER not in names:
                raise ValueError("Invalid archive membership or missing inventory")
            for member in members:
                if not member.isfile():
                    raise ValueError(f"Non-file archive member: {member.name}")
                safe_target(member.name)
            inventory_stream = archive.extractfile(INVENTORY_MEMBER)
            if inventory_stream is None:
                raise ValueError("Missing inventory stream")
            inventory_bytes = inventory_stream.read()
            inventory = json.loads(inventory_bytes)
            if (
                inventory["size"] != "70m"
                or inventory["partition"] != "development"
                or len(inventory["conditions"]) != 6
            ):
                raise ValueError("Unexpected transfer scope")
            expected = {row["path"]: row for row in inventory["files"]}
            if set(names) != set(expected) | {INVENTORY_MEMBER}:
                raise ValueError("Archive members differ from the embedded allowlist")
            for row in inventory["required_existing"].values():
                target = safe_target(row["path"])
                if (
                    not target.is_file()
                    or target.stat().st_size != row["bytes"]
                    or digest(target) != row["sha256"]
                ):
                    raise ValueError(f"Persistent input identity mismatch: {row['path']}")

            actions = {}
            inventory_target = safe_target(INVENTORY_MEMBER)
            inventory_expected = {
                "bytes": len(inventory_bytes),
                "sha256": hashlib.sha256(inventory_bytes).hexdigest(),
            }
            actions[INVENTORY_MEMBER] = write_member(
                io.BytesIO(inventory_bytes),
                inventory_target,
                inventory_expected,
            )

            for name, row in expected.items():
                stream = archive.extractfile(name)
                if stream is None:
                    raise ValueError(f"Missing archive stream: {name}")
                actions[name] = write_member(stream, safe_target(name), row)

        result = {
            "archive": {
                "bytes": ARCHIVE_BYTES,
                "sha256": ARCHIVE_SHA256,
                "relay_elapsed_seconds": 60.1,
                "relay_exit_code_note": (
                    "PowerShell process exit fields were null; received archive identity "
                    "is authoritative and exact"
                ),
            },
            "conditions": inventory["conditions"],
            "files": len(expected),
            "extracted": sum(value == "extracted" for value in actions.values()),
            "already_exact": sum(value == "already_exact" for value in actions.values()),
            "elapsed_seconds": time.monotonic() - started,
            "complete": True,
            "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with (RESULT / "verification.json").open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2, sort_keys=True)
            output.write("\n")
        print(json.dumps(result, sort_keys=True))
    except BaseException as error:
        (RESULT / "failed.txt").write_text(str(error) + "\n", encoding="utf-8")
        (RESULT / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
