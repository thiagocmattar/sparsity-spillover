"""Token-cache identity, deterministic block order, and full validation coverage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any
from uuid import uuid4


TRAINING_ORDER_SCHEME = "seeded_complete_block_permutation_wrap_v1"
FULL_VALIDATION_DOCUMENTS = 500
FULL_VALIDATION_TOKENS = 693_668
FULL_VALIDATION_COMPLETE_BLOCKS = 338
FULL_VALIDATION_EVALUATED_TOKENS = 692_224
FULL_VALIDATION_EXCLUDED_TAIL = 1_444
FULL_VALIDATION_SHA256 = "51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451"


def write_token_cache(
    rows: Iterable[Mapping[str, Any]],
    *,
    tokenizer: Any,
    output_dir: str | Path,
    dataset_name: str,
    dataset_revision: str,
    split: str,
    text_column: str = "text",
    block_size: int = 2048,
    append_eos: bool = True,
) -> dict[str, Any]:
    """Stream documents into a durable int32 cache and publish metadata last."""

    import numpy as np

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokens_path = output / "tokens.int32.bin"
    metadata_path = output / "metadata.json"
    if tokens_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Token cache already exists: {output}")
    if append_eos and tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must provide eos_token_id when append_eos is true.")

    temporary = output / f".{tokens_path.name}.{uuid4().hex}.tmp"
    documents = tokens = 0
    buffer: list[int] = []
    try:
        with temporary.open("xb") as handle:
            for row in rows:
                text = row.get(text_column)
                if not text:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False)
                if append_eos:
                    ids.append(tokenizer.eos_token_id)
                buffer.extend(ids)
                documents += 1
                tokens += len(ids)
                if len(buffer) >= 1_000_000:
                    np.asarray(buffer, dtype=np.int32).tofile(handle)
                    buffer.clear()
            if buffer:
                np.asarray(buffer, dtype=np.int32).tofile(handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(tokens_path)
    finally:
        temporary.unlink(missing_ok=True)

    complete_blocks, excluded_tail = divmod(tokens, int(block_size))
    metadata = {
        "dataset_name": dataset_name,
        "dataset_revision": dataset_revision,
        "split": split,
        "text_column": text_column,
        "tokenizer_name": getattr(tokenizer, "name_or_path", None),
        "documents": documents,
        "tokens": tokens,
        "tokens_bytes": tokens_path.stat().st_size,
        "tokens_sha256": file_sha256(tokens_path),
        "tokens_path": str(tokens_path),
        "dtype": "int32",
        "append_eos": bool(append_eos),
        "block_size": int(block_size),
        "complete_blocks": complete_blocks,
        "evaluated_complete_block_tokens": complete_blocks * int(block_size),
        "excluded_tail_tokens": excluded_tail,
    }
    _atomic_json(metadata_path, metadata)
    return metadata


def verify_token_cache(metadata: Mapping[str, Any], *, base: str | Path | None = None) -> Path:
    if metadata.get("dtype") != "int32":
        raise ValueError("Token cache must use int32.")
    token_count = _nonnegative_int(metadata.get("tokens"), "tokens")
    token_bytes = _nonnegative_int(metadata.get("tokens_bytes"), "tokens_bytes")
    if token_bytes != token_count * 4:
        raise ValueError("Token byte count does not equal tokens * 4.")
    path = Path(str(metadata.get("tokens_path", "")))
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    path = path.resolve()
    if path.stat().st_size != token_bytes:
        raise ValueError("Token file size differs from metadata.")
    expected = metadata.get("tokens_sha256")
    if not isinstance(expected, str) or file_sha256(path) != expected:
        raise ValueError("Token file SHA-256 differs from metadata.")
    return path


def require_full_minipile_validation(metadata: Mapping[str, Any]) -> None:
    """Check the known 500-document validation baseline without making it universal."""

    expected = {
        "documents": FULL_VALIDATION_DOCUMENTS,
        "tokens": FULL_VALIDATION_TOKENS,
        "tokens_sha256": FULL_VALIDATION_SHA256,
        "block_size": 2048,
        "split": "validation",
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    complete, tail = divmod(int(metadata.get("tokens", -1)), int(metadata.get("block_size", 0) or 1))
    if complete != FULL_VALIDATION_COMPLETE_BLOCKS:
        mismatches.append("complete_blocks")
    if complete * 2048 != FULL_VALIDATION_EVALUATED_TOKENS:
        mismatches.append("evaluated_complete_block_tokens")
    if tail != FULL_VALIDATION_EXCLUDED_TAIL:
        mismatches.append("excluded_tail_tokens")
    if mismatches:
        raise ValueError("Full MiniPile validation identity mismatch: " + ", ".join(mismatches))


def complete_block_starts(token_count: int, block_size: int) -> list[int]:
    if token_count < 0 or block_size <= 0:
        raise ValueError("Token count must be nonnegative and block size positive.")
    return [index * block_size for index in range(token_count // block_size)]


def build_training_schedule(
    np: Any,
    *,
    token_count: int,
    block_size: int,
    max_steps: int,
    gradient_accumulation_steps: int,
    micro_batch_size: int,
    seed: int,
) -> tuple[Any, str, dict[str, Any]]:
    complete_blocks, tail = divmod(int(token_count), int(block_size))
    if complete_blocks <= 0:
        raise ValueError("Training cache has no complete block.")
    if min(max_steps, gradient_accumulation_steps, micro_batch_size) <= 0:
        raise ValueError("Training schedule dimensions must be positive.")
    sequences_per_update = int(gradient_accumulation_steps) * int(micro_batch_size)
    scheduled_blocks = int(max_steps) * sequences_per_update
    metadata = {
        "scheme": TRAINING_ORDER_SCHEME,
        "seed": int(seed),
        "token_count": int(token_count),
        "block_size": int(block_size),
        "complete_blocks": complete_blocks,
        "excluded_tail_tokens": tail,
        "max_steps": int(max_steps),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "micro_batch_size": int(micro_batch_size),
        "sequences_per_update": sequences_per_update,
        "scheduled_blocks": scheduled_blocks,
        "wrapped_blocks": max(0, scheduled_blocks - complete_blocks),
    }
    permutation = np.random.default_rng(int(seed)).permutation(complete_blocks).astype(np.int64, copy=False)
    order = np.resize(permutation, scheduled_blocks)
    starts = (order * int(block_size)).reshape(
        int(max_steps), int(gradient_accumulation_steps), int(micro_batch_size)
    )
    return starts, integer_array_sha256(np, starts, metadata), metadata


def integer_array_sha256(np: Any, values: Any, metadata: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\n")
    digest.update(np.asarray(values, dtype="<i8").tobytes(order="C"))
    return digest.hexdigest()


def index_sequence_sha256(indices: Iterable[int], metadata: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\n")
    for index in indices:
        digest.update(struct.pack("<q", int(index)))
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer.")
    return value
