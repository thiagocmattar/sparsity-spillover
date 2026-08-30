#!/usr/bin/env python
"""Build the approved MiniPile token caches directly from pinned HF revisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Mapping
from uuid import uuid4


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sparsity_research.data import file_sha256  # noqa: E402


DATASET_NAME = "JeanKaddour/minipile"
DATASET_REVISION = "18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0"
TOKENIZER_NAME = "EleutherAI/pythia-14m-deduped"
TOKENIZER_REVISION = "7386d9a4ae45aef494a6e704910394def3037fc5"
BLOCK_SIZE = 2_048
CACHE_ROOT = REPO_ROOT / "data" / "tokenized" / "minipile-pythia-14m-full"
EXPECTED = {
    "validation": {
        "documents": 500,
        "tokens": 693_668,
        "tokens_bytes": 2_774_672,
        "tokens_sha256": "51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451",
        "complete_blocks": 338,
        "evaluated_complete_block_tokens": 692_224,
        "excluded_tail_tokens": 1_444,
    },
    "train": {
        "documents": 1_000_000,
        "tokens": 1_491_711_416,
        "tokens_bytes": 5_966_845_664,
        "tokens_sha256": "da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c",
        "complete_blocks": 728_374,
        "evaluated_complete_block_tokens": 1_491_709_952,
        "excluded_tail_tokens": 1_464,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("validation", "train"),
        default=("validation", "train"),
        help="Build validation first so it serves as a fast tokenization-equivalence gate.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--log-every-documents", type=int, default=10_000)
    return parser.parse_args()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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


def exact_fields(metadata: Mapping[str, Any], split: str) -> dict[str, Any]:
    expected = {
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "split": split,
        "text_column": "text",
        "tokenizer_name": TOKENIZER_NAME,
        "tokenizer_revision": TOKENIZER_REVISION,
        "dtype": "int32",
        "append_eos": True,
        "block_size": BLOCK_SIZE,
        **EXPECTED[split],
    }
    return {key: metadata.get(key) for key in expected if metadata.get(key) != expected[key]}


def require_exact(metadata: Mapping[str, Any], split: str, *, tokens_path: Path) -> None:
    mismatches = exact_fields(metadata, split)
    if tokens_path.stat().st_size != EXPECTED[split]["tokens_bytes"]:
        mismatches["actual_file_bytes"] = tokens_path.stat().st_size
    actual_sha256 = file_sha256(tokens_path)
    if actual_sha256 != EXPECTED[split]["tokens_sha256"]:
        mismatches["actual_file_sha256"] = actual_sha256
    if mismatches:
        raise RuntimeError(f"{split} cache differs from the approved identity: {mismatches}")


def existing_exact_cache(split: str) -> dict[str, Any] | None:
    output = CACHE_ROOT / split
    tokens_path = output / "tokens.int32.bin"
    metadata_path = output / "metadata.json"
    if not tokens_path.exists() and not metadata_path.exists():
        return None
    if not tokens_path.exists() or not metadata_path.exists():
        raise RuntimeError(f"Incomplete existing cache must be handled explicitly: {output}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require_exact(metadata, split, tokens_path=tokens_path)
    print(json.dumps({"event": "cache_already_exact", "split": split}, sort_keys=True), flush=True)
    return metadata


def build_split(dataset: Any, tokenizer: Any, split: str, *, batch_size: int, log_every: int) -> dict[str, Any]:
    import numpy as np

    existing = existing_exact_cache(split)
    if existing is not None:
        return existing
    if len(dataset) != EXPECTED[split]["documents"]:
        raise RuntimeError(
            f"{split} document count differs before tokenization: "
            f"{len(dataset)} != {EXPECTED[split]['documents']}"
        )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Pinned tokenizer has no EOS token id.")

    output = CACHE_ROOT / split
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob(".tokens.int32.bin.hf-build-*.tmp"):
        stale.unlink()
    tokens_path = output / "tokens.int32.bin"
    metadata_path = output / "metadata.json"
    temporary = output / f".tokens.int32.bin.hf-build-{uuid4().hex}.tmp"
    started = perf_counter()
    documents = tokens = 0
    buffer: list[int] = []
    digest = hashlib.sha256()

    try:
        with temporary.open("xb") as handle:
            for start in range(0, len(dataset), batch_size):
                texts = [text for text in dataset[start : start + batch_size]["text"] if text]
                encoded = tokenizer(
                    texts,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]
                for ids in encoded:
                    buffer.extend(ids)
                    buffer.append(tokenizer.eos_token_id)
                    documents += 1
                    tokens += len(ids) + 1
                if len(buffer) >= 1_000_000:
                    payload = np.asarray(buffer, dtype="<i4").tobytes(order="C")
                    handle.write(payload)
                    digest.update(payload)
                    buffer.clear()
                if documents and (documents % log_every < batch_size or documents == len(dataset)):
                    elapsed = perf_counter() - started
                    print(
                        json.dumps(
                            {
                                "event": "tokenization_progress",
                                "split": split,
                                "documents": documents,
                                "tokens": tokens,
                                "elapsed_seconds": elapsed,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            if buffer:
                payload = np.asarray(buffer, dtype="<i4").tobytes(order="C")
                handle.write(payload)
                digest.update(payload)
            handle.flush()
            os.fsync(handle.fileno())

        complete_blocks, excluded_tail = divmod(tokens, BLOCK_SIZE)
        metadata = {
            "append_eos": True,
            "block_size": BLOCK_SIZE,
            "complete_blocks": complete_blocks,
            "dataset_name": DATASET_NAME,
            "dataset_revision": DATASET_REVISION,
            "documents": documents,
            "dtype": "int32",
            "evaluated_complete_block_tokens": complete_blocks * BLOCK_SIZE,
            "excluded_tail_tokens": excluded_tail,
            "split": split,
            "text_column": "text",
            "tokenizer_name": TOKENIZER_NAME,
            "tokenizer_revision": TOKENIZER_REVISION,
            "tokens": tokens,
            "tokens_bytes": temporary.stat().st_size,
            "tokens_path": "tokens.int32.bin",
            "tokens_sha256": digest.hexdigest(),
        }
        mismatches = exact_fields(metadata, split)
        if mismatches:
            raise RuntimeError(f"{split} cache differs from the approved identity: {mismatches}")
        temporary.replace(tokens_path)
        atomic_json(metadata_path, metadata)
        require_exact(metadata, split, tokens_path=tokens_path)
        print(
            json.dumps(
                {
                    "event": "cache_published",
                    "split": split,
                    "elapsed_seconds": perf_counter() - started,
                    **EXPECTED[split],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return metadata
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.log_every_documents <= 0:
        raise ValueError("Batch size and logging interval must be positive.")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(
        json.dumps(
            {
                "event": "hf_cache_build_start",
                "dataset": DATASET_NAME,
                "dataset_revision": DATASET_REVISION,
                "tokenizer": TOKENIZER_NAME,
                "tokenizer_revision": TOKENIZER_REVISION,
                "splits": args.splits,
                "cache_root": str(CACHE_ROOT),
                "hf_home": os.environ.get("HF_HOME"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision=TOKENIZER_REVISION)
    results = {}
    for split in args.splits:
        loaded = load_dataset(DATASET_NAME, revision=DATASET_REVISION, split=split)
        results[split] = build_split(
            loaded,
            tokenizer,
            split,
            batch_size=args.batch_size,
            log_every=args.log_every_documents,
        )
        del loaded
    print(
        json.dumps(
            {
                "event": "hf_cache_build_complete",
                "splits": {
                    split: {key: value for key, value in metadata.items() if key in EXPECTED[split]}
                    for split, metadata in results.items()
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
