import json

import numpy as np
import pytest

from sparsity_research.data import (
    FULL_VALIDATION_COMPLETE_BLOCKS,
    FULL_VALIDATION_EVALUATED_TOKENS,
    FULL_VALIDATION_EXCLUDED_TAIL,
    FULL_VALIDATION_SHA256,
    FULL_VALIDATION_TOKENS,
    build_training_schedule,
    complete_block_starts,
    require_full_minipile_validation,
    verify_token_cache,
    write_token_cache,
)


def test_full_validation_contract_is_all_500_documents():
    metadata = {
        "documents": 500,
        "tokens": FULL_VALIDATION_TOKENS,
        "tokens_sha256": FULL_VALIDATION_SHA256,
        "block_size": 2048,
        "split": "validation",
    }
    require_full_minipile_validation(metadata)
    assert FULL_VALIDATION_COMPLETE_BLOCKS == 338
    assert FULL_VALIDATION_EVALUATED_TOKENS == 692_224
    assert FULL_VALIDATION_EXCLUDED_TAIL == 1_444


def test_old_half_validation_is_rejected_by_new_baseline():
    with pytest.raises(ValueError, match="mismatch"):
        require_full_minipile_validation(
            {
                "documents": 250,
                "tokens": 311_739,
                "tokens_sha256": "x",
                "block_size": 2048,
                "split": "validation",
            }
        )


def test_training_schedule_is_deterministic_and_hashes_realized_starts():
    first, first_hash, metadata = build_training_schedule(
        np,
        token_count=10 * 8,
        block_size=8,
        max_steps=2,
        gradient_accumulation_steps=1,
        micro_batch_size=3,
        seed=7,
    )
    second, second_hash, _ = build_training_schedule(
        np,
        token_count=10 * 8,
        block_size=8,
        max_steps=2,
        gradient_accumulation_steps=1,
        micro_batch_size=3,
        seed=7,
    )
    assert np.array_equal(first, second)
    assert first_hash == second_hash
    assert metadata["scheduled_blocks"] == 6
    assert len(set(first.reshape(-1).tolist())) == 6


def test_complete_block_starts_excludes_tail():
    assert complete_block_starts(10, 4) == [0, 4]


class _Tokenizer:
    eos_token_id = 9
    name_or_path = "fake-tokenizer"

    @staticmethod
    def encode(text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) % 7 for character in text]


def test_token_cache_is_durable_and_byte_verified(tmp_path):
    output = tmp_path / "cache"
    metadata = write_token_cache(
        [{"text": "ab"}, {"text": "c"}, {"text": ""}],
        tokenizer=_Tokenizer(),
        output_dir=output,
        dataset_name="test",
        dataset_revision="abc",
        split="validation",
        block_size=2,
    )
    assert metadata["documents"] == 2
    assert metadata["tokens"] == 5
    assert metadata["complete_blocks"] == 2
    assert metadata["excluded_tail_tokens"] == 1
    assert verify_token_cache(metadata) == (output / "tokens.int32.bin").resolve()
    assert json.loads((output / "metadata.json").read_text())["tokens_sha256"] == metadata["tokens_sha256"]

