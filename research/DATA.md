# Data and Baseline Protocol

## Known pinned inputs

| Input | Identity |
| --- | --- |
| Dataset | `JeanKaddour/minipile` |
| Dataset revision | `18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0` |
| Text field | `text` |
| Training split | `train`, all 1,000,000 documents |
| Validation split | `validation`, all 500 documents |
| Tokenizer | `EleutherAI/pythia-14m-deduped` |
| Tokenizer revision | `7386d9a4ae45aef494a6e704910394def3037fc5` |
| Tokenization | no added special tokens during encode; append EOS per document |
| Token storage | `int32` |
| Sequence length | 2,048 input tokens |

## Verified historical cache facts

These hashes identify caches in the source repository. A new repository must
either copy and verify those exact bytes or rebuild them and record the new
verified identity.

### Training

- Documents: 1,000,000
- Tokens: 1,491,711,416
- Complete blocks: 728,374
- Excluded tail: 1,464 tokens
- SHA-256:
  `da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c`

### Full validation — new default

- Documents: 500
- Tokens: 693,668
- Complete blocks: 338
- Evaluated input tokens: 692,224
- Excluded tail: 1,444 tokens
- SHA-256:
  `51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451`

The full cache follows the dataset's deterministic source order. Do not split it
into tuning and confirmation partitions by default. If a future experiment
needs a holdout, define a new explicit protocol with the user rather than
silently reviving the old split.

## Historical context only

The previous workflow divided the same 500 source documents into two disjoint
250-document partitions. Historical A1/A2 selection loss used 152 complete
sequences (311,296 input tokens); confirmation had 186 complete sequences.
Those identities explain old artifacts but are not the new validation default.

Consequently, a new full-validation loss is not numerically interchangeable
with an old selection-partition loss. Re-evaluate checkpoints under the same
full-validation protocol before presenting a controlled loss comparison.

## Data-order rule

Construct complete non-overlapping training blocks, generate one seeded
permutation of their indices, and consume that exact order. Wrap only if the
approved budget exceeds one permutation. Store the ordered-start hash and all
inputs used to construct it: token count, block size, steps, microbatch,
accumulation, and data-order seed.

## Before using data in a run

Record in the run README and manifest:

- dataset/tokenizer names and immutable revisions;
- document and token counts;
- token-file byte count and SHA-256;
- append-EOS and block-size settings;
- complete blocks and excluded tail;
- exact training-order seed and schedule hash;
- number of validation documents, sequences, and evaluated tokens.

## Open questions

- Whether a future study needs an untouched holdout beyond the full 500-document
  validation split.
- Whether larger model scales should reuse this tokenizer and global batch.
- Which training horizon and seed allocation the next approved experiment will
  use.
