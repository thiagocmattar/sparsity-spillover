# Data identity

| Input | Pinned identity |
|---|---|
| Dataset | `JeanKaddour/minipile` at `18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0` |
| Tokenizer | `EleutherAI/pythia-14m-deduped` at `7386d9a4ae45aef494a6e704910394def3037fc5` |
| Text | `text` field; no added special tokens during encoding; append EOS per document |
| Storage | Source-order concatenation, int32; complete nonoverlapping 2,048-token blocks |

| Split | Documents | Tokens | Complete blocks | Excluded tail |
|---|---:|---:|---:|---:|
| train | 1,000,000 | 1,491,711,416 | 728,374 | 1,464 |
| validation | 500 | 693,668 | 338 | 1,444 |

Token-file SHA-256:

- Train: `da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c`
- Validation: `51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451`

`scripts/prepare_data.py` downloads the pinned public inputs and checks these
identities. Data is never included in the repository. All model sizes use this
same tokenizer and validation set.

Training permutes complete block indices once with seed 1234, consumes that
order, and wraps 714 blocks to fill 712 updates × 1,024 sequences. The resolved
config stores the schedule hash. Evaluation covers all 338 complete validation
blocks, totaling 692,224 input tokens; the excluded tail is always reported.
