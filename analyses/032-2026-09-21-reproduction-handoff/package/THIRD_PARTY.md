# Third-party source notices

The project license is intentionally undecided. Existing third-party licenses
continue to apply to their respective material.

- Sakana/Sparser-derived kernel work: original notice in
  `LICENSES/sparser-LICENSE`. The final implementation is a Pythia adaptation,
  not an unchanged upstream performance benchmark.
- FlashAttention: pinned commit
  `e2743ab5b3803bb672b16437ba98a3b1d4576c50`; notices are in `LICENSES/` and
  beside derived attention files. Source: https://github.com/Dao-AILab/flash-attention
- CUTLASS: pinned commit `7127592069c2fe01b041e174ba4345ef9b279671`;
  notice in `LICENSES/`. Source: https://github.com/NVIDIA/cutlass
- Pythia architectures/tokenizer and MiniPile are fetched from their pinned
  upstream revisions. Their distribution terms remain with their providers.

The dependency fetcher checks archive SHA-256 values before extracting files.
Full vendor trees, compiled extensions, model weights and datasets are omitted.
