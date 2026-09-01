# Canonical Run 018 initialization

This directory holds the one approved, locally generated, random-pretraining
initialization for every Run 018 condition. It is not a released Pythia
checkpoint. `00_generate_initialization.py` constructed the checked-in
Pythia-70M architecture without calling `from_pretrained` for model weights,
applied the frozen Run 017 small-init/Wang-init recipe at seed 1234, and then
proved an exact safetensors round trip.

`metadata.json` is the tracked provenance record. The two binary files are
deliberately ignored by Git and remain required launch inputs:

- `pythia70m-seed1234.safetensors`: 281,715,344 bytes, SHA-256
  `024e01975e1a52ead00340afd7a5c3f0b7c2fa0542d9dd5998e648ec14f73501`;
- `pythia70m-seed1234-rng.pt`: 14,823 bytes, SHA-256
  `ff839f490cbbbec528181113451802f52c734fb45ae693fc800991bc2be36762`.

The model contains 76 tensors and 281,706,496 tensor bytes. Strict loading
must realize parameter SHA-256
`e8b8d8e48880f8ff25e421ed29b04a81eb417300f2b4a01a8c4d56f2591a1062`.
The RNG artifact restores the trusted post-initialization Python, NumPy, and
Torch CPU states only after its file hash is verified. The worker seeds the
CUDA RNG to 1234 before loading and does not replace that CUDA state.

The generator refuses to overwrite any of these three files. Regeneration is
therefore a new explicit provenance action, not part of remote setup or a
worker retry.
