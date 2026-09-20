# K016

K016 freezes the development-selected Pythia-70M policy. Dispatch is by declared
architecture topology, never by checkpoint identity, runtime density, or kappa:

- A0: native fallback;
- A1-H: K001 at `h` in layers 4--5;
- A4-Z: K001 at `a,m,h,z` in layers 3--5;
- A7-Z-POST: K001 at `h,z` in layers 3--5.

All final measurements require the complete 338-block validation gate. The six
interior checkpoints are useful retrospective transfer measurements, but are
not a fresh holdout because K009 results on them were inspected earlier.
