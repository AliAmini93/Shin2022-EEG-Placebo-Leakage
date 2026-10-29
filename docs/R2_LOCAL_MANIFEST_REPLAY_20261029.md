# R2 Hypothesis A frozen manifest replay: owner-local receipt (2026-10-29)

**Gate: R2_STANDALONE_MANIFEST_EXACT_PASS.** This is a transcript-backed
report from the researcher's local Ubuntu execution, not a claim that the
GitHub-hosted CI had access to the original Shin2022 EEG files.

## Verification

- Mandatory fixed research Python interpreter used; no new environment.
- 12/12 Python unit and synthetic/integration tests passed on local Ubuntu.
- Input frozen feature-cache SHA256 **PASS**:
  `66f9344fbc0a054253bcbbd719098bff7b4807a6d9de339724c9e1e7565a6075`
- Inclusion registry SHA256 (manifest report):
  `1a355d5da968a0faa13dd3c88d29dbcf4f2920be6116f4a4f07dee843b6586ce`
- Recomputed complete trial-level manifest SHA256 **EXACT PASS**:
  `20c5a8fe49ffae85e085e0adae3153dda8792d21c55a28d83289a910c44ce9a4`.
  The manifest itself remains **LOCAL ONLY**, not in this repository.
- Scored population: **2,188 distinct trials** across ten participants; five repeats produce **10,940 scored instances per scheme**.
- `WS5F`: 250 folds, 798 exposed scored instances, 0 in CLEAN/PLACEBO/SHAM, 0 clone fallbacks; within-training clone count (sham) 4.
- `LORO`: 550 folds, 1,015 exposed scored instances, 0 in CLEAN/PLACEBO/SHAM, 0 clone fallbacks; within-training clone count (sham) 23.
- Gate output: `R2_STANDALONE_MANIFEST_EXACT_PASS`.
- Checkout HEAD used for exact replay: `74f9b873a41950e245aeea020c299a5d26bf4a68`.

## Boundary

These results establish exact replay of the **R2 original-manifest construction**
on the frozen original R1 feature cache and registry. They do **not** establish
independent raw-data-to-cache reconstruction, TS+LR/EEGNet model re-training,
or final paper-number reproduction.

Next gate: frozen TS+LR model fits (M3) on the validated local cache; after
model consistency checks, a bounded EEGNet fit (M6) followed by the full
frozen schedule only if all invariants remain valid.

**Privacy:** no raw EEG, per-participant scores, path-level registry,
trial-level manifest, or individual predictions were committed.
