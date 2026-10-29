# Cached-data reproduction: R2 Hypothesis A

## Scope

This folder ports the original, frozen Hypothesis-A **cache-level** computation as a standalone Python package. It does **not** download Shin2022, decode BCI2000 files, generate the R1 feature cache, or distribute the local trial-index manifest. All such materials remain local to the researcher.

- `pipeline.dataset`: source-cache rows and path enumeration, path-keyed aliases.
- `pipeline.manifest`: five-repeat WS-5F and LORO LEAKED / CLEAN / PLACEBO folds; sample-hash checks. It produces a trial-level manifest **locally only**.
- `pipeline.core` and `pipeline.models`: frozen TS+LR / EEGNet-8,2 schedules and participant-level statistical routines.
- `pipeline.runner`: 959 TS+LR / 1,725 EEGNet historical fits when run on the frozen data; do not presume a fresh run must have this same fit count unless all cache/plan invariants pass.
- `pipeline.analysis`: original four-contrast Holm and magnitude gates; **exports only aggregated outcomes**, with single-person sham exposed only as booleans.

The port is designed to preserve the scientific algorithm and hard-stop gates. The upstream originals contained fixed absolute paths and ad-hoc `sys.path` modifications; these were replaced with required CLI paths and package-relative imports. Original R2 results are **not** recalculated by adding these files.

## Local inputs (all must remain private)

1. Original R1 `feature_cache_v1.npz`, whose SHA-256 must equal `66f9344fbc0a054253bcbbd719098bff7b4807a6d9de339724c9e1e7565a6075`; contains `cov830_W1`, `eeg440_W1`, `w1_hash`, `label` and cache registry columns.
2. Original R1 `S2_CUE_TASK_MASK_REGISTRY.json` from the private study; each path row needs `path`, `subject`, `annex_sha256`, `cue_task_record_mask`, and `protocols` fields.
3. Unchanged R2 `R2_FROZEN_ANALYSIS_PLAN.json`. It records the manifest hash and the original evaluation design.
4. A **private output directory outside the public repository** for trial-index manifests, predictions and per-person model results.

This is computational independence from the *original source repository*, not independence from original scientific inputs. On fresh source data one must first reconstruct and independently validate the R1 feature cache and registry; this is **not implemented yet**.

## Exact CLI entry points

Use the existing research Python interpreter, specified by the workstation policy. Do not create another venv. In the commands below `PY` means that interpreter, and `CACHE`, `REGISTRY`, `PLAN` and `OUT` are locally configured absolute paths outside the public checkout.

```bash
"$PY" -m shin2022_leakage.pipeline.manifest \
  --cache "$CACHE" --registry "$REGISTRY" \
  --out "$OUT/folds.json" --summary "$OUT/fold_summary.json"

"$PY" -m shin2022_leakage.pipeline.runner \
  --cache "$CACHE" --manifest "$OUT/folds.json" \
  --plan "$PLAN" --model M3 --out "$OUT/m3_private.json"

"$PY" -m shin2022_leakage.pipeline.runner \
  --cache "$CACHE" --manifest "$OUT/folds.json" \
  --plan "$PLAN" --model M6 --out "$OUT/m6_private.json"

"$PY" -m shin2022_leakage.pipeline.analysis \
  --m3 "$OUT/m3_private.json" --m6 "$OUT/m6_private.json" \
  --plan "$PLAN" --out "$OUT/aggregates.json"
```

Install the package in the **same existing environment** for these module commands with `"$PY" -m pip install -e '.[replication]'` only if required dependencies are missing and installation is authorized. The project's existing source-checkout launcher can be used without installation only for `validate`, `summary` and `recompute`.

## Safety and confirmation stages

1. Test code and import paths on synthetic cache/registry fixtures, without personal data.
2. Run manifest-only on genuine local cache; verify the **full** frozen manifest SHA and the 798/1,015 exposures.
3. Run M3 TS+LR first; compare precise aggregate levels and the number of unique fits.
4. Only after invariants and numerical comparison, run M6 EEGNet on the local RTX 5090 with original seeds, fixed 100 epochs and conservative GPU sharing; respect owner's GPU jobs.
5. Run aggregate `analysis` and compare all primary values, their CIs, 4 Holm-adjusted p-values and negative controls.

A pass at steps 1–3 is **not** evidence that 1,725 EEGNet training fits reproduced. Do not publish participant-level JSON outputs or cache data.

## Scientific limitations

- The R2 Hypothesis A is a pre-specified re-analysis of already-viewed Shin data, *not* externally blinded confirmation.
- `LEAKED` versus `PLACEBO` does not isolate exact-twin memorization causally, because the distinct-training-trial count remains unequal.
- The published `10,940` test-instance count represents `2,188` distinct scored trials seen in five repeats, not independent trials.
- A public root commit in this repository was created independently of the original research repository; no private Git history is imported.
