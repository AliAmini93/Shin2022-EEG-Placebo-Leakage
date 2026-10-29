# Original recording decoding and score-free cache construction

The raw-to-cache operations were ported from the frozen research scripts, with
all imports localized to this standalone package. They require user-supplied
original recordings, original condition-integrity register and validated file
hashes. The repository contains none of these inputs.

## Modules

- `shin2022_leakage.pipeline.reader`: BCI2000 header, state and signal reader; sensitive header fields are replaced with in-process HMAC digests.
- `shin2022_leakage.pipeline.census`: score-free trial segmentation and pre-fit checks, writing trial-level arrays **outside Git**.
- `shin2022_leakage.pipeline.cache_build`: causal per-record filtering, W1 feature extraction, sample hashes and TS+LR and EEGNet input cache.
- `shin2022_leakage.pipeline.manifest`: frozen split and placebo construction.
- `shin2022_leakage.pipeline.runner`: model fitting with unchanged frozen schedule.
- `shin2022_leakage.pipeline.analysis`: aggregate-only inference.

## How to run with existing authorized source data

All examples assume `PY` is the researcher's mandated existing Python executable,
`REGISTER` is the verified, local original condition-integrity register,
`STORE` contains content-addressed `<SHA256>.dat` original files,
`OUT` is a **private folder outside this public repository**,
and `REGISTRY` is the original frozen inclusion registry.
You must not substitute an independently invented registry and still
call results "frozen exact replication".

```bash
"$PY" -m shin2022_leakage.pipeline.census \
  --register "$REGISTER" --store "$STORE" \
  --derived "$OUT/derived" --output "$OUT/census.json"

"$PY" -m shin2022_leakage.pipeline.cache_build \
  --registry "$REGISTRY" --store "$STORE" \
  --derived "$OUT/derived" --output "$OUT/feature_cache_v1.npz"
```

The original source-cache bytes must be compared to the frozen SHA-256 before
running any model. A new cache produced by these modules might be semantically
equivalent but **cannot be assumed byte-identical** (NPZ headers, NumPy versions,
file ordering). Use the manifest's invariants and held-out numeric comparison,
not a casual hash override.

**Intentionally missing:** the original G0-E1r condition-integrity register
generation and end-to-end validation against the public archive's metadata.
These depend on a separate prior audit. This is a transparent reproducibility
limitation, not a claim that the source-data pipeline has been independently
replicated in this release.

No raw EEG, trial-level arrays or private metadata may be committed to Git.
