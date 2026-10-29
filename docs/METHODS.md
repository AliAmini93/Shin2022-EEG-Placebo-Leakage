# Methods, verification and release boundaries

## Evaluation design

The reported study examines duplicates in a public Shin2022 human EEG benchmark, using the 2,188 distinct scored trials from ten participants, evaluated in five repeats (10,940 scored *instances* per evaluation scheme). Nine participants were exposed to a duplicate-driven training/test crossing. The tenth was a sham/negative-control participant.

- **LEAKED:** path-based folds that allow an exact recording copy in the training data.
- **CLEAN:** remove cross-fold duplicate copies through content grouping.
- **PLACEBO:** match LEAKED training-set size and class counts and add training-only clones, without exposing scored test twins.

The hypothesis-A primary estimate is participant-mean balanced accuracy of LEAKED minus PLACEBO, over **nine affected participants**. Models: Tangent Space + logistic regression (TS+LR) and EEGNet; evaluation protocols: within-subject five-fold (WS-5F) and leave-one-run-out (LORO).

Exact two-sided paired sign-flips compare participant means and are Holm-adjusted across four prespecified primary contrasts. The reported participant-level 95% confidence intervals are t-based. The placebo equivalence criterion uses 90% t confidence intervals within ±2 percentage points; this does **not** mean that all individual placebo effects are zero.

## What the commands establish

`validate` checks the included, manually curated pooled-result receipt for expected contrast labels, observed mean values, participant counts, fold-exposure counts, confidence-interval bounds and multiplicity values. It is **not** an independent calculation from EEG.

`summary` presents those same pooled outcomes.

`recompute` performs exact two-sided paired sign flips for up to 20 participants, participant-bootstrap 95% confidence intervals (10,000 seeded draws), 90%/95% t intervals, and Holm correction. It expects a *local* JSON object with `contrasts` of participant-level **differences expressed as fractions**, and produces **only aggregate statistics**. It does not train models, create experiment folds or derive paired differences from EEG. Do not commit input files containing participant-level values.

Synthetic unit tests cover the aggregate receipt, deliberate corruption, exact paired p-values and output shape. Nothing is downloaded or executed on a GPU.

## Limitations

- Exposure counts 798 (WS-5F) and 1,015 (LORO) count scored instances across repeats, not distinct individual participants.
- Training size and class balance are matched; in exposed WS-5F folds, distinct training-trial counts still differ by about 2.1%. Therefore the experiment does not fully isolate exact-twin memorization as a causal mechanism.
- These are bounded results from one dataset and two models; no generic duplicate-detection method is claimed.
- The R2 design was fixed before its models were fitted but revisited Shin data already viewed in earlier analyses. It must not be represented as blind confirmation or independent external replication.
- No raw EEG, cross-fold raw manifests, individual predictions or model weights are distributed.

## Full reproduction status

The source-data ingestion, signal preprocessing, recording-identity audit, original folds and TS+LR/EEGNet training are **not yet ported**. Consequently the included tools reproduce only aggregate verification and local paired inference, **not the paper's end-to-end numerical experiment**.

## Licensing and public release

Source EEG data, third-party model implementations and study code each require their own permission review. A public repository does not by itself grant an open-source reuse license. Agree on authorship, ethics, funding and release policy before claiming an authorized final software artifact.
