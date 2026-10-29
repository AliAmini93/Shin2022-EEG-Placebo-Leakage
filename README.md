# Recording-Level Leakage in the Shin2022 EEG Benchmark

**Standalone Python tools for checking placebo-controlled leakage results.**

This repository accompanies the research study *Recording-Level Leakage in a Public EEG Benchmark: A Placebo-Controlled Evaluation*. It provides a small, installable Python package for **verifying the published aggregate results** and **recomputing participant-level statistical contrasts from locally supplied input**. It does not distribute human EEG recordings, participant-level results, or the full EEG model-training pipeline.

## Research question

Can recording-level duplicates in a public EEG benchmark inflate within-subject evaluation, beyond effects explained by training-set size, class balance and within-training duplication?

The study compares three arms on the same scored examples: **LEAKED** (test-identical training copies), **CLEAN** (duplicate-grouped), and **PLACEBO** (matched within-training duplicates without test-identical copies). The primary contrast is LEAKED minus PLACEBO.

## Results at a glance

| Evaluation | Model | Difference in balanced accuracy | Participant-level 95% t CI |
| --- | --- | ---: | ---: |
| Within-subject five-fold (WS-5F) | Tangent Space + LR | +2.55 pp | [1.94, 3.15] pp |
| Within-subject five-fold (WS-5F) | EEGNet | +3.12 pp | [2.23, 4.00] pp |
| Leave-one-run-out (LORO) | Tangent Space + LR | +2.43 pp | [1.76, 3.10] pp |
| Leave-one-run-out (LORO) | EEGNet | +3.09 pp | [2.02, 4.15] pp |

All four primary comparisons were positive for nine affected participants, with four-test Holm-adjusted p = 0.015625. These results apply to this benchmark and should not be interpreted as proof of an entirely isolated causal effect: exposed WS-5F folds retain approximately 2.1% imbalance in *distinct* training trials.

## Repository contents

```text
src/shin2022_leakage/        Python package: validation, statistics and CLI
src/shin2022_leakage/data/   Aggregate-only results, included in the package
tests/                       Deterministic synthetic unit tests
tools/run.py                 Zero-install entry point for source checkouts
docs/METHODS.md              Scope, assumptions and limitations
pyproject.toml               Package metadata and dependencies
```

**No generated figures, datasets or shell scripts are tracked.**

## Quick start

Requirements: Python 3.11+, NumPy and SciPy. From a clone, without installing the package:

```bash
python tools/run.py validate
python tools/run.py summary
python -m unittest discover -s tests -v
```

For a conventional install in an existing environment:

```bash
python -m pip install -e .
shin2022-leakage validate
shin2022-leakage summary
```

You do **not** need the original EEG dataset for these commands. On the authors' research workstation, all commands must use the pre-existing fixed research Python interpreter; do not create an additional environment.

### Recompute statistics from local paired differences

The `recompute` command accepts a **local-only**, untracked JSON file with participant-level paired differences expressed as *proportions* (not percentage points):

```json
{"contrasts":{"example":[0.02,0.03,0.01,0.04]}}
```

```bash
python tools/run.py recompute --input /path/to/local_paired_differences.json
```

Output is aggregate-only JSON written to standard output; input values are never uploaded or committed. The recompute command supports exact paired sign-flips (up to 20 participants), participant-level t confidence intervals, seeded bootstrap intervals and Holm multiplicity correction. See [Methods](docs/METHODS.md).

## Reproducibility boundary

**Implemented and testable here:** validating curated aggregate outcomes and recomputing paired summary statistics from supplied differences. **Not included yet:** original EEG import/preprocessing, duplicate-identity registry, fold construction, TS+LR training or EEGNet training. Thus, this is **not** an end-to-end replication of the EEG experiments. Claims of full reproducibility require independent execution of those components on source data.

This was a pre-specified reanalysis of Shin2022 data whose related outcomes had been examined in earlier work; it is not a blind external replication. Other exploratory and external tests are not part of this release.

## Data, ethics and reuse

Only pooled numerical summaries are distributed. Obtain original EEG recordings from their authorized provider under its terms; no participant-level observations are included. Publication, author attribution, data permissions and a code reuse license require the appropriate approvals. **This repository currently has no open-source license**; public visibility alone does not grant permission to reuse the code.

See [Methods and limitations](docs/METHODS.md) for precise interpretation.