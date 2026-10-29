"""Frozen-style paired statistical functions for local-only participant differences.

Values are supplied as proportions; summaries return proportions. The caller handles
conversion to percentage points for display. No participant values are emitted.
"""
from __future__ import annotations
from itertools import product
import numpy as np
from scipy.stats import t as student_t

BOOT_SEED = 20261027
N_BOOT = 10_000


def _finite_vector(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or a.size < 2 or not np.all(np.isfinite(a)):
        raise ValueError("Expected a finite one-dimensional vector of at least two values")
    return a


def signflip_p(values) -> float:
    """Exact two-sided sign-flip p. No Monte Carlo approximation for n <= 20."""
    d = _finite_vector(values)
    if d.size > 20:
        raise ValueError("Exact enumeration is limited to 20 participants")
    observed = abs(float(d.mean()))
    null = (d * np.array(list(product((1.0, -1.0), repeat=len(d))))).mean(axis=1)
    return float(np.mean(np.abs(null) >= observed - 1e-12))


def t_interval(values, confidence: float = 0.95) -> list[float]:
    d = _finite_vector(values)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    q = student_t.ppf((1 + confidence) / 2, len(d) - 1)
    halfwidth = float(q * d.std(ddof=1) / np.sqrt(len(d)))
    return [float(d.mean() - halfwidth), float(d.mean() + halfwidth)]


def bootstrap_interval(values, confidence: float = 0.95, seed: int = BOOT_SEED) -> list[float]:
    d = _finite_vector(values)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    indices = np.random.default_rng(seed).integers(0, len(d), size=(N_BOOT, len(d)))
    means = d[indices].mean(axis=1)
    pct = 50 * (1 - confidence)
    return [float(x) for x in np.percentile(means, [pct, 100 - pct])]


def holm(p_values: dict[str, float]) -> dict[str, float]:
    if any(not 0 <= p <= 1 or not np.isfinite(p) for p in p_values.values()):
        raise ValueError("p-values must be finite and in [0,1]")
    order = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    last = 0.0
    for i, key in enumerate(order):
        last = max(last, min(1.0, (len(order) - i) * p_values[key]))
        adjusted[key] = last
    return {k: adjusted[k] for k in p_values}


def paired_summary(values) -> dict[str, object]:
    d = _finite_vector(values)
    return {
        "participants": int(d.size),
        "mean_diff": float(d.mean()),
        "ci95_t": t_interval(d),
        "ci90_t": t_interval(d, 0.90),
        "ci95_bootstrap": bootstrap_interval(d),
        "p_exact": signflip_p(d),
        "positive": int(np.sum(d > 0)),
        "negative": int(np.sum(d < 0)),
        "zero": int(np.sum(d == 0)),
    }


def summarize_contrasts(contrasts: dict[str, list[float]]) -> dict[str, object]:
    if not contrasts or len(contrasts) > 20:
        raise ValueError("Provide between 1 and 20 contrasts")
    summaries = {key: paired_summary(value) for key, value in contrasts.items()}
    adjusted = holm({key: float(row["p_exact"]) for key, row in summaries.items()})
    for key, row in summaries.items():
        row["p_holm"] = adjusted[key]
    return {"schema": "paired_statistical_summary_v1", "units": "proportions",
            "contrasts": summaries}
