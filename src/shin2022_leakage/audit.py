"""Aggregate-receipt validation: no EEG recordings or individual scores are loaded."""
from __future__ import annotations

import json
import math
from importlib.resources import files
from pathlib import Path
from typing import Any

EXPECTED_MEANS_PP = {
    "TS_LR_WS5F": 2.548170947884417,
    "EEGNet_WS5F": 3.115216160490201,
    "TS_LR_LORO": 2.4287460140428582,
    "EEGNet_LORO": 3.0871599126496854,
}


def load_results(source: str | Path | None = None) -> dict[str, Any]:
    if source is None:
        return json.loads(
            files("shin2022_leakage").joinpath("data/aggregate_results.json").read_text(encoding="utf-8")
        )
    return json.loads(Path(source).read_text(encoding="utf-8"))


def validate_results(data: dict[str, Any]) -> list[str]:
    """Return explicit errors rather than relying on optimized-away asserts."""
    errors: list[str] = []
    def check(ok: bool, reason: str) -> None:
        if not ok:
            errors.append(reason)

    if data.get("schema") != "paper_a_public_aggregate_summary_v1":
        return ["Unsupported aggregate schema"]
    scope = data.get("scope", {})
    check(scope.get("participants_total") == 10, "Expected 10 participants")
    check(scope.get("participants_affected") == 9, "Expected 9 affected participants")
    check(scope.get("unique_scored_trials") == 2188, "Unexpected unique scored-trial count")
    check(scope.get("evaluation_repeats") == 5, "Unexpected repeat count")
    check(scope.get("scored_instances_per_scheme") == 10940, "Unexpected scored-instance count")
    check(scope.get("ws5f_scored_instances_exposed") == 798, "Unexpected WS-5F exposure count")
    check(scope.get("loro_scored_instances_exposed") == 1015, "Unexpected LORO exposure count")
    check(
        scope.get("unique_scored_trials", 0) * scope.get("evaluation_repeats", 0)
        == scope.get("scored_instances_per_scheme"),
        "Scored trials x repeats must equal scored instances",
    )
    primary = data.get("primary_leaked_minus_placebo", {})
    check(set(primary) == set(EXPECTED_MEANS_PP), "Wrong primary contrast keys")
    for key, expected in EXPECTED_MEANS_PP.items():
        if key not in primary:
            continue
        row = primary[key]
        mean = row.get("mean_pp")
        ci = row.get("ci95_t_pp")
        check(isinstance(mean, (float, int)) and math.isfinite(mean)
              and math.isclose(mean, expected, rel_tol=0, abs_tol=1e-9),
              f"{key}: reference mean mismatch")
        check(isinstance(ci, list) and len(ci) == 2
              and all(isinstance(v, (float, int)) and math.isfinite(v) for v in ci)
              and ci[0] < expected < ci[1], f"{key}: invalid 95% confidence interval")
        check(row.get("participants") == row.get("positive_participants") == 9,
              f"{key}: inconsistent participant count")
        check(row.get("p_exact") == 2 / 512, f"{key}: exact p mismatch")
        check(row.get("p_holm4") == 8 / 512, f"{key}: Holm p mismatch")
    placebo = data.get("secondary_placebo_minus_clean", {})
    check(set(placebo) == {"TS_LR_WS5F", "EEGNet_WS5F"}, "Unexpected WS-5F placebo contrasts")
    for key, row in placebo.items():
        ci = row.get("ci90_t_pp")
        check(isinstance(ci, list) and len(ci) == 2
              and all(isinstance(v, (float, int)) and math.isfinite(v) for v in ci)
              and -2 < ci[0] < row.get("mean_pp", -999) < ci[1] < 2
              and row.get("frozen_classification") == "PRACTICALLY_NULL",
              f"{key}: placebo equivalence check failed")
    return errors


def describe(data: dict[str, Any]) -> str:
    errors = validate_results(data)
    if errors:
        raise ValueError("Invalid aggregate report: " + "; ".join(errors))
    lines = [
        "Shin2022 EEG benchmark — aggregate leakage audit",
        "Contrast: LEAKED minus matched PLACEBO, balanced accuracy (percentage points)",
        "",
    ]
    for key, r in data["primary_leaked_minus_placebo"].items():
        a, b = r["ci95_t_pp"]
        lines.append(f"{key:16s}  {r['mean_pp']:+.2f} pp  95% t CI [{a:+.2f}, {b:+.2f}]  "
                     f"Holm p={r['p_holm4']:.6f}")
    lines.extend(["", "Scope: nine affected participants; 2,188 unique scored trials.",
                  "These are reported aggregates, NOT a rerun of EEG training."])
    return "\n".join(lines)
