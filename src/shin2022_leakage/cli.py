"""Python CLI; does not require raw EEG or write data into the repository."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .audit import describe, load_results, validate_results
from .stats import summarize_contrasts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="shin2022-leakage",
        description="Verify pooled results or recompute participant-level paired statistics locally.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate", help="Validate the packaged pooled numerical receipt")
    check.add_argument("--input", type=Path, help="Optional alternative aggregate JSON")
    report = sub.add_parser("summary", help="Show the study's pooled numerical results")
    report.add_argument("--input", type=Path, help="Optional alternative aggregate JSON")
    run = sub.add_parser("recompute", help="Recompute paired statistical summaries from a local-only JSON")
    run.add_argument("--input", type=Path, required=True, help="Local JSON: {'contrasts': {'name': [proportion, ...]}}")
    args = parser.parse_args(argv)
    try:
        if args.command in ("validate", "summary"):
            data = load_results(args.input)
            problems = validate_results(data)
            if problems:
                for problem in problems:
                    print(f"FAIL: {problem}", file=sys.stderr)
                return 1
            print("AGGREGATE_VALIDATION_PASS" if args.command == "validate" else describe(data))
            return 0
        content = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(content, dict) or not isinstance(content.get("contrasts"), dict):
            raise ValueError("Expected JSON object with a 'contrasts' mapping")
        print(json.dumps(summarize_contrasts(content["contrasts"]), indent=2, sort_keys=True))
        return 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
