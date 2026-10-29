"""Zero-install source checkout entry point. Uses the caller's Python interpreter."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shin2022_leakage.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
