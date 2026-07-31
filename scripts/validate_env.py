#!/usr/bin/env python
"""Validate the local runtime before executing the ML pipeline."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from configs.settings import load_settings  # noqa: E402

REQUIRED_DISTRIBUTIONS = (
    "dvc",
    "mlflow",
    "numpy",
    "pandas",
    "pydantic-settings",
    "scikit-learn",
    "torch",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the environment-validation command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        help="Write the validation output to this file instead of stdout.",
    )
    return parser


def installed_version(distribution: str) -> str | None:
    """Return a distribution version, or None when it is unavailable."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_packages() -> list[str]:
    """Print package versions and return missing distribution names."""
    missing = []
    for distribution in REQUIRED_DISTRIBUTIONS:
        version = installed_version(distribution)
        print(f"{distribution:<20} {version or 'MISSING'}")
        if version is None:
            missing.append(distribution)
    return missing


def validate_paths() -> list[Path]:
    """Print required data paths and return missing paths."""
    settings = load_settings()
    required_paths = (
        settings.resolved_data_dir / "raw",
        settings.resolved_data_dir / "raw" / "ratings.csv",
        settings.resolved_data_dir / "raw" / "movies.csv",
    )
    missing = []
    for path in required_paths:
        exists = path.exists()
        print(f"{path} {'OK' if exists else 'MISSING'}")
        if not exists:
            missing.append(path)
    return missing


def run_validation() -> int:
    """Run environment checks and return a process exit code."""
    print(f"Python               {sys.version.split()[0]}")
    print("\nDependencies")
    missing_packages = validate_packages()
    print("\nData")
    missing_paths = validate_paths()
    if missing_packages or missing_paths:
        print("\nEnvironment validation failed.")
        return 1
    print("\nEnvironment validation passed.")
    return 0


def main() -> int:
    """Validate the environment and optionally persist a report."""
    arguments = build_parser().parse_args()
    if arguments.report is None:
        return run_validation()
    report = (
        arguments.report if arguments.report.is_absolute() else ROOT / arguments.report
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8") as stream, redirect_stdout(stream):
        return run_validation()


if __name__ == "__main__":
    raise SystemExit(main())
