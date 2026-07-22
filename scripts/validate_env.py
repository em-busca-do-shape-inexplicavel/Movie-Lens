#!/usr/bin/env python
"""Validate the local runtime before executing the ML pipeline."""

from __future__ import annotations

import importlib.metadata
import sys
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


def main() -> int:
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


if __name__ == "__main__":
    raise SystemExit(main())
