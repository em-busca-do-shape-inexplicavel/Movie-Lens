#!/usr/bin/env python
"""Execute the reproducible recommendation training pipeline."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from configs.settings import load_settings
from training.configuration import load_training_config
from training.pipeline import run_training_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the training command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser


def resolve_project_path(path: Path) -> Path:
    """Resolve a CLI path relative to the project root."""
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    """Run training and print machine-readable metrics."""
    arguments = build_parser().parse_args()
    settings = load_settings()
    config = load_training_config(resolve_project_path(arguments.params))
    result = run_training_pipeline(
        settings.resolved_data_dir / "raw",
        resolve_project_path(arguments.output_dir),
        config,
    )
    print(json.dumps(result.metrics, indent=2, ensure_ascii=False))
    print(f"Model saved to: {result.artifacts.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
