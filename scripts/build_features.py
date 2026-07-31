#!/usr/bin/env python
"""Build leakage-safe temporal features for recommendation training."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.pipeline import build_temporal_features


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    """Persist train, validation, and untouched test tables."""
    arguments = build_parser().parse_args()
    artifacts = build_temporal_features(arguments.dataset, arguments.output_dir)
    print(f"Training features saved to: {artifacts.train}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
