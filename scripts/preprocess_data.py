#!/usr/bin/env python
"""Normalize raw MovieLens ratings for downstream DVC stages."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.pipeline import preprocess_ratings


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    """Persist the normalized interaction dataset."""
    arguments = build_parser().parse_args()
    dataset = preprocess_ratings(arguments.raw_dir, arguments.output_dir)
    print(f"Preprocessed ratings saved to: {dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
