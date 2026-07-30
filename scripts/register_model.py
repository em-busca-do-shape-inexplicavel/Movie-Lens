#!/usr/bin/env python
"""Register an existing MLflow run in the MovieLens model registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from configs.settings import load_settings
from tracking.mlflow_tracker import register_model_version


def build_parser() -> argparse.ArgumentParser:
    """Build the registry command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--alias", default=None)
    return parser


def main() -> int:
    """Register the run and print the created version."""
    arguments = build_parser().parse_args()
    settings = load_settings()
    publication = register_model_version(
        run_id=arguments.run_id,
        model_name=arguments.model_name or settings.mlflow_model_name,
        description=arguments.description
        or "MovieLens PyTorch recommender trained with temporal leave-two-out splitting.",
        stage=arguments.stage or settings.mlflow_model_stage,
        alias=arguments.alias or settings.mlflow_model_alias,
    )
    print(f"Registered model: {publication.name} version {publication.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())