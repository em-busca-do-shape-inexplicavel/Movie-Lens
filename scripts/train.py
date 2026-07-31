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

from configs.settings import Settings, load_settings
from training.configuration import load_training_config
from training.pipeline import TrainingPipelineResult, run_training_pipeline
from tracking.mlflow_tracker import (
    MlflowTrackingOptions,
    MlflowTrackingResult,
    track_training_run,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the training command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser


def resolve_project_path(path: Path) -> Path:
    """Resolve a CLI path relative to the project root."""
    return path if path.is_absolute() else ROOT / path


def _execute_training(
    arguments: argparse.Namespace,
) -> tuple[TrainingPipelineResult, MlflowTrackingResult]:
    settings = load_settings()
    config = load_training_config(resolve_project_path(arguments.params))
    result = run_training_pipeline(
        settings.resolved_data_dir / "raw",
        resolve_project_path(arguments.output_dir),
        config,
    )
    tracking = track_training_run(
        config=config,
        result=result,
        params_path=resolve_project_path(arguments.params),
        options=_tracking_options(settings),
    )
    return result, tracking


def _tracking_options(settings: Settings) -> MlflowTrackingOptions:
    return MlflowTrackingOptions(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=settings.mlflow_experiment_name,
        model_name=settings.mlflow_model_name,
        model_stage=settings.mlflow_model_stage,
        model_alias=settings.mlflow_model_alias,
    )


def _print_summary(
    result: TrainingPipelineResult, tracking: MlflowTrackingResult
) -> None:
    print(json.dumps(result.metrics, indent=2, ensure_ascii=False))
    print(f"Model saved to: {result.artifacts.model}")
    print(f"MLflow run: {tracking.run_id}")
    print(
        f"Registered model: {tracking.model_name} v{tracking.registered_model_version}"
    )


def main() -> int:
    """Run training and print machine-readable metrics."""
    result, tracking = _execute_training(build_parser().parse_args())
    _print_summary(result, tracking)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
