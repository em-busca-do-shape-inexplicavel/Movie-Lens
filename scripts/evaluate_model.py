#!/usr/bin/env python
"""Evaluate the selected model once and publish it to MLflow Registry."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from configs.settings import Settings, load_settings
from models.pytorch_recommender import PyTorchRecommender
from tracking.mlflow_tracker import (
    MlflowTrackingOptions,
    MlflowTrackingResult,
    track_training_run,
)
from training.configuration import TrainingPipelineConfig, load_training_config
from training.pipeline import TrainingPipelineResult, evaluate_saved_model


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    """Evaluate the winner, log the final run, and publish it."""
    arguments = build_parser().parse_args()
    settings = load_settings()
    test = pd.read_parquet(arguments.test_data)
    top_k = load_training_config(arguments.params).top_k
    result = evaluate_saved_model(test, arguments.output_dir, top_k)
    config = _effective_config(result, top_k)
    winner = _winner_name(arguments.output_dir)
    tracking = _track_final(config, result, arguments.params, winner, settings)
    _save_registry_metadata(arguments.output_dir, tracking, winner)
    print(json.dumps(result.metrics, indent=2, ensure_ascii=False))
    return 0


def _effective_config(
    result: TrainingPipelineResult, top_k: int
) -> TrainingPipelineConfig:
    model = PyTorchRecommender.load(result.artifacts.model)
    return TrainingPipelineConfig(model.config, top_k)


def _winner_name(output_dir: Path) -> str:
    summary = json.loads((output_dir / "experiment_summary.json").read_text())
    return str(summary["winner"])


def _track_final(config, result, params_path, winner, settings: Settings):
    options = MlflowTrackingOptions(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=settings.mlflow_experiment_name,
        model_name=settings.mlflow_model_name,
        model_stage=settings.mlflow_model_stage,
        model_alias=settings.mlflow_model_alias,
        run_name=f"final-{winner}",
        extra_tags={"run_kind": "final", "selected_candidate": winner},
    )
    return track_training_run(
        config=config,
        result=result,
        params_path=params_path,
        options=options,
    )


def _save_registry_metadata(
    output_dir: Path, tracking: MlflowTrackingResult, winner: str
) -> None:
    payload = {
        "winner": winner,
        "run_id": tracking.run_id,
        "model_name": tracking.model_name,
        "registered_model_version": tracking.registered_model_version,
    }
    path = output_dir / "registry.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
