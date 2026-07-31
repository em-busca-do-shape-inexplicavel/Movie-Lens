#!/usr/bin/env python
"""Track candidate runs and retrain the validation-selected model."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from configs.settings import Settings, load_settings
from data.pipeline import load_feature_splits
from models.pytorch_recommender import PyTorchRecommender
from tracking.mlflow_tracker import MlflowTrackingOptions, track_training_run
from training.configuration import (
    ExperimentCandidateConfig,
    ExperimentSelectionConfig,
    TrainingPipelineConfig,
    load_experiment_config,
    load_training_config,
)
from training.experiments import (
    CandidateExperimentResult,
    save_experiment_summary,
    select_best_candidate,
)
from training.pipeline import fit_selected_model, train_candidate


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    """Run candidate experiments and retrain the winner."""
    arguments = build_parser().parse_args()
    settings = load_settings()
    base_config = load_training_config(arguments.params)
    experiment = load_experiment_config(arguments.params)
    train, validation, _ = load_feature_splits(arguments.features_dir)
    candidates = _run_candidates(
        train, validation, arguments, base_config, experiment, settings
    )
    winner = select_best_candidate(
        candidates, experiment.selection_metric, experiment.direction
    )
    _persist_winner(train, validation, arguments.output_dir, winner)
    _save_summary(arguments.output_dir, candidates, winner, experiment)
    print(f"Selected candidate: {winner.name} ({winner.run_id})")
    return 0


def _run_candidates(
    train,
    validation,
    arguments: argparse.Namespace,
    base_config: TrainingPipelineConfig,
    experiment: ExperimentSelectionConfig,
    settings: Settings,
) -> list[CandidateExperimentResult]:
    return [
        _run_candidate(
            candidate, train, validation, arguments, base_config.top_k, settings
        )
        for candidate in experiment.candidates
    ]


def _run_candidate(
    candidate: ExperimentCandidateConfig,
    train,
    validation,
    arguments: argparse.Namespace,
    top_k: int,
    settings: Settings,
) -> CandidateExperimentResult:
    config = TrainingPipelineConfig(candidate.model, top_k)
    output_dir = arguments.output_dir / "experiments" / candidate.name
    result = train_candidate(train, validation, output_dir, config)
    tracked = _track_candidate(
        candidate.name, config, result, arguments.params, settings
    )
    return CandidateExperimentResult(candidate.name, config, result, tracked.run_id)


def _track_candidate(name, config, result, params_path, settings):
    options = MlflowTrackingOptions(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=settings.mlflow_experiment_name,
        model_name=settings.mlflow_model_name,
        publish_model=False,
        run_name=f"candidate-{name}",
        extra_tags={"run_kind": "candidate", "candidate": name},
    )
    return track_training_run(
        config=config,
        result=result,
        params_path=params_path,
        options=options,
    )


def _persist_winner(train, validation, output_dir, winner) -> None:
    model = PyTorchRecommender.load(winner.result.artifacts.model)
    assert model.history is not None
    fit_selected_model(train, validation, output_dir, winner.config, model.history)


def _save_summary(output_dir, candidates, winner, experiment) -> None:
    save_experiment_summary(
        output_dir / "experiment_summary.json",
        candidates,
        winner,
        experiment.selection_metric,
        experiment.direction,
    )


if __name__ == "__main__":
    raise SystemExit(main())
