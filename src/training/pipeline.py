"""Reusable end-to-end training pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from data.loaders import load_ratings
from data.splitting import temporal_leave_two_out
from evaluation.metrics import evaluate_top_k, mae, rmse
from models.factory import create_model
from models.pytorch_recommender import (
    NeuralRecommenderConfig,
    NeuralTrainingHistory,
    PyTorchRecommender,
)
from training.configuration import TrainingPipelineConfig

MetricValue = str | int | float | bool


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    """Files produced by one pipeline execution."""

    model: Path
    metrics: Path
    config: Path
    selection_history: Path


@dataclass(frozen=True, slots=True)
class TrainingPipelineResult:
    """Metrics and artifact locations returned by the pipeline."""

    metrics: dict[str, MetricValue]
    artifacts: TrainingArtifacts


def run_training_pipeline(
    raw_data_dir: Path,
    output_dir: Path,
    config: TrainingPipelineConfig,
) -> TrainingPipelineResult:
    """Select the epoch, train the final model, and persist results."""
    ratings = load_ratings(raw_data_dir)
    train, validation, test = temporal_leave_two_out(ratings)
    selection_history = _select_best_epoch(train, validation, config.model)
    final_config = replace(config.model, epochs=selection_history.best_epoch)
    final_model = _fit_final_model(train, validation, final_config)
    metrics = _evaluate_final_model(final_model, test, config.top_k)
    metrics.update(_pipeline_metadata(train, validation, test, selection_history))
    artifacts = _save_artifacts(
        output_dir, final_model, metrics, final_config, selection_history
    )
    return TrainingPipelineResult(metrics=metrics, artifacts=artifacts)


def _select_best_epoch(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: NeuralRecommenderConfig,
) -> NeuralTrainingHistory:
    model = create_model("pytorch", config=config).fit(train, validation)
    assert model.history is not None
    return model.history


def _fit_final_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: NeuralRecommenderConfig,
) -> PyTorchRecommender:
    train_validation = pd.concat([train, validation], ignore_index=True)
    return create_model("pytorch", config=config).fit(train_validation)


def _evaluate_final_model(
    model: PyTorchRecommender, test: pd.DataFrame, top_k: int
) -> dict[str, MetricValue]:
    predictions = model.predict_pairs(test)
    ranking = _evaluate_ranking(model, test, top_k)
    actual = test["rating"].to_numpy(float)
    return {
        "model": "pytorch",
        "test_rmse": rmse(actual, predictions),
        "test_mae": mae(actual, predictions),
        **_ranking_metric_values(ranking, top_k),
    }


def _evaluate_ranking(
    model: PyTorchRecommender, test: pd.DataFrame, top_k: int
) -> pd.Series:
    ranking, _, _ = evaluate_top_k(
        test,
        model.recommend,
        model.catalog_size,
        k=top_k,
        relevance_threshold=model.config.relevance_threshold,
    )
    return ranking


def _ranking_metric_values(ranking: pd.Series, top_k: int) -> dict[str, MetricValue]:
    return {
        f"test_precision@{top_k}": float(ranking[f"precision@{top_k}"]),
        f"test_recall@{top_k}": float(ranking[f"recall@{top_k}"]),
        f"test_ndcg@{top_k}": float(ranking[f"ndcg@{top_k}"]),
        f"test_hit_rate@{top_k}": float(ranking[f"hit_rate@{top_k}"]),
        f"test_coverage@{top_k}": float(ranking[f"catalog_coverage@{top_k}"]),
    }


def _pipeline_metadata(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    history: NeuralTrainingHistory,
) -> dict[str, MetricValue]:
    best_validation = history.validation_rmse[history.best_epoch - 1]
    return {
        "selected_epochs": history.best_epoch,
        "selection_epochs_ran": len(history.train_rmse),
        "stopped_early": history.stopped_early,
        "best_validation_rmse": best_validation,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }


def _save_artifacts(
    output_dir: Path,
    model: PyTorchRecommender,
    metrics: dict[str, MetricValue],
    config: NeuralRecommenderConfig,
    history: NeuralTrainingHistory,
) -> TrainingArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = TrainingArtifacts(
        model=output_dir / "model.pt",
        metrics=output_dir / "metrics.json",
        config=output_dir / "config.json",
        selection_history=output_dir / "selection_history.json",
    )
    model.save(artifacts.model)
    _save_json(artifacts.metrics, metrics)
    _save_json(artifacts.config, asdict(config))
    _save_json(artifacts.selection_history, asdict(history))
    return artifacts


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
