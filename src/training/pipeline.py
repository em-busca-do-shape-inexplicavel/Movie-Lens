"""Reusable end-to-end training pipeline."""

from __future__ import annotations

import json
import time
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
    candidate = train_candidate(train, validation, output_dir, config)
    history = _load_history(candidate.artifacts.selection_history)
    fit_selected_model(train, validation, output_dir, config, history)
    return evaluate_saved_model(test, output_dir, config.top_k)


def train_candidate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    output_dir: Path,
    config: TrainingPipelineConfig,
) -> TrainingPipelineResult:
    """Train one candidate and evaluate it only on validation data."""
    started_at = time.perf_counter()
    model = create_model("pytorch", config=config.model).fit(train, validation)
    assert model.history is not None
    metrics = _evaluate_model(model, validation, config.top_k, "validation")
    metrics.update(_selection_metadata(train, validation, model.history))
    metrics["training_seconds"] = round(time.perf_counter() - started_at, 4)
    artifacts = _save_artifacts(output_dir, model, metrics, config.model, model.history)
    return TrainingPipelineResult(metrics, artifacts)


def fit_selected_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    output_dir: Path,
    config: TrainingPipelineConfig,
    history: NeuralTrainingHistory,
) -> TrainingPipelineResult:
    """Retrain the selected candidate on train and validation data."""
    started_at = time.perf_counter()
    final_config = replace(config.model, epochs=history.best_epoch)
    model = _fit_final_model(train, validation, final_config)
    metrics = _selection_metadata(train, validation, history)
    metrics["training_seconds"] = round(time.perf_counter() - started_at, 4)
    artifacts = _save_artifacts(
        output_dir, model, metrics, final_config, history, "training_metrics.json"
    )
    return TrainingPipelineResult(metrics, artifacts)


def evaluate_saved_model(
    test: pd.DataFrame, output_dir: Path, top_k: int
) -> TrainingPipelineResult:
    """Evaluate the selected model once on the untouched test split."""
    model = PyTorchRecommender.load(output_dir / "model.pt")
    metrics = _read_json(output_dir / "training_metrics.json")
    metrics.update(_evaluate_model(model, test, top_k, "test"))
    metrics["test_rows"] = len(test)
    artifacts = _existing_artifacts(output_dir)
    _save_json(artifacts.metrics, metrics)
    return TrainingPipelineResult(metrics, artifacts)


def _fit_final_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: NeuralRecommenderConfig,
) -> PyTorchRecommender:
    train_validation = pd.concat([train, validation], ignore_index=True)
    return create_model("pytorch", config=config).fit(train_validation)


def _evaluate_model(
    model: PyTorchRecommender,
    holdout: pd.DataFrame,
    top_k: int,
    prefix: str,
) -> dict[str, MetricValue]:
    predictions = model.predict_pairs(holdout)
    ranking = _evaluate_ranking(model, holdout, top_k)
    actual = holdout["rating"].to_numpy(float)
    return {
        "model": "pytorch",
        f"{prefix}_rmse": rmse(actual, predictions),
        f"{prefix}_mae": mae(actual, predictions),
        **_ranking_metric_values(ranking, top_k, prefix),
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


def _ranking_metric_values(
    ranking: pd.Series, top_k: int, prefix: str
) -> dict[str, MetricValue]:
    return {
        f"{prefix}_precision@{top_k}": float(ranking[f"precision@{top_k}"]),
        f"{prefix}_recall@{top_k}": float(ranking[f"recall@{top_k}"]),
        f"{prefix}_ndcg@{top_k}": float(ranking[f"ndcg@{top_k}"]),
        f"{prefix}_hit_rate@{top_k}": float(ranking[f"hit_rate@{top_k}"]),
        f"{prefix}_coverage@{top_k}": float(ranking[f"catalog_coverage@{top_k}"]),
    }


def _selection_metadata(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    history: NeuralTrainingHistory,
) -> dict[str, MetricValue]:
    best_validation = history.validation_rmse[history.best_epoch - 1]
    return {
        "best_epoch": history.best_epoch,
        "selected_epochs": history.best_epoch,
        "selection_epochs_ran": len(history.train_rmse),
        "stopped_early": history.stopped_early,
        "best_validation_rmse": best_validation,
        "train_rows": len(train),
        "validation_rows": len(validation),
    }


def _save_artifacts(
    output_dir: Path,
    model: PyTorchRecommender,
    metrics: dict[str, MetricValue],
    config: NeuralRecommenderConfig,
    history: NeuralTrainingHistory,
    metrics_filename: str = "metrics.json",
) -> TrainingArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = TrainingArtifacts(
        model=output_dir / "model.pt",
        metrics=output_dir / metrics_filename,
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_history(path: Path) -> NeuralTrainingHistory:
    payload = _read_json(path)
    payload["train_rmse"] = tuple(payload["train_rmse"])
    payload["validation_rmse"] = tuple(payload["validation_rmse"])
    return NeuralTrainingHistory(**payload)


def _existing_artifacts(output_dir: Path) -> TrainingArtifacts:
    return TrainingArtifacts(
        model=output_dir / "model.pt",
        metrics=output_dir / "metrics.json",
        config=output_dir / "config.json",
        selection_history=output_dir / "selection_history.json",
    )
