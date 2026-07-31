from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from models.pytorch_recommender import NeuralRecommenderConfig, PyTorchRecommender
from training.configuration import TrainingPipelineConfig, load_training_config
from training.pipeline import run_training_pipeline


def test_load_training_config(tmp_path: Path) -> None:
    path = tmp_path / "params.yaml"
    path.write_text(
        "training:\n  embedding_dim: 8\n  hidden_dims: [16, 8]\n"
        "evaluation:\n  top_k: 5\n",
        encoding="utf-8",
    )

    config = load_training_config(path)

    assert config.model.embedding_dim == 8
    assert config.model.hidden_dims == (16, 8)
    assert config.top_k == 5


def test_pipeline_persists_loadable_artifacts(tmp_path: Path) -> None:
    raw_data_dir = tmp_path / "raw"
    raw_data_dir.mkdir()
    _sample_ratings().to_csv(raw_data_dir / "ratings.csv", index=False)
    config = TrainingPipelineConfig(model=_test_model_config(), top_k=2)

    result = run_training_pipeline(raw_data_dir, tmp_path / "artifacts", config)

    assert all(path.exists() for path in _artifact_paths(result.artifacts))
    assert result.metrics["selected_epochs"] == 1
    assert result.metrics["stopped_early"]
    assert result.metrics["best_epoch"] == 1
    assert result.metrics["training_seconds"] >= 0
    persisted_metrics = json.loads(result.artifacts.metrics.read_text())
    assert persisted_metrics == result.metrics
    assert PyTorchRecommender.load(result.artifacts.model).catalog_size > 0


def _sample_ratings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "movieId": [10, 11, 12, 13, 10, 12, 14, 15, 11, 13, 14, 16],
            "rating": [5, 4, 5, 5, 4, 3, 4, 5, 5, 4, 4, 5],
            "timestamp": list(range(1, 13)),
        }
    )


def _test_model_config() -> NeuralRecommenderConfig:
    return NeuralRecommenderConfig(
        embedding_dim=4,
        hidden_dims=(8,),
        ranking_weight=0.0,
        batch_size=3,
        epochs=4,
        early_stopping_patience=1,
        early_stopping_min_delta=100.0,
    )


def _artifact_paths(artifacts) -> tuple[Path, ...]:
    return (
        artifacts.model,
        artifacts.metrics,
        artifacts.config,
        artifacts.selection_history,
    )
