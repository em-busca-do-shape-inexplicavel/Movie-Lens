"""MLflow logging and model registry helpers."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types import ColSpec, Schema

from models.pytorch_recommender import PyTorchRecommender
from training.configuration import TrainingPipelineConfig
from training.pipeline import TrainingPipelineResult

DEFAULT_MODEL_NAME = "movie-lens-pytorch-recommender"
DEFAULT_MODEL_STAGE = "Production"
DEFAULT_MODEL_ALIAS = "production"
DEFAULT_MODEL_DESCRIPTION = (
    "MovieLens PyTorch recommender trained with temporal leave-two-out splitting."
)


@dataclass(frozen=True, slots=True)
class MlflowTrackingResult:
    """Result returned after logging and registering a run."""

    run_id: str
    model_name: str
    registered_model_version: str | None = None


class _MovieLensPyfuncModel(mlflow.pyfunc.PythonModel):
    """Load a fitted recommender checkpoint inside the registry artifact."""

    def load_context(self, context: Any) -> None:
        checkpoint = Path(context.artifacts["checkpoint"])
        self._model = PyTorchRecommender.load(checkpoint)

    def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
        frame = (
            model_input
            if isinstance(model_input, pd.DataFrame)
            else pd.DataFrame(model_input)
        )
        predictions = self._model.predict_pairs(frame[["user_id", "movie_id"]])
        return pd.DataFrame({"prediction": predictions})


def track_training_run(
    *,
    config: TrainingPipelineConfig,
    result: TrainingPipelineResult,
    params_path: Path,
    tracking_uri: str,
    experiment_name: str,
    model_name: str = DEFAULT_MODEL_NAME,
    model_stage: str = DEFAULT_MODEL_STAGE,
    model_alias: str = DEFAULT_MODEL_ALIAS,
    description: str = DEFAULT_MODEL_DESCRIPTION,
) -> MlflowTrackingResult:
    """Log metrics, artifacts, and register the trained model version."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    tags = _build_tags(result.metrics)
    with mlflow.start_run(run_name=f"{model_name}-training") as run:
        _log_parameters(config)
        _log_metrics(result.metrics)
        mlflow.log_artifact(str(params_path))
        mlflow.log_artifacts(str(result.artifacts.model.parent))
        input_example = _model_input_example()
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=_MovieLensPyfuncModel(),
            artifacts={"checkpoint": str(result.artifacts.model)},
            input_example=input_example,
            signature=_model_signature(),
        )
        mlflow.set_tags(tags)
    publication = register_model_version(
        run_id=run.info.run_id,
        model_name=model_name,
        description=description,
        stage=model_stage,
        alias=model_alias,
        tags=tags,
        tracking_uri=tracking_uri,
    )
    return MlflowTrackingResult(
        run_id=run.info.run_id,
        model_name=model_name,
        registered_model_version=str(publication.version),
    )


def register_model_version(
    *,
    run_id: str,
    model_name: str = DEFAULT_MODEL_NAME,
    description: str = DEFAULT_MODEL_DESCRIPTION,
    stage: str = DEFAULT_MODEL_STAGE,
    alias: str = DEFAULT_MODEL_ALIAS,
    tags: Mapping[str, str] | None = None,
    tracking_uri: str | None = None,
) -> mlflow.entities.model_registry.model_version.ModelVersion:
    """Register the logged pyfunc model and promote the created version."""
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    registered = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model", name=model_name, tags=dict(tags or {})
    )
    _update_registered_model(client, model_name, description, tags)
    _promote_version(client, model_name, str(registered.version), stage, alias)
    return registered


def _model_input_example() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": pd.Series([1], dtype="int64"),
            "movie_id": pd.Series([1], dtype="int64"),
        }
    )


def _model_signature() -> ModelSignature:
    return ModelSignature(
        inputs=Schema(
            [
                ColSpec("long", "user_id"),
                ColSpec("long", "movie_id"),
            ]
        ),
        outputs=Schema([ColSpec("double", "prediction")]),
    )


def _log_parameters(config: TrainingPipelineConfig) -> None:
    model = config.model
    mlflow.log_params(
        {
            "model": "pytorch",
            "seed": model.random_state,
            "embedding_dim": model.embedding_dim,
            "hidden_dims": ",".join(str(dimension) for dimension in model.hidden_dims),
            "learning_rate": model.learning_rate,
            "weight_decay": model.weight_decay,
            "ranking_weight": model.ranking_weight,
            "patience": model.early_stopping_patience,
            "min_delta": model.early_stopping_min_delta,
            "relevance_threshold": model.relevance_threshold,
            "batch_size": model.batch_size,
            "epochs": model.epochs,
            "device": model.device,
            "top_k": config.top_k,
        }
    )


def _log_metrics(metrics: Mapping[str, Any]) -> None:
    for key, value in metrics.items():
        metric_name = _sanitize_metric_name(key)
        if isinstance(value, bool):
            mlflow.set_tag(metric_name, str(value).lower())
            continue
        if isinstance(value, (int, float)):
            mlflow.log_metric(metric_name, float(value))
            continue
        mlflow.set_tag(metric_name, str(value))


def _sanitize_metric_name(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_./ -]", "_", name)
    return sanitized.replace(" ", "_")


def _build_tags(metrics: Mapping[str, Any]) -> dict[str, str]:
    tags = {
        "project": "movie-lens",
        "framework": "pytorch",
        "tracking": "mlflow",
    }
    best_epoch = metrics.get("best_epoch") or metrics.get("selected_epochs")
    if best_epoch is not None:
        tags["best_epoch"] = str(best_epoch)
    if "selection_epochs_ran" in metrics:
        tags["selection_epochs_ran"] = str(metrics["selection_epochs_ran"])
    if "stopped_early" in metrics:
        tags["stopped_early"] = str(metrics["stopped_early"]).lower()
    git_commit = _git_commit_hash()
    if git_commit is not None:
        tags["git_commit"] = git_commit
    return tags


def _git_commit_hash() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def _update_registered_model(
    client: MlflowClient,
    model_name: str,
    description: str,
    tags: Mapping[str, str] | None,
) -> None:
    client.update_registered_model(model_name, description=description)
    for key, value in (tags or {}).items():
        client.set_registered_model_tag(model_name, key, value)


def _promote_version(
    client: MlflowClient,
    model_name: str,
    version: str,
    stage: str,
    alias: str,
) -> None:
    client.transition_model_version_stage(model_name, version, stage=stage)
    client.set_registered_model_alias(model_name, alias, version)

    promoted = client.get_model_version(model_name, version)
    if promoted.current_stage != stage:
        raise mlflow.exceptions.MlflowException(
            f"Model version {model_name} v{version} was not promoted to {stage}."
        )
    aliased = client.get_model_version_by_alias(model_name, alias)
    if str(aliased.version) != str(version):
        raise mlflow.exceptions.MlflowException(
            f"Alias {alias!r} does not point to {model_name} v{version}."
        )

    client.set_model_version_tag(model_name, version, "stage", stage)
    client.set_model_version_tag(model_name, version, "alias", alias)
