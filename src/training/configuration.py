"""YAML configuration for the executable training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from models.pytorch_recommender import NeuralRecommenderConfig


@dataclass(frozen=True, slots=True)
class TrainingPipelineConfig:
    """Model training and recommendation evaluation settings."""

    model: NeuralRecommenderConfig
    top_k: int = 10


@dataclass(frozen=True, slots=True)
class ExperimentCandidateConfig:
    """Named hyperparameter configuration tracked as one MLflow run."""

    name: str
    model: NeuralRecommenderConfig


@dataclass(frozen=True, slots=True)
class ExperimentSelectionConfig:
    """Candidate configurations and validation selection rule."""

    candidates: tuple[ExperimentCandidateConfig, ...]
    selection_metric: str
    direction: str


def load_training_config(path: Path) -> TrainingPipelineConfig:
    """Load pipeline parameters from YAML.

    Args:
        path: YAML file containing training and evaluation sections.

    Returns:
        Validated pipeline configuration.
    """
    payload = _load_yaml(path)
    model_config = _build_model_config(payload.get("training", {}))
    top_k = int(payload.get("evaluation", {}).get("top_k", 10))
    if top_k < 1:
        raise ValueError("top_k must be positive")
    return TrainingPipelineConfig(model=model_config, top_k=top_k)


def load_experiment_config(path: Path) -> ExperimentSelectionConfig:
    """Load and validate at least three experiment candidates."""
    payload = _load_yaml(path)
    experiment = payload.get("experiments", {})
    candidates = _build_candidates(payload.get("training", {}), experiment)
    metric = str(experiment.get("selection_metric", "validation_ndcg@10"))
    direction = str(experiment.get("direction", "maximize"))
    if direction not in {"minimize", "maximize"}:
        raise ValueError("experiment direction must be minimize or maximize")
    return ExperimentSelectionConfig(candidates, metric, direction)


def _load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError("training configuration must be a mapping")
    return payload


def _build_model_config(parameters: dict[str, Any]) -> NeuralRecommenderConfig:
    values = dict(parameters)
    if "hidden_dims" in values:
        values["hidden_dims"] = tuple(values["hidden_dims"])
    return NeuralRecommenderConfig(**values)


def _build_candidates(
    base: dict[str, Any], experiment: dict[str, Any]
) -> tuple[ExperimentCandidateConfig, ...]:
    raw_candidates = experiment.get("candidates", [])
    candidates = tuple(_build_candidate(base, item) for item in raw_candidates)
    names = {candidate.name for candidate in candidates}
    if len(candidates) < 3:
        raise ValueError("at least three experiment candidates are required")
    if len(names) != len(candidates):
        raise ValueError("experiment candidate names must be unique")
    return candidates


def _build_candidate(
    base: dict[str, Any], item: dict[str, Any]
) -> ExperimentCandidateConfig:
    name = str(item.get("name", "")).strip()
    if not name:
        raise ValueError("every experiment candidate requires a name")
    parameters = dict(base)
    parameters.update(item.get("overrides", {}))
    return ExperimentCandidateConfig(name=name, model=_build_model_config(parameters))
