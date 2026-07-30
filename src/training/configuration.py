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
