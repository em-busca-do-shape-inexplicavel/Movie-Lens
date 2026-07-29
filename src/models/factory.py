"""Factory for recommendation models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from models.matrix_factorization import MatrixFactorizationRecommender
from models.popularity import PopularityRecommender
from models.pytorch_recommender import PyTorchRecommender
from models.sklearn_baselines import (
    SklearnBiasRecommender,
    SklearnMeanRecommender,
)


class ModelKind(StrEnum):
    """Available recommendation model identifiers."""

    POPULARITY = "popularity"
    MATRIX_FACTORIZATION = "matrix_factorization"
    SKLEARN_MEAN = "sklearn_mean"
    SKLEARN_BIAS = "sklearn_bias"
    PYTORCH = "pytorch"


_MODEL_BUILDERS: dict[ModelKind, type[Any]] = {
    ModelKind.POPULARITY: PopularityRecommender,
    ModelKind.MATRIX_FACTORIZATION: MatrixFactorizationRecommender,
    ModelKind.SKLEARN_MEAN: SklearnMeanRecommender,
    ModelKind.SKLEARN_BIAS: SklearnBiasRecommender,
    ModelKind.PYTORCH: PyTorchRecommender,
}


def create_model(kind: str | ModelKind, **kwargs: Any) -> Any:
    """Create a registered recommendation model.

    Args:
        kind: Registered model identifier.
        **kwargs: Arguments forwarded to the model constructor.

    Returns:
        A new recommendation model.

    Raises:
        ValueError: If the model identifier is unknown.
    """
    try:
        key = ModelKind(kind)
    except ValueError as error:
        raise ValueError(f"unknown model: {kind}") from error
    return _MODEL_BUILDERS[key](**kwargs)
