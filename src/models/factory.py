"""Factory for recommendation models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from models.matrix_factorization import MatrixFactorizationRecommender
from models.popularity import PopularityRecommender


class ModelKind(StrEnum):
    """Available recommendation model identifiers."""

    POPULARITY = "popularity"
    MATRIX_FACTORIZATION = "matrix_factorization"


_MODEL_BUILDERS: dict[ModelKind, type[Any]] = {
    ModelKind.POPULARITY: PopularityRecommender,
    ModelKind.MATRIX_FACTORIZATION: MatrixFactorizationRecommender,
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
