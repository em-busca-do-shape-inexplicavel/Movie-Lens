"""Factory for preprocessing strategies."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from data.preprocessors.base import PreprocessorStrategy
from data.preprocessors.explicit import ExplicitFeedbackPreprocessor
from data.preprocessors.implicit import ImplicitFeedbackPreprocessor


class PreprocessorKind(StrEnum):
    """Available preprocessing strategy identifiers."""

    EXPLICIT = "explicit"
    IMPLICIT = "implicit"


_PREPROCESSORS: dict[PreprocessorKind, type[PreprocessorStrategy]] = {
    PreprocessorKind.EXPLICIT: ExplicitFeedbackPreprocessor,
    PreprocessorKind.IMPLICIT: ImplicitFeedbackPreprocessor,
}


def create_preprocessor(
    kind: str | PreprocessorKind, **kwargs: Any
) -> PreprocessorStrategy:
    """Create a preprocessing strategy.

    Args:
        kind: Registered strategy identifier.
        **kwargs: Arguments forwarded to the strategy constructor.

    Returns:
        Configured preprocessing strategy.

    Raises:
        ValueError: If the strategy identifier is unknown.
    """
    try:
        key = PreprocessorKind(kind)
    except ValueError as error:
        raise ValueError(f"unknown preprocessor: {kind}") from error
    return _PREPROCESSORS[key](**kwargs)
