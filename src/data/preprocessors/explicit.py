"""Explicit-feedback preprocessing."""

from __future__ import annotations

import pandas as pd

from data.preprocessors.base import PreprocessorStrategy


class ExplicitFeedbackPreprocessor(PreprocessorStrategy):
    """Preserve ratings at or above a configurable minimum."""

    def __init__(self, minimum_rating: float = 0.5) -> None:
        self.minimum_rating = minimum_rating

    def transform(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Filter explicit ratings and label their feedback type."""
        transformed = ratings.loc[ratings["rating"] >= self.minimum_rating].copy()
        transformed["feedback"] = "explicit"
        return transformed.reset_index(drop=True)
