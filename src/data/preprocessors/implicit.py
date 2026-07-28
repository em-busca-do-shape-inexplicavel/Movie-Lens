"""Implicit-feedback preprocessing."""

from __future__ import annotations

import pandas as pd

from data.preprocessors.base import PreprocessorStrategy


class ImplicitFeedbackPreprocessor(PreprocessorStrategy):
    """Convert sufficiently high ratings into positive interactions."""

    def __init__(self, relevance_threshold: float = 4.0) -> None:
        self.relevance_threshold = relevance_threshold

    def transform(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Keep relevant ratings and create a binary interaction target."""
        transformed = ratings.loc[ratings["rating"] >= self.relevance_threshold].copy()
        transformed["interaction"] = 1
        transformed["feedback"] = "implicit"
        return transformed.reset_index(drop=True)
