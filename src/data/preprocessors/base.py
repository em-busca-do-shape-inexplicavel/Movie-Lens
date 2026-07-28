"""Preprocessing strategy contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class PreprocessorStrategy(ABC):
    """Transform ratings according to a feedback definition."""

    @abstractmethod
    def transform(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Transform ratings without mutating the input.

        Args:
            ratings: Raw explicit MovieLens ratings.

        Returns:
            A transformed copy of the interactions.
        """
