"""Scikit-Learn rating baselines with a recommendation interface."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURE_COLUMNS = ["user_id", "movie_id"]
RATING_MIN = 0.5
RATING_MAX = 5.0


class RatingEstimator(Protocol):
    """Minimal estimator interface required by the recommender."""

    def fit(self, features: pd.DataFrame, target: pd.Series) -> Any:
        """Fit the estimator."""
        ...

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict rating values."""
        ...


class SklearnRatingRecommender:
    """Adapt a Scikit-Learn regressor to rating and Top-K prediction."""

    def __init__(self, estimator: RatingEstimator) -> None:
        self.estimator = estimator
        self._catalog = np.array([], dtype=int)
        self._seen_items: dict[int, set[int]] = {}
        self._is_fitted = False

    def fit(self, interactions: pd.DataFrame) -> SklearnRatingRecommender:
        """Fit the estimator and cache catalog interaction state."""
        _validate_interactions(interactions)
        self.estimator.fit(interactions[FEATURE_COLUMNS], interactions["rating"])
        self._catalog = np.sort(interactions["movie_id"].unique()).astype(int)
        self._seen_items = _build_seen_items(interactions)
        self._is_fitted = True
        return self

    def predict_pairs(self, pairs: pd.DataFrame) -> np.ndarray:
        """Predict clipped ratings for user-movie pairs."""
        self._ensure_fitted()
        predictions = self.estimator.predict(pairs[FEATURE_COLUMNS])
        return np.clip(np.asarray(predictions, dtype=float), RATING_MIN, RATING_MAX)

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Return the highest-scored unseen fitted movies."""
        self._ensure_fitted()
        if k < 1:
            raise ValueError("k must be positive")
        candidates = self._candidate_frame(user_id)
        if candidates.empty:
            return []
        candidates["score"] = self.predict_pairs(candidates)
        ranked = candidates.sort_values(
            ["score", "movie_id"], ascending=[False, True], kind="stable"
        )
        return ranked.head(k)["movie_id"].astype(int).tolist()

    @property
    def catalog_size(self) -> int:
        """Return the fitted catalog size."""
        self._ensure_fitted()
        return len(self._catalog)

    def _candidate_frame(self, user_id: int) -> pd.DataFrame:
        seen = self._seen_items.get(user_id, set())
        movie_ids = [movie_id for movie_id in self._catalog if movie_id not in seen]
        return pd.DataFrame({"user_id": user_id, "movie_id": movie_ids})

    def _ensure_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("model is not fitted")


class SklearnMeanRecommender(SklearnRatingRecommender):
    """Predict the global training mean for every user-movie pair."""

    def __init__(self) -> None:
        super().__init__(DummyRegressor(strategy="mean"))


class SklearnBiasRecommender(SklearnRatingRecommender):
    """Learn additive user and movie weights with one-hot linear regression."""

    def __init__(
        self,
        *,
        alpha: float = 0.0001,
        max_iter: int = 2000,
        tolerance: float = 0.0001,
        random_state: int = 42,
    ) -> None:
        estimator = _build_bias_estimator(alpha, max_iter, tolerance, random_state)
        super().__init__(estimator)


def _build_bias_estimator(
    alpha: float, max_iter: int, tolerance: float, random_state: int
) -> Pipeline:
    encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    regressor = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=alpha,
        max_iter=max_iter,
        tol=tolerance,
        random_state=random_state,
        average=True,
    )
    return make_pipeline(encoder, regressor)


def _validate_interactions(interactions: pd.DataFrame) -> None:
    required = {*FEATURE_COLUMNS, "rating"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if interactions.empty:
        raise ValueError("interactions cannot be empty")


def _build_seen_items(interactions: pd.DataFrame) -> dict[int, set[int]]:
    return interactions.groupby("user_id")["movie_id"].agg(set).to_dict()
