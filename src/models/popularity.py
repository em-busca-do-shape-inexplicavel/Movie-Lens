"""Non-personalized popularity baseline."""

from __future__ import annotations

import pandas as pd


class PopularityRecommender:
    """Rank globally popular movies and filter each user's seen items."""

    def __init__(self) -> None:
        self._ranking: pd.DataFrame | None = None
        self._seen_items: dict[int, set[int]] = {}

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        """Build the popularity ranking from historical interactions."""
        required = {"user_id", "movie_id", "rating"}
        missing = required - set(interactions.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        self._ranking = (
            interactions.groupby("movie_id")
            .agg(
                interaction_count=("rating", "size"),
                rating_mean=("rating", "mean"),
            )
            .reset_index()
            .sort_values(
                ["interaction_count", "rating_mean", "movie_id"],
                ascending=[False, False, True],
                kind="stable",
            )
            .reset_index(drop=True)
        )
        self._seen_items = (
            interactions.groupby("user_id")["movie_id"].agg(set).to_dict()
        )
        return self

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Return the globally popular unseen movies for a user."""
        ranking = self.ranking
        seen = self._seen_items.get(user_id, set())
        candidates = ranking.loc[
            ~ranking["movie_id"].isin(seen), "movie_id"
        ]
        return candidates.head(k).astype(int).tolist()

    @property
    def ranking(self) -> pd.DataFrame:
        """Return the fitted ranking."""
        if self._ranking is None:
            raise RuntimeError("model is not fitted")
        return self._ranking.copy()

    @property
    def catalog_size(self) -> int:
        """Return the number of ranked movies."""
        return len(self.ranking)
