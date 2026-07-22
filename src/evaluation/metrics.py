"""Regression and ranking metrics for recommenders."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return root mean squared error."""
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("arrays must have equal, non-zero length")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Return the fraction of K recommendations that are relevant."""
    if k < 1:
        raise ValueError("k must be positive")
    hits = len(set(recommended[:k]) & relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Return the fraction of relevant items recovered in the first K."""
    if k < 1:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def evaluate_top_k(
    holdout: pd.DataFrame,
    recommend: Callable[[int, int], list[int]],
    catalog_size: int,
    *,
    k: int = 10,
    relevance_threshold: float = 4.0,
) -> tuple[pd.Series, pd.DataFrame, dict[int, list[int]]]:
    """Evaluate Top-K recommendations against relevant holdout items."""
    if catalog_size < 1:
        raise ValueError("catalog_size must be positive")

    recommendations = _recommend_for_users(holdout, recommend, k)
    relevant_by_user = _relevant_by_user(holdout, relevance_threshold)
    user_metrics = _build_user_metrics(recommendations, relevant_by_user, k)
    summary = _build_summary(holdout, recommendations, user_metrics, catalog_size, k)
    return summary, user_metrics, recommendations


def _recommend_for_users(
    holdout: pd.DataFrame,
    recommend: Callable[[int, int], list[int]],
    k: int,
) -> dict[int, list[int]]:
    return {
        int(user_id): recommend(int(user_id), k)
        for user_id in holdout["user_id"].unique()
    }


def _relevant_by_user(holdout: pd.DataFrame, threshold: float) -> dict[int, set[int]]:
    return (
        holdout.loc[holdout["rating"] >= threshold]
        .groupby("user_id")["movie_id"]
        .agg(set)
        .to_dict()
    )


def _build_user_metrics(
    recommendations: dict[int, list[int]],
    relevant_by_user: dict[int, set[int]],
    k: int,
) -> pd.DataFrame:
    rows = []
    for user_id, relevant in relevant_by_user.items():
        recommended = recommendations[int(user_id)]
        rows.append(
            {
                "user_id": int(user_id),
                f"precision@{k}": precision_at_k(recommended, relevant, k),
                f"recall@{k}": recall_at_k(recommended, relevant, k),
                "hit": bool(set(recommended[:k]) & relevant),
            }
        )

    return pd.DataFrame(rows)


def _build_summary(
    holdout: pd.DataFrame,
    recommendations: dict[int, list[int]],
    user_metrics: pd.DataFrame,
    catalog_size: int,
    k: int,
) -> pd.Series:
    unique_count = _unique_recommended_count(recommendations, k)
    return pd.Series(
        {
            "holdout_users": holdout["user_id"].nunique(),
            "evaluable_users": len(user_metrics),
            f"precision@{k}": user_metrics[f"precision@{k}"].mean(),
            f"recall@{k}": user_metrics[f"recall@{k}"].mean(),
            f"hit_rate@{k}": user_metrics["hit"].mean(),
            f"catalog_coverage@{k}": unique_count / catalog_size,
        },
        name="value",
    )


def _unique_recommended_count(recommendations: dict[int, list[int]], k: int) -> int:
    return len({item for items in recommendations.values() for item in items[:k]})
