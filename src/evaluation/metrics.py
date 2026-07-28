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

    user_ids = holdout["user_id"].unique()
    recommendations = {
        int(user_id): recommend(int(user_id), k) for user_id in user_ids
    }
    relevant_by_user = (
        holdout.loc[holdout["rating"] >= relevance_threshold]
        .groupby("user_id")["movie_id"]
        .agg(set)
        .to_dict()
    )

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

    user_metrics = pd.DataFrame(rows)
    unique_recommended = {
        item for items in recommendations.values() for item in items[:k]
    }
    summary = pd.Series(
        {
            "holdout_users": len(user_ids),
            "evaluable_users": len(relevant_by_user),
            f"precision@{k}": user_metrics[f"precision@{k}"].mean(),
            f"recall@{k}": user_metrics[f"recall@{k}"].mean(),
            f"hit_rate@{k}": user_metrics["hit"].mean(),
            f"catalog_coverage@{k}": len(unique_recommended) / catalog_size,
        },
        name="value",
    )
    return summary, user_metrics, recommendations
