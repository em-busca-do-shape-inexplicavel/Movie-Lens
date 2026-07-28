"""Leakage-safe train, validation, and test splits."""

from __future__ import annotations

import pandas as pd


def temporal_leave_two_out(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reserve the two latest interactions of every user.

    The last interaction becomes test data, the penultimate interaction
    becomes validation data, and all previous interactions become training
    data. Users must have at least three interactions.
    """
    required = {"user_id", "movie_id", "timestamp"}
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    history_sizes = interactions.groupby("user_id").size()
    short_histories = history_sizes[history_sizes < 3]
    if not short_histories.empty:
        raise ValueError("every user must have at least three interactions")

    ordered = interactions.assign(
        rated_at=pd.to_datetime(interactions["timestamp"], unit="s", utc=True)
    ).sort_values(["user_id", "rated_at", "movie_id"], kind="stable")

    ordered["interaction_order"] = ordered.groupby("user_id").cumcount()
    ordered["history_size"] = ordered.groupby("user_id")["movie_id"].transform(
        "size"
    )
    ordered["position_from_end"] = (
        ordered["history_size"] - ordered["interaction_order"]
    )

    train = ordered.loc[ordered["position_from_end"] > 2].copy()
    validation = ordered.loc[ordered["position_from_end"] == 2].copy()
    test = ordered.loc[ordered["position_from_end"] == 1].copy()
    return train, validation, test
