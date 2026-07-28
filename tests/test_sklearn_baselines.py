from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.sklearn_baselines import (
    SklearnBiasRecommender,
    SklearnMeanRecommender,
)


@pytest.fixture
def interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "movie_id": [10, 11, 10, 12, 11, 13],
            "rating": [5.0, 4.0, 4.5, 2.0, 3.5, 1.0],
        }
    )


def test_mean_model_predicts_training_average(interactions: pd.DataFrame) -> None:
    model = SklearnMeanRecommender().fit(interactions)

    predictions = model.predict_pairs(interactions[["user_id", "movie_id"]])

    np.testing.assert_allclose(predictions, interactions["rating"].mean())


def test_bias_model_predicts_valid_ratings_and_filters_seen(
    interactions: pd.DataFrame,
) -> None:
    model = SklearnBiasRecommender(max_iter=1000).fit(interactions)

    predictions = model.predict_pairs(interactions[["user_id", "movie_id"]])
    recommendations = model.recommend(user_id=1, k=3)

    assert np.isfinite(predictions).all()
    assert ((predictions >= 0.5) & (predictions <= 5.0)).all()
    assert not set(recommendations) & {10, 11}
    assert model.catalog_size == 4


def test_bias_model_accepts_unknown_ids(interactions: pd.DataFrame) -> None:
    model = SklearnBiasRecommender(max_iter=1000).fit(interactions)
    unknown_pairs = pd.DataFrame({"user_id": [999], "movie_id": [999]})

    prediction = model.predict_pairs(unknown_pairs)

    assert np.isfinite(prediction).all()


def test_recommend_returns_empty_when_user_has_seen_catalog(
    interactions: pd.DataFrame,
) -> None:
    model = SklearnMeanRecommender().fit(interactions)
    all_seen = pd.concat(
        [
            interactions,
            pd.DataFrame(
                {"user_id": [1, 1], "movie_id": [12, 13], "rating": [3.0, 3.0]}
            ),
        ],
        ignore_index=True,
    )

    model.fit(all_seen)

    assert model.recommend(user_id=1, k=10) == []


def test_model_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        SklearnMeanRecommender().recommend(user_id=1)
