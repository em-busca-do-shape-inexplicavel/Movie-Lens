from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.pytorch_recommender import NeuralRecommenderConfig, PyTorchRecommender


@pytest.fixture
def interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "movie_id": [10, 11, 12, 10, 12, 13, 11, 13, 14],
            "rating": [5.0, 4.0, 1.0, 4.5, 2.0, 5.0, 4.5, 4.0, 1.5],
        }
    )


def build_model(seed: int = 7) -> PyTorchRecommender:
    config = NeuralRecommenderConfig(
        embedding_dim=4,
        hidden_dims=(8,),
        learning_rate=0.01,
        batch_size=3,
        epochs=4,
        random_state=seed,
    )
    return PyTorchRecommender(config)


def test_predicts_valid_ratings_and_filters_seen(
    interactions: pd.DataFrame,
) -> None:
    model = build_model().fit(interactions)

    predictions = model.predict_pairs(interactions[["user_id", "movie_id"]])
    recommendations = model.recommend(user_id=1, k=2)

    assert np.isfinite(predictions).all()
    assert ((predictions >= 0.5) & (predictions <= 5.0)).all()
    assert not set(recommendations) & {10, 11, 12}
    assert model.catalog_size == 5
    assert model.history is not None
    assert len(model.history.train_rmse) == 4
    assert model.history.best_epoch == 4


def test_is_reproducible_with_fixed_seed(interactions: pd.DataFrame) -> None:
    pairs = interactions[["user_id", "movie_id"]]

    first = build_model(seed=42).fit(interactions).predict_pairs(pairs)
    second = build_model(seed=42).fit(interactions).predict_pairs(pairs)

    np.testing.assert_allclose(first, second)


def test_unknown_ids_fall_back_to_training_mean(interactions: pd.DataFrame) -> None:
    model = build_model().fit(interactions)
    pairs = pd.DataFrame({"user_id": [999], "movie_id": [999]})

    prediction = model.predict_pairs(pairs)

    np.testing.assert_allclose(prediction, interactions["rating"].mean())


def test_requires_fit_and_valid_configuration() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        build_model().recommend(user_id=1)
    with pytest.raises(ValueError, match="positive"):
        PyTorchRecommender(NeuralRecommenderConfig(embedding_dim=0))


def test_early_stopping_restores_best_epoch(interactions: pd.DataFrame) -> None:
    config = NeuralRecommenderConfig(
        embedding_dim=4,
        hidden_dims=(8,),
        batch_size=3,
        epochs=8,
        early_stopping_patience=1,
        early_stopping_min_delta=100.0,
    )

    model = PyTorchRecommender(config).fit(interactions, interactions)

    assert model.history is not None
    assert model.history.best_epoch == 1
    assert model.history.stopped_early
    assert len(model.history.train_rmse) == 2


def test_checkpoint_round_trip(interactions: pd.DataFrame, tmp_path: Path) -> None:
    pairs = interactions[["user_id", "movie_id"]]
    model = build_model().fit(interactions)
    checkpoint_path = model.save(tmp_path / "model.pt")

    restored = PyTorchRecommender.load(checkpoint_path)

    np.testing.assert_allclose(
        restored.predict_pairs(pairs), model.predict_pairs(pairs)
    )
    assert restored.recommend(user_id=1, k=2) == model.recommend(user_id=1, k=2)
