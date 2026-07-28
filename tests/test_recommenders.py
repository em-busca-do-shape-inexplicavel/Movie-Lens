from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from models.matrix_factorization import MatrixFactorizationRecommender
from models.popularity import PopularityRecommender


def sample_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "movie_id": [10, 11, 12, 10, 12, 13, 11, 13, 14],
            "rating": [5.0, 4.0, 1.0, 4.5, 2.0, 5.0, 4.5, 4.0, 1.5],
        }
    )


class PopularityRecommenderTest(unittest.TestCase):
    def test_recommends_ranked_unseen_items(self) -> None:
        model = PopularityRecommender().fit(sample_interactions())

        recommendations = model.recommend(user_id=1, k=2)

        self.assertEqual(recommendations, [13, 14])
        self.assertFalse(set(recommendations) & {10, 11, 12})

    def test_requires_fit(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            PopularityRecommender().recommend(user_id=1)


class MatrixFactorizationRecommenderTest(unittest.TestCase):
    def test_predicts_finite_ratings_and_filters_seen_items(self) -> None:
        interactions = sample_interactions()
        model = MatrixFactorizationRecommender(
            n_factors=4,
            learning_rate=0.02,
            regularization=0.01,
            epochs=15,
            random_state=7,
        ).fit(interactions)

        predictions = model.predict_pairs(interactions[["user_id", "movie_id"]])
        recommendations = model.recommend(user_id=1, k=2)

        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue(((predictions >= 0.5) & (predictions <= 5.0)).all())
        self.assertFalse(set(recommendations) & {10, 11, 12})
        self.assertIsNotNone(model.history)
        assert model.history is not None
        self.assertLess(model.history.train_rmse[-1], model.history.train_rmse[0])

    def test_is_reproducible_with_fixed_seed(self) -> None:
        interactions = sample_interactions()
        params = {
            "n_factors": 3,
            "epochs": 3,
            "random_state": 42,
        }
        first = MatrixFactorizationRecommender(**params).fit(interactions)
        second = MatrixFactorizationRecommender(**params).fit(interactions)

        first_predictions = first.predict_pairs(
            interactions[["user_id", "movie_id"]]
        )
        second_predictions = second.predict_pairs(
            interactions[["user_id", "movie_id"]]
        )

        np.testing.assert_allclose(first_predictions, second_predictions)


if __name__ == "__main__":
    unittest.main()
