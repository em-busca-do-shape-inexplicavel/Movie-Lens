from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from evaluation.metrics import evaluate_top_k, precision_at_k, recall_at_k, rmse


class MetricsTest(unittest.TestCase):
    def test_ranking_metrics(self) -> None:
        recommended = [10, 20, 30, 40]
        relevant = {20, 40}

        self.assertEqual(precision_at_k(recommended, relevant, 4), 0.5)
        self.assertEqual(recall_at_k(recommended, relevant, 4), 1.0)

    def test_rmse(self) -> None:
        actual = np.array([3.0, 4.0])
        predicted = np.array([2.0, 5.0])

        self.assertEqual(rmse(actual, predicted), 1.0)

    def test_top_k_summary(self) -> None:
        holdout = pd.DataFrame(
            {
                "user_id": [1, 2],
                "movie_id": [20, 99],
                "rating": [5.0, 3.0],
            }
        )

        def recommend(user_id: int, k: int) -> list[int]:
            del user_id
            return [10, 20, 30][:k]

        summary, user_metrics, recommendations = evaluate_top_k(
            holdout, recommend, catalog_size=10, k=3
        )

        self.assertEqual(summary["evaluable_users"], 1)
        self.assertAlmostEqual(summary["precision@3"], 1 / 3)
        self.assertEqual(summary["recall@3"], 1.0)
        self.assertEqual(len(user_metrics), 1)
        self.assertEqual(set(recommendations), {1, 2})


if __name__ == "__main__":
    unittest.main()
