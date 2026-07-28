from __future__ import annotations

import unittest

import pandas as pd

from data.splitting import temporal_leave_two_out


class TemporalSplitTest(unittest.TestCase):
    def test_reserves_latest_interactions_per_user(self) -> None:
        interactions = pd.DataFrame(
            {
                "user_id": [1, 1, 1, 1, 2, 2, 2, 2],
                "movie_id": [10, 11, 12, 13, 20, 21, 22, 23],
                "rating": [3.0, 4.0, 5.0, 4.5, 2.0, 3.0, 4.0, 5.0],
                "timestamp": [1, 2, 3, 4, 10, 20, 30, 40],
            }
        )

        train, validation, test = temporal_leave_two_out(interactions)

        self.assertEqual(set(train["movie_id"]), {10, 11, 20, 21})
        self.assertEqual(set(validation["movie_id"]), {12, 22})
        self.assertEqual(set(test["movie_id"]), {13, 23})

    def test_rejects_short_user_history(self) -> None:
        interactions = pd.DataFrame(
            {
                "user_id": [1, 1],
                "movie_id": [10, 11],
                "timestamp": [1, 2],
            }
        )

        with self.assertRaisesRegex(ValueError, "at least three"):
            temporal_leave_two_out(interactions)


if __name__ == "__main__":
    unittest.main()
