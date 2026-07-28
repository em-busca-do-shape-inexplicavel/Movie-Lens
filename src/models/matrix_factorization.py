"""Explicit-feedback matrix factorization trained with SGD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    """Per-epoch training losses."""

    train_rmse: tuple[float, ...]


class MatrixFactorizationRecommender:
    """Learn user and movie latent factors from explicit ratings."""

    def __init__(
        self,
        n_factors: int = 20,
        learning_rate: float = 0.01,
        regularization: float = 0.05,
        epochs: int = 10,
        random_state: int = 42,
    ) -> None:
        if n_factors < 1 or epochs < 1:
            raise ValueError("n_factors and epochs must be positive")
        if learning_rate <= 0 or regularization < 0:
            raise ValueError("invalid optimization parameters")

        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.random_state = random_state

        self._user_to_idx: dict[int, int] = {}
        self._movie_to_idx: dict[int, int] = {}
        self._idx_to_movie: np.ndarray | None = None
        self._seen_items: dict[int, set[int]] = {}
        self._global_mean = 0.0
        self._user_bias: np.ndarray | None = None
        self._movie_bias: np.ndarray | None = None
        self._user_factors: np.ndarray | None = None
        self._movie_factors: np.ndarray | None = None
        self.history: TrainingHistory | None = None

    def fit(self, interactions: pd.DataFrame) -> "MatrixFactorizationRecommender":
        """Fit biases and latent factors with stochastic gradient descent."""
        self._validate_interactions(interactions)
        users = np.sort(interactions["user_id"].unique()).astype(int)
        movies = np.sort(interactions["movie_id"].unique()).astype(int)
        self._user_to_idx = {user_id: idx for idx, user_id in enumerate(users)}
        self._movie_to_idx = {
            movie_id: idx for idx, movie_id in enumerate(movies)
        }
        self._idx_to_movie = movies
        self._seen_items = (
            interactions.groupby("user_id")["movie_id"].agg(set).to_dict()
        )
        self._global_mean = float(interactions["rating"].mean())

        user_idx = interactions["user_id"].map(self._user_to_idx).to_numpy(int)
        movie_idx = interactions["movie_id"].map(self._movie_to_idx).to_numpy(int)
        ratings = interactions["rating"].to_numpy(float)
        self._initialize_parameters(len(users), len(movies))

        rng = np.random.default_rng(self.random_state)
        epoch_rmse: list[float] = []
        for _ in range(self.epochs):
            order = rng.permutation(len(ratings))
            self._run_epoch(user_idx, movie_idx, ratings, order)
            predictions = self._predict_known_indices(user_idx, movie_idx)
            epoch_rmse.append(float(np.sqrt(np.mean((ratings - predictions) ** 2))))

        self.history = TrainingHistory(train_rmse=tuple(epoch_rmse))
        return self

    def predict_pairs(self, pairs: pd.DataFrame) -> np.ndarray:
        """Predict ratings for user-movie pairs, including cold-start IDs."""
        self._ensure_fitted()
        predictions = np.full(len(pairs), self._global_mean, dtype=float)
        for row_idx, (user_id, movie_id) in enumerate(
            pairs[["user_id", "movie_id"]].itertuples(index=False, name=None)
        ):
            user_idx = self._user_to_idx.get(int(user_id))
            movie_idx = self._movie_to_idx.get(int(movie_id))
            predictions[row_idx] = self._predict_one(user_idx, movie_idx)
        return np.clip(predictions, 0.5, 5.0)

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Rank unseen fitted movies for a user."""
        self._ensure_fitted()
        if k < 1:
            raise ValueError("k must be positive")

        assert self._idx_to_movie is not None
        assert self._movie_bias is not None
        assert self._movie_factors is not None

        user_idx = self._user_to_idx.get(user_id)
        scores = self._global_mean + self._movie_bias.copy()
        if user_idx is not None:
            assert self._user_bias is not None
            assert self._user_factors is not None
            scores += self._user_bias[user_idx]
            scores += self._movie_factors @ self._user_factors[user_idx]

        seen = self._seen_items.get(user_id, set())
        candidate_mask = np.array(
            [movie_id not in seen for movie_id in self._idx_to_movie], dtype=bool
        )
        candidate_indices = np.flatnonzero(candidate_mask)
        candidate_scores = scores[candidate_indices]
        order = np.argsort(-candidate_scores, kind="stable")[:k]
        return self._idx_to_movie[candidate_indices[order]].astype(int).tolist()

    @property
    def catalog_size(self) -> int:
        """Return the number of movies learned during fit."""
        self._ensure_fitted()
        assert self._idx_to_movie is not None
        return len(self._idx_to_movie)

    def _initialize_parameters(self, n_users: int, n_movies: int) -> None:
        rng = np.random.default_rng(self.random_state)
        self._user_bias = np.zeros(n_users)
        self._movie_bias = np.zeros(n_movies)
        self._user_factors = rng.normal(0.0, 0.1, (n_users, self.n_factors))
        self._movie_factors = rng.normal(0.0, 0.1, (n_movies, self.n_factors))

    def _run_epoch(
        self,
        user_idx: np.ndarray,
        movie_idx: np.ndarray,
        ratings: np.ndarray,
        order: np.ndarray,
    ) -> None:
        assert self._user_bias is not None
        assert self._movie_bias is not None
        assert self._user_factors is not None
        assert self._movie_factors is not None

        lr = self.learning_rate
        reg = self.regularization
        for sample_idx in order:
            u = user_idx[sample_idx]
            i = movie_idx[sample_idx]
            user_vector = self._user_factors[u].copy()
            movie_vector = self._movie_factors[i].copy()
            prediction = (
                self._global_mean
                + self._user_bias[u]
                + self._movie_bias[i]
                + np.dot(user_vector, movie_vector)
            )
            error = ratings[sample_idx] - prediction

            self._user_bias[u] += lr * (error - reg * self._user_bias[u])
            self._movie_bias[i] += lr * (error - reg * self._movie_bias[i])
            self._user_factors[u] += lr * (error * movie_vector - reg * user_vector)
            self._movie_factors[i] += lr * (error * user_vector - reg * movie_vector)

    def _predict_known_indices(
        self, user_idx: np.ndarray, movie_idx: np.ndarray
    ) -> np.ndarray:
        assert self._user_bias is not None
        assert self._movie_bias is not None
        assert self._user_factors is not None
        assert self._movie_factors is not None
        return (
            self._global_mean
            + self._user_bias[user_idx]
            + self._movie_bias[movie_idx]
            + np.sum(
                self._user_factors[user_idx] * self._movie_factors[movie_idx], axis=1
            )
        )

    def _predict_one(self, user_idx: int | None, movie_idx: int | None) -> float:
        prediction = self._global_mean
        if user_idx is not None:
            assert self._user_bias is not None
            prediction += self._user_bias[user_idx]
        if movie_idx is not None:
            assert self._movie_bias is not None
            prediction += self._movie_bias[movie_idx]
        if user_idx is not None and movie_idx is not None:
            assert self._user_factors is not None
            assert self._movie_factors is not None
            prediction += float(
                np.dot(self._user_factors[user_idx], self._movie_factors[movie_idx])
            )
        return prediction

    def _ensure_fitted(self) -> None:
        if self._user_factors is None or self._movie_factors is None:
            raise RuntimeError("model is not fitted")

    @staticmethod
    def _validate_interactions(interactions: pd.DataFrame) -> None:
        required = {"user_id", "movie_id", "rating"}
        missing = required - set(interactions.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        if interactions.empty:
            raise ValueError("interactions cannot be empty")
