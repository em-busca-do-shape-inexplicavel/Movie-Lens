"""Explicit-feedback matrix factorization trained with SGD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    """Per-epoch training losses."""

    train_rmse: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class MatrixFactorizationConfig:
    """Optimization and model hyperparameters."""

    n_factors: int = 20
    learning_rate: float = 0.01
    regularization: float = 0.05
    epochs: int = 10
    random_state: int = 42


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
        self.config = MatrixFactorizationConfig(
            n_factors, learning_rate, regularization, epochs, random_state
        )
        self._validate_config()
        self._reset_state()

    def _reset_state(self) -> None:
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

    def fit(self, interactions: pd.DataFrame) -> MatrixFactorizationRecommender:
        """Fit biases and latent factors with stochastic gradient descent."""
        self._validate_interactions(interactions)
        user_idx, movie_idx, ratings = self._prepare_fit(interactions)
        self._initialize_parameters(len(self._user_to_idx), len(self._movie_to_idx))
        epoch_rmse = self._train(user_idx, movie_idx, ratings)
        self.history = TrainingHistory(train_rmse=tuple(epoch_rmse))
        return self

    def _prepare_fit(
        self, interactions: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        users = np.sort(interactions["user_id"].unique()).astype(int)
        movies = np.sort(interactions["movie_id"].unique()).astype(int)
        self._user_to_idx = {user_id: idx for idx, user_id in enumerate(users)}
        self._movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies)}
        self._idx_to_movie = movies
        self._seen_items = (
            interactions.groupby("user_id")["movie_id"].agg(set).to_dict()
        )
        self._global_mean = float(interactions["rating"].mean())
        user_idx = interactions["user_id"].map(self._user_to_idx).to_numpy(int)
        movie_idx = interactions["movie_id"].map(self._movie_to_idx).to_numpy(int)
        ratings = interactions["rating"].to_numpy(float)
        return user_idx, movie_idx, ratings

    def _train(
        self, user_idx: np.ndarray, movie_idx: np.ndarray, ratings: np.ndarray
    ) -> list[float]:
        rng = np.random.default_rng(self.config.random_state)
        epoch_rmse: list[float] = []
        for _ in range(self.config.epochs):
            order = rng.permutation(len(ratings))
            self._run_epoch(user_idx, movie_idx, ratings, order)
            predictions = self._predict_known_indices(user_idx, movie_idx)
            epoch_rmse.append(float(np.sqrt(np.mean((ratings - predictions) ** 2))))
        return epoch_rmse

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
        scores = self._score_movies(user_id)
        candidate_indices = self._candidate_indices(user_id)
        order = np.argsort(-scores[candidate_indices], kind="stable")[:k]
        assert self._idx_to_movie is not None
        return self._idx_to_movie[candidate_indices[order]].astype(int).tolist()

    def _score_movies(self, user_id: int) -> np.ndarray:
        assert self._movie_bias is not None
        assert self._movie_factors is not None
        user_idx = self._user_to_idx.get(user_id)
        scores = self._global_mean + self._movie_bias.copy()
        if user_idx is None:
            return scores
        assert self._user_bias is not None
        assert self._user_factors is not None
        scores += self._user_bias[user_idx]
        scores += self._movie_factors @ self._user_factors[user_idx]
        return scores

    def _candidate_indices(self, user_id: int) -> np.ndarray:
        assert self._idx_to_movie is not None
        seen = self._seen_items.get(user_id, set())
        candidate_mask = np.array(
            [movie_id not in seen for movie_id in self._idx_to_movie], dtype=bool
        )
        return np.flatnonzero(candidate_mask)

    @property
    def catalog_size(self) -> int:
        """Return the number of movies learned during fit."""
        self._ensure_fitted()
        assert self._idx_to_movie is not None
        return len(self._idx_to_movie)

    def _initialize_parameters(self, n_users: int, n_movies: int) -> None:
        rng = np.random.default_rng(self.config.random_state)
        self._user_bias = np.zeros(n_users)
        self._movie_bias = np.zeros(n_movies)
        shape_users = (n_users, self.config.n_factors)
        shape_movies = (n_movies, self.config.n_factors)
        self._user_factors = rng.normal(0.0, 0.1, shape_users)
        self._movie_factors = rng.normal(0.0, 0.1, shape_movies)

    def _run_epoch(
        self,
        user_idx: np.ndarray,
        movie_idx: np.ndarray,
        ratings: np.ndarray,
        order: np.ndarray,
    ) -> None:
        for sample_idx in order:
            self._update_sample(
                user_idx[sample_idx], movie_idx[sample_idx], ratings[sample_idx]
            )

    def _update_sample(self, user_idx: int, movie_idx: int, rating: float) -> None:
        assert self._user_factors is not None
        assert self._movie_factors is not None
        user_vector = self._user_factors[user_idx].copy()
        movie_vector = self._movie_factors[movie_idx].copy()
        error = rating - self._sample_prediction(user_idx, movie_idx)
        self._update_biases(user_idx, movie_idx, error)
        self._update_factors(user_idx, movie_idx, error, user_vector, movie_vector)

    def _sample_prediction(self, user_idx: int, movie_idx: int) -> float:
        assert self._user_bias is not None
        assert self._movie_bias is not None
        assert self._user_factors is not None
        assert self._movie_factors is not None
        return float(
            self._global_mean
            + self._user_bias[user_idx]
            + self._movie_bias[movie_idx]
            + np.dot(self._user_factors[user_idx], self._movie_factors[movie_idx])
        )

    def _update_biases(self, user_idx: int, movie_idx: int, error: float) -> None:
        assert self._user_bias is not None
        assert self._movie_bias is not None
        lr = self.config.learning_rate
        reg = self.config.regularization
        self._user_bias[user_idx] += lr * (error - reg * self._user_bias[user_idx])
        self._movie_bias[movie_idx] += lr * (error - reg * self._movie_bias[movie_idx])

    def _update_factors(
        self,
        user_idx: int,
        movie_idx: int,
        error: float,
        user_vector: np.ndarray,
        movie_vector: np.ndarray,
    ) -> None:
        assert self._user_factors is not None
        assert self._movie_factors is not None
        lr = self.config.learning_rate
        reg = self.config.regularization
        self._user_factors[user_idx] += lr * (error * movie_vector - reg * user_vector)
        self._movie_factors[movie_idx] += lr * (
            error * user_vector - reg * movie_vector
        )

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

    def _validate_config(self) -> None:
        if self.config.n_factors < 1 or self.config.epochs < 1:
            raise ValueError("n_factors and epochs must be positive")
        if self.config.learning_rate <= 0 or self.config.regularization < 0:
            raise ValueError("invalid optimization parameters")

    @staticmethod
    def _validate_interactions(interactions: pd.DataFrame) -> None:
        required = {"user_id", "movie_id", "rating"}
        missing = required - set(interactions.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        if interactions.empty:
            raise ValueError("interactions cannot be empty")
