"""Neural collaborative filtering for explicit MovieLens ratings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.seeds import set_global_seed

RATING_MIN = 0.5
RATING_MAX = 5.0


@dataclass(frozen=True, slots=True)
class NeuralTrainingHistory:
    """Training RMSE measured after each epoch."""

    train_rmse: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class NeuralRecommenderConfig:
    """Architecture and optimization parameters."""

    embedding_dim: int = 32
    hidden_dims: tuple[int, ...] = (64, 32)
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    ranking_weight: float = 1.0
    relevance_threshold: float = 4.0
    batch_size: int = 1024
    epochs: int = 10
    random_state: int = 42
    device: str = "cpu"


class NeuralRatingNetwork(nn.Module):
    """Combine user and movie embeddings in a multilayer perceptron."""

    def __init__(
        self, n_users: int, n_movies: int, config: NeuralRecommenderConfig
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, config.embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, config.embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.movie_embedding.weight, std=0.05)
        input_dim = config.embedding_dim * 3
        self.mlp = _build_mlp(input_dim, config.hidden_dims)

    def forward(
        self, user_indices: torch.Tensor, movie_indices: torch.Tensor
    ) -> torch.Tensor:
        """Predict bounded explicit ratings for indexed pairs."""
        user_vectors = self.user_embedding(user_indices)
        movie_vectors = self.movie_embedding(movie_indices)
        interaction = user_vectors * movie_vectors
        features = torch.cat((user_vectors, movie_vectors, interaction), dim=1)
        logits = self.mlp(features).squeeze(1)
        return RATING_MIN + (RATING_MAX - RATING_MIN) * torch.sigmoid(logits)


class PyTorchRecommender:
    """Train and serve a neural explicit-feedback recommender."""

    def __init__(self, config: NeuralRecommenderConfig | None = None) -> None:
        self.config = config or NeuralRecommenderConfig()
        self._validate_config()
        self._device = _resolve_device(self.config.device)
        self._reset_state()

    def _reset_state(self) -> None:
        self._user_to_idx: dict[int, int] = {}
        self._movie_to_idx: dict[int, int] = {}
        self._idx_to_movie = np.array([], dtype=int)
        self._seen_items: dict[int, set[int]] = {}
        self._global_mean = 0.0
        self._known_matrix: torch.Tensor | None = None
        self._known_counts: torch.Tensor | None = None
        self.network: NeuralRatingNetwork | None = None
        self.history: NeuralTrainingHistory | None = None

    def fit(self, interactions: pd.DataFrame) -> PyTorchRecommender:
        """Fit embeddings and MLP using explicit ratings."""
        _validate_interactions(interactions)
        set_global_seed(self.config.random_state)
        dataset = self._prepare_fit(interactions)
        self._known_matrix = self._build_known_matrix(dataset)
        self._known_counts = self._known_matrix.sum(dim=1)
        self.network = NeuralRatingNetwork(
            len(self._user_to_idx), len(self._movie_to_idx), self.config
        ).to(self._device)
        losses = self._train(dataset)
        self.history = NeuralTrainingHistory(train_rmse=losses)
        return self

    def _prepare_fit(self, interactions: pd.DataFrame) -> TensorDataset:
        users = np.sort(interactions["user_id"].unique()).astype(int)
        movies = np.sort(interactions["movie_id"].unique()).astype(int)
        self._user_to_idx = {user_id: idx for idx, user_id in enumerate(users)}
        self._movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies)}
        self._idx_to_movie = movies
        self._seen_items = _build_seen_items(interactions)
        self._global_mean = float(interactions["rating"].mean())
        return _build_dataset(interactions, self._user_to_idx, self._movie_to_idx)

    def _train(self, dataset: TensorDataset) -> tuple[float, ...]:
        assert self.network is not None
        loader = self._build_loader(dataset)
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        losses = [
            self._train_epoch(loader, optimizer) for _ in range(self.config.epochs)
        ]
        return tuple(losses)

    def _build_loader(self, dataset: TensorDataset) -> DataLoader:
        generator = torch.Generator().manual_seed(self.config.random_state)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
        )

    def _build_known_matrix(self, dataset: TensorDataset) -> torch.Tensor:
        users, movies, _ = dataset.tensors
        shape = (len(self._user_to_idx), len(self._movie_to_idx))
        matrix = torch.zeros(shape, dtype=torch.bool, device=self._device)
        matrix[users.to(self._device), movies.to(self._device)] = True
        return matrix

    def _train_epoch(
        self, loader: DataLoader, optimizer: torch.optim.Optimizer
    ) -> float:
        assert self.network is not None
        self.network.train()
        squared_error = 0.0
        sample_count = 0
        for users, movies, ratings in loader:
            users, movies, ratings = self._move_batch(users, movies, ratings)
            optimizer.zero_grad()
            predictions = self.network(users, movies)
            mean_squared_error = nn.functional.mse_loss(predictions, ratings)
            loss = mean_squared_error + self._ranking_loss(users, ratings, predictions)
            loss.backward()
            optimizer.step()
            squared_error += mean_squared_error.item() * len(ratings)
            sample_count += len(ratings)
        return float(np.sqrt(squared_error / sample_count))

    def _ranking_loss(
        self, users: torch.Tensor, ratings: torch.Tensor, positive_scores: torch.Tensor
    ) -> torch.Tensor:
        assert self._known_counts is not None
        relevant = ratings >= self.config.relevance_threshold
        has_negative = self._known_counts[users] < self.catalog_size
        eligible = relevant & has_negative
        if not eligible.any() or self.config.ranking_weight == 0:
            return positive_scores.new_tensor(0.0)
        relevant_users = users[eligible]
        negative_movies = self._sample_negatives(relevant_users)
        assert self.network is not None
        negative_scores = self.network(relevant_users, negative_movies)
        pairwise = -nn.functional.logsigmoid(
            positive_scores[eligible] - negative_scores
        )
        return self.config.ranking_weight * pairwise.mean()

    def _sample_negatives(self, users: torch.Tensor) -> torch.Tensor:
        assert self._known_matrix is not None
        count = len(users)
        movies = torch.randint(self.catalog_size, (count,), device=self._device)
        collisions = self._known_matrix[users, movies]
        while collisions.any():
            movies[collisions] = torch.randint(
                self.catalog_size, (int(collisions.sum()),), device=self._device
            )
            collisions = self._known_matrix[users, movies]
        return movies

    def _move_batch(
        self, users: torch.Tensor, movies: torch.Tensor, ratings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            users.to(self._device),
            movies.to(self._device),
            ratings.to(self._device),
        )

    def predict_pairs(self, pairs: pd.DataFrame) -> np.ndarray:
        """Predict ratings and use the training mean for unknown IDs."""
        self._ensure_fitted()
        _validate_pairs(pairs)
        predictions = np.full(len(pairs), self._global_mean, dtype=float)
        user_indices = pairs["user_id"].map(self._user_to_idx)
        movie_indices = pairs["movie_id"].map(self._movie_to_idx)
        known = user_indices.notna() & movie_indices.notna()
        if known.any():
            predictions[known] = self._predict_indices(
                user_indices[known].to_numpy(np.int64),
                movie_indices[known].to_numpy(np.int64),
            )
        return np.clip(predictions, RATING_MIN, RATING_MAX)

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Rank unseen training-catalog movies for a user."""
        self._ensure_fitted()
        if k < 1:
            raise ValueError("k must be positive")
        candidate_indices = self._candidate_indices(user_id)
        if len(candidate_indices) == 0:
            return []
        scores = self._score_candidates(user_id, candidate_indices)
        movie_ids = self._idx_to_movie[candidate_indices]
        order = np.lexsort((movie_ids, -scores))[:k]
        return movie_ids[order].astype(int).tolist()

    def _score_candidates(self, user_id: int, movie_indices: np.ndarray) -> np.ndarray:
        user_idx = self._user_to_idx.get(user_id)
        if user_idx is None:
            return np.full(len(movie_indices), self._global_mean)
        user_indices = np.full(len(movie_indices), user_idx, dtype=np.int64)
        return self._predict_indices(user_indices, movie_indices)

    def _predict_indices(
        self, user_indices: np.ndarray, movie_indices: np.ndarray
    ) -> np.ndarray:
        assert self.network is not None
        self.network.eval()
        batches = []
        with torch.no_grad():
            for start in range(0, len(user_indices), self.config.batch_size):
                stop = start + self.config.batch_size
                users = torch.as_tensor(user_indices[start:stop], device=self._device)
                movies = torch.as_tensor(movie_indices[start:stop], device=self._device)
                batches.append(self.network(users, movies).cpu().numpy())
        return np.concatenate(batches)

    def _candidate_indices(self, user_id: int) -> np.ndarray:
        seen = self._seen_items.get(user_id, set())
        mask = np.array([movie_id not in seen for movie_id in self._idx_to_movie])
        return np.flatnonzero(mask)

    @property
    def catalog_size(self) -> int:
        """Return the fitted movie catalog size."""
        self._ensure_fitted()
        return len(self._idx_to_movie)

    def _ensure_fitted(self) -> None:
        if self.network is None:
            raise RuntimeError("model is not fitted")

    def _validate_config(self) -> None:
        positive = (
            self.config.embedding_dim,
            self.config.batch_size,
            self.config.epochs,
        )
        if any(value < 1 for value in positive) or not self.config.hidden_dims:
            raise ValueError("dimensions, batch_size, and epochs must be positive")
        if any(dimension < 1 for dimension in self.config.hidden_dims):
            raise ValueError("hidden dimensions must be positive")
        if self.config.learning_rate <= 0 or self.config.weight_decay < 0:
            raise ValueError("invalid optimization parameters")
        if self.config.ranking_weight < 0:
            raise ValueError("ranking_weight cannot be negative")
        if not RATING_MIN <= self.config.relevance_threshold <= RATING_MAX:
            raise ValueError("relevance_threshold must be within the rating scale")


def _build_mlp(input_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(current_dim, hidden_dim), nn.ReLU()))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, 1))
    return nn.Sequential(*layers)


def _build_dataset(
    interactions: pd.DataFrame,
    user_to_idx: dict[int, int],
    movie_to_idx: dict[int, int],
) -> TensorDataset:
    users = interactions["user_id"].map(user_to_idx).to_numpy(np.int64)
    movies = interactions["movie_id"].map(movie_to_idx).to_numpy(np.int64)
    ratings = interactions["rating"].to_numpy(np.float32)
    return TensorDataset(
        torch.from_numpy(users), torch.from_numpy(movies), torch.from_numpy(ratings)
    )


def _build_seen_items(interactions: pd.DataFrame) -> dict[int, set[int]]:
    return interactions.groupby("user_id")["movie_id"].agg(set).to_dict()


def _resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    return device


def _validate_interactions(interactions: pd.DataFrame) -> None:
    required = {"user_id", "movie_id", "rating"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if interactions.empty:
        raise ValueError("interactions cannot be empty")


def _validate_pairs(pairs: pd.DataFrame) -> None:
    missing = {"user_id", "movie_id"}.difference(pairs.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
