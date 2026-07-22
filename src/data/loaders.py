"""MovieLens CSV loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ratings(raw_data_dir: Path) -> pd.DataFrame:
    """Load ratings with normalized column names."""
    path = raw_data_dir / "ratings.csv"
    ratings = pd.read_csv(path)
    return ratings.rename(
        columns={"userId": "user_id", "movieId": "movie_id"}
    )


def load_movies(raw_data_dir: Path) -> pd.DataFrame:
    """Load movies with a normalized movie identifier."""
    path = raw_data_dir / "movies.csv"
    movies = pd.read_csv(path)
    return movies.rename(columns={"movieId": "movie_id"})
