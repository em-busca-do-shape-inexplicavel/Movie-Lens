"""Persist reproducible, leakage-safe datasets for the DVC pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.loaders import load_ratings
from data.splitting import temporal_leave_two_out

INTERACTION_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]


@dataclass(frozen=True, slots=True)
class FeatureArtifacts:
    """Model-ready temporal split files."""

    train: Path
    validation: Path
    test: Path
    metadata: Path


def preprocess_ratings(raw_data_dir: Path, output_dir: Path) -> Path:
    """Clean raw ratings and persist a normalized Parquet dataset."""
    ratings = load_ratings(raw_data_dir)
    cleaned = _clean_ratings(ratings)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = output_dir / "ratings.parquet"
    cleaned.to_parquet(dataset, index=False)
    _write_json(output_dir / "preprocess_summary.json", _summary(ratings, cleaned))
    return dataset


def build_temporal_features(dataset: Path, output_dir: Path) -> FeatureArtifacts:
    """Create temporal train, validation, and test feature tables."""
    ratings = pd.read_parquet(dataset)
    train, validation, test = temporal_leave_two_out(ratings)
    artifacts = _feature_artifacts(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_splits(artifacts, train, validation, test)
    _write_json(artifacts.metadata, _feature_metadata(train, validation, test))
    return artifacts


def load_feature_splits(
    feature_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load model-ready temporal split tables."""
    artifacts = _feature_artifacts(feature_dir)
    return tuple(
        pd.read_parquet(path)
        for path in (artifacts.train, artifacts.validation, artifacts.test)
    )


def _clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    missing = set(INTERACTION_COLUMNS).difference(ratings.columns)
    if missing:
        raise ValueError(f"missing rating columns: {sorted(missing)}")
    cleaned = ratings.loc[:, INTERACTION_COLUMNS].dropna().drop_duplicates()
    cleaned = cleaned.astype(
        {
            "user_id": "int64",
            "movie_id": "int64",
            "rating": "float32",
            "timestamp": "int64",
        }
    )
    if not cleaned["rating"].between(0.5, 5.0).all():
        raise ValueError("ratings must be between 0.5 and 5.0")
    return cleaned.sort_values(["user_id", "timestamp", "movie_id"], kind="stable")


def _summary(raw: pd.DataFrame, cleaned: pd.DataFrame) -> dict[str, int]:
    return {
        "raw_rows": len(raw),
        "clean_rows": len(cleaned),
        "removed_rows": len(raw) - len(cleaned),
        "users": cleaned["user_id"].nunique(),
        "movies": cleaned["movie_id"].nunique(),
    }


def _feature_artifacts(output_dir: Path) -> FeatureArtifacts:
    return FeatureArtifacts(
        train=output_dir / "train.parquet",
        validation=output_dir / "validation.parquet",
        test=output_dir / "test.parquet",
        metadata=output_dir / "feature_metadata.json",
    )


def _write_splits(
    artifacts: FeatureArtifacts,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    for path, frame in zip(
        (artifacts.train, artifacts.validation, artifacts.test),
        (train, validation, test),
        strict=True,
    ):
        frame.loc[:, INTERACTION_COLUMNS].to_parquet(path, index=False)


def _feature_metadata(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> dict[str, int]:
    return {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_users": train["user_id"].nunique(),
        "train_movies": train["movie_id"].nunique(),
    }


def _write_json(path: Path, payload: dict[str, int]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
