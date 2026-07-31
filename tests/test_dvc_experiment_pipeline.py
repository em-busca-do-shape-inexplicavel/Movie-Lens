from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from data.pipeline import (
    build_temporal_features,
    load_feature_splits,
    preprocess_ratings,
)
from training.configuration import TrainingPipelineConfig, load_experiment_config
from training.experiments import CandidateExperimentResult, select_best_candidate
from training.pipeline import TrainingArtifacts, TrainingPipelineResult


def test_data_stages_persist_leakage_safe_temporal_splits(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _sample_ratings().to_csv(raw_dir / "ratings.csv", index=False)

    dataset = preprocess_ratings(raw_dir, tmp_path / "processed")
    artifacts = build_temporal_features(dataset, tmp_path / "features")
    train, validation, test = load_feature_splits(tmp_path / "features")

    assert all(
        path.exists()
        for path in (
            artifacts.train,
            artifacts.validation,
            artifacts.test,
            artifacts.metadata,
        )
    )
    assert (len(train), len(validation), len(test)) == (6, 3, 3)
    _assert_temporal_order(train, validation, test)
    metadata = json.loads(artifacts.metadata.read_text())
    assert metadata["train_users"] == 3


def test_experiment_config_requires_three_named_candidates(tmp_path: Path) -> None:
    path = tmp_path / "params.yaml"
    path.write_text(
        """
training:
  embedding_dim: 4
experiments:
  selection_metric: validation_ndcg@2
  direction: maximize
  candidates:
    - name: low
      overrides: {ranking_weight: 0.2}
    - name: medium
      overrides: {ranking_weight: 1.0}
    - name: high
      overrides: {ranking_weight: 2.0}
""".strip(),
        encoding="utf-8",
    )

    experiment = load_experiment_config(path)

    assert [candidate.name for candidate in experiment.candidates] == [
        "low",
        "medium",
        "high",
    ]
    assert experiment.candidates[2].model.ranking_weight == 2.0


def test_candidate_selection_uses_validation_metric_only(tmp_path: Path) -> None:
    candidates = [
        _candidate(tmp_path, "low", 0.01),
        _candidate(tmp_path, "winner", 0.03),
        _candidate(tmp_path, "high", 0.02),
    ]

    winner = select_best_candidate(candidates, "validation_ndcg@10", "maximize")

    assert winner.name == "winner"


def test_dvc_pipeline_has_required_stages() -> None:
    payload = yaml.safe_load(Path("dvc.yaml").read_text(encoding="utf-8"))

    assert list(payload["stages"]) == [
        "validate",
        "preprocess",
        "feature_eng",
        "train",
        "evaluate",
    ]


def _candidate(tmp_path: Path, name: str, score: float) -> CandidateExperimentResult:
    artifact = tmp_path / name
    artifacts = TrainingArtifacts(artifact, artifact, artifact, artifact)
    result = TrainingPipelineResult({"validation_ndcg@10": score}, artifacts)
    model = load_experiment_config(Path("params.yaml")).candidates[0].model
    config = TrainingPipelineConfig(model)
    return CandidateExperimentResult(name, config, result, f"run-{name}")


def _assert_temporal_order(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> None:
    for user_id in validation["user_id"]:
        user_train = train.loc[train["user_id"] == user_id, "timestamp"]
        user_validation = validation.loc[validation["user_id"] == user_id, "timestamp"]
        user_test = test.loc[test["user_id"] == user_id, "timestamp"]
        assert user_train.max() < user_validation.iloc[0] < user_test.iloc[0]


def _sample_ratings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "movieId": [10, 11, 12, 13, 10, 12, 14, 15, 11, 13, 14, 16],
            "rating": [5, 4, 5, 5, 4, 3, 4, 5, 5, 4, 4, 5],
            "timestamp": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
        }
    )
