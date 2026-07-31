from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from models.pytorch_recommender import NeuralRecommenderConfig
from tracking.mlflow_tracker import register_model_version, track_training_run
from training.configuration import TrainingPipelineConfig
from training.pipeline import run_training_pipeline


def test_mlflow_tracking_logs_run_and_promotes_model(tmp_path: Path) -> None:
    raw_data_dir = tmp_path / "raw"
    raw_data_dir.mkdir()
    _sample_ratings().to_csv(raw_data_dir / "ratings.csv", index=False)
    config = TrainingPipelineConfig(model=_test_model_config(), top_k=2)
    params_path = tmp_path / "params.yaml"
    params_path.write_text("training:\n  embedding_dim: 4\n", encoding="utf-8")

    result = run_training_pipeline(raw_data_dir, tmp_path / "artifacts", config)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "movie-lens-recommender"
    mlflow.create_experiment(
        experiment_name,
        artifact_location=(tmp_path / "mlartifacts").as_uri(),
    )

    tracked = track_training_run(
        config=config,
        result=result,
        params_path=params_path,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        model_name="movie-lens-pytorch-recommender",
    )

    assert tracked.run_id
    assert tracked.registered_model_version is not None

    run = mlflow.get_run(tracked.run_id)
    assert run.data.params["model"] == "pytorch"
    assert float(run.data.metrics["training_seconds"]) >= 0

    downloaded = mlflow.artifacts.download_artifacts(
        run_id=tracked.run_id, artifact_path="model.pt"
    )
    assert Path(downloaded).exists()

    client = MlflowClient()
    version = client.get_model_version(
        tracked.model_name, tracked.registered_model_version
    )
    assert version.current_stage == "Production"
    aliased = client.get_model_version_by_alias(tracked.model_name, "production")
    assert str(aliased.version) == tracked.registered_model_version

    loaded = mlflow.pyfunc.load_model(f"models:/{tracked.model_name}@production")
    predictions = loaded.predict(pd.DataFrame({"user_id": [1], "movie_id": [10]}))
    assert list(predictions.columns) == ["prediction"]
    assert len(predictions) == 1
    assert loaded.metadata.signature is not None

    candidate = track_training_run(
        config=config,
        result=result,
        params_path=params_path,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        model_name=tracked.model_name,
        publish_model=False,
        run_name="candidate-test",
        extra_tags={"run_kind": "candidate"},
    )
    assert candidate.registered_model_version is None
    assert mlflow.get_run(candidate.run_id).data.tags["run_kind"] == "candidate"
    versions = MlflowClient().search_model_versions(f"name='{tracked.model_name}'")
    assert len(versions) == 1

    other_tracking_uri = f"sqlite:///{tmp_path / 'other.db'}"
    mlflow.set_tracking_uri(other_tracking_uri)
    registered_again = register_model_version(
        run_id=tracked.run_id,
        model_name=tracked.model_name,
        stage="Staging",
        alias="candidate",
        tracking_uri=tracking_uri,
    )
    assert str(registered_again.version) == "2"
    assert mlflow.get_tracking_uri() == tracking_uri
    candidate = MlflowClient().get_model_version_by_alias(
        tracked.model_name, "candidate"
    )
    assert str(candidate.version) == str(registered_again.version)


def _sample_ratings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "movieId": [10, 11, 12, 13, 10, 12, 14, 15, 11, 13, 14, 16],
            "rating": [5, 4, 5, 5, 4, 3, 4, 5, 5, 4, 4, 5],
            "timestamp": list(range(1, 13)),
        }
    )


def _test_model_config() -> NeuralRecommenderConfig:
    return NeuralRecommenderConfig(
        embedding_dim=4,
        hidden_dims=(8,),
        ranking_weight=0.0,
        batch_size=3,
        epochs=4,
        early_stopping_patience=1,
        early_stopping_min_delta=100.0,
    )
