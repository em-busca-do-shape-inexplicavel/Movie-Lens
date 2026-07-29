from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
from configs.settings import load_settings

from data.preprocessors.explicit import ExplicitFeedbackPreprocessor
from data.preprocessors.factory import create_preprocessor
from data.preprocessors.implicit import ImplicitFeedbackPreprocessor
from models.factory import create_model
from models.matrix_factorization import MatrixFactorizationRecommender
from models.popularity import PopularityRecommender
from models.pytorch_recommender import PyTorchRecommender
from models.sklearn_baselines import (
    SklearnBiasRecommender,
    SklearnMeanRecommender,
)
from training.seeds import set_global_seed


def test_model_factory_creates_registered_models() -> None:
    assert isinstance(create_model("popularity"), PopularityRecommender)
    assert isinstance(
        create_model("matrix_factorization"), MatrixFactorizationRecommender
    )
    assert isinstance(create_model("sklearn_mean"), SklearnMeanRecommender)
    assert isinstance(create_model("sklearn_bias"), SklearnBiasRecommender)
    assert isinstance(create_model("pytorch"), PyTorchRecommender)


def test_model_factory_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        create_model("unknown")


def test_preprocessor_factory_creates_strategies() -> None:
    explicit = create_preprocessor("explicit")
    implicit = create_preprocessor("implicit")

    assert isinstance(explicit, ExplicitFeedbackPreprocessor)
    assert isinstance(implicit, ImplicitFeedbackPreprocessor)


def test_implicit_strategy_converts_relevant_ratings() -> None:
    ratings = pd.DataFrame({"rating": [3.0, 4.0, 5.0]})
    strategy = create_preprocessor("implicit", relevance_threshold=4.0)

    transformed = strategy.transform(ratings)

    assert transformed["rating"].tolist() == [4.0, 5.0]
    assert transformed["interaction"].tolist() == [1, 1]
    assert (transformed["feedback"] == "implicit").all()


def test_global_seed_is_reproducible() -> None:
    set_global_seed(42)
    first = (random.random(), np.random.random())
    set_global_seed(42)
    second = (random.random(), np.random.random())

    assert first == second


def test_settings_resolve_project_data_directory() -> None:
    settings = load_settings()

    assert settings.resolved_data_dir.name == "data"
    assert settings.random_seed == 42
