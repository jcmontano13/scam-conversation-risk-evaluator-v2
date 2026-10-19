"""
test_model_registry.py

Description:
    Runs automated unit checks on model_registry.py. These tests confirm that
    the registry returns all five COMP903 models, that scaling is applied
    inside the pipelines that need it, that models with randomness use the
    fixed project seed, and that tuned settings are applied correctly.

Inputs:
    No direct command-line input.
    The tests use src/models/model_registry.py and config.py, plus a small
    synthetic dataset created inside the tests.

Output:
    No output file is created by the tests themselves.
    Test results are displayed in the terminal through pytest.

Run:
    - pytest tests/test_model_registry.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import MODEL_NAMES, SEED
from src.models.model_registry import SCALED_MODELS, build_model, build_models

EXPECTED_ESTIMATORS = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Gaussian Naive Bayes": GaussianNB,
    "Random Forest": RandomForestClassifier,
    "k-Nearest Neighbours": KNeighborsClassifier,
}

MODELS_WITH_SEED = ["Logistic Regression", "Decision Tree", "Random Forest"]


def make_small_dataset(rows=60, seed=0):
    """
    Description:
        Creates a small synthetic dataset with four numeric features and a
        binary label, so the tests do not depend on the project data files.

    Output:
        Returns X (rows x 4 array) and y (binary labels).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=[10, 0, 2, 0.5], scale=[3, 0.1, 2, 0.15], size=(rows, 4))
    y = (X[:, 0] + X[:, 2] > 12).astype(int)
    return X, y


def test_five_models_are_returned_in_config_order():
    models = build_models()
    assert list(models.keys()) == MODEL_NAMES
    assert len(models) == 5


def test_every_model_is_a_pipeline_ending_in_model_step():
    for model_name, pipeline in build_models().items():
        assert isinstance(pipeline, Pipeline), f"{model_name} is not a Pipeline"
        assert pipeline.steps[-1][0] == "model", f"{model_name} last step is not 'model'"


def test_each_model_uses_the_expected_estimator():
    for model_name, pipeline in build_models().items():
        estimator = pipeline.named_steps["model"]
        assert isinstance(estimator, EXPECTED_ESTIMATORS[model_name])


def test_logistic_regression_and_knn_are_scaled():
    assert set(SCALED_MODELS) == {"Logistic Regression", "k-Nearest Neighbours"}

    for model_name in SCALED_MODELS:
        pipeline = build_model(model_name)
        assert pipeline.steps[0][0] == "scaler"
        assert isinstance(pipeline.named_steps["scaler"], StandardScaler)


def test_tree_and_probabilistic_models_are_not_scaled():
    for model_name in ["Decision Tree", "Gaussian Naive Bayes", "Random Forest"]:
        pipeline = build_model(model_name)
        assert "scaler" not in pipeline.named_steps, f"{model_name} should not be scaled"


def test_models_with_randomness_use_project_seed():
    assert SEED == 42

    for model_name in MODELS_WITH_SEED:
        estimator = build_model(model_name).named_steps["model"]
        assert estimator.random_state == SEED, f"{model_name} does not use random_state={SEED}"


def test_logistic_regression_matches_comp902_settings():
    estimator = build_model("Logistic Regression").named_steps["model"]
    assert estimator.max_iter == 1000
    assert estimator.random_state == SEED


def test_random_forest_gives_identical_results_when_rerun():
    X, y = make_small_dataset()
    first = build_model("Random Forest").fit(X, y).predict_proba(X)
    second = build_model("Random Forest").fit(X, y).predict_proba(X)
    assert np.array_equal(first, second)


def test_scaler_is_fitted_on_training_rows_only():
    X, y = make_small_dataset(rows=80)
    X_train, y_train = X[:40], y[:40]

    pipeline = build_model("k-Nearest Neighbours").fit(X_train, y_train)
    scaler = pipeline.named_steps["scaler"]

    assert np.allclose(scaler.mean_, X_train.mean(axis=0))
    assert not np.allclose(scaler.mean_, X.mean(axis=0))


def test_tuned_settings_are_applied_in_both_key_formats():
    plain = build_model("k-Nearest Neighbours", {"n_neighbors": 15})
    prefixed = build_model("k-Nearest Neighbours", {"model__n_neighbors": 15})

    assert plain.named_steps["model"].n_neighbors == 15
    assert prefixed.named_steps["model"].n_neighbors == 15


def test_settings_for_one_model_do_not_change_the_others():
    models = build_models({"Random Forest": {"max_depth": 5}})

    assert models["Random Forest"].named_steps["model"].max_depth == 5
    assert models["Decision Tree"].named_steps["model"].max_depth is None


def test_unknown_model_names_are_rejected():
    with pytest.raises(ValueError):
        build_model("Support Vector Machine")

    with pytest.raises(ValueError):
        build_models({"Support Vector Machine": {"C": 1.0}})


def test_each_call_returns_new_untrained_models():
    first = build_models()
    second = build_models()

    for model_name in MODEL_NAMES:
        assert first[model_name] is not second[model_name]

        with pytest.raises(NotFittedError):
            first[model_name].predict(np.zeros((1, 4)))