"""
model_registry.py

Description:
    Builds the five machine learning models compared in COMP903 from one
    place, so every script (cross-validation, tuning, testing, calibration,
    ablation) trains exactly the same model definitions.

    Every model is wrapped in a scikit-learn Pipeline with a final step
    named "model". Models that depend on feature scale (Logistic Regression
    and k-Nearest Neighbours) also have a "scaler" step. Because the scaler
    sits inside the pipeline, it is fitted on the training folds only, which
    prevents information from the validation or test rows leaking into
    training.

    Logistic Regression and Decision Tree use the same settings as the
    COMP902 baseline script (train_baseline_models.py), so their results
    stay comparable with COMP902.

Inputs:
    - config.py (SEED and MODEL_NAMES)

Output:
    - No files. Other scripts import build_models() or build_model().

Run:
    - python src/models/model_registry.py
      (prints a summary of the five model definitions)
"""

import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import MODEL_NAMES, SEED

# Models that need their features scaled before training
SCALED_MODELS = ["Logistic Regression", "k-Nearest Neighbours"]


def create_estimator(model_name):
    """
    Description:
        Creates the untrained scikit-learn estimator for one model.

    Input:
        model_name: One of the names in config.MODEL_NAMES.

    Output:
        Returns a scikit-learn estimator with the default COMP903 settings.
    """
    if model_name == "Logistic Regression":
        return LogisticRegression(random_state=SEED, max_iter=1000)

    if model_name == "Decision Tree":
        return DecisionTreeClassifier(random_state=SEED)

    if model_name == "Gaussian Naive Bayes":
        return GaussianNB()

    if model_name == "Random Forest":
        return RandomForestClassifier(random_state=SEED)

    if model_name == "k-Nearest Neighbours":
        return KNeighborsClassifier()

    raise ValueError(f"Unknown model name: {model_name}. Expected one of {MODEL_NAMES}")


def build_model(model_name, params=None):
    """
    Description:
        Builds one model as a Pipeline. A "scaler" step is added for models
        in SCALED_MODELS. Optional parameters (for example, the settings
        selected by grid search) are applied to the "model" step.

    Input:
        model_name: One of the names in config.MODEL_NAMES.
        params: Optional dictionary of model settings, for example
                {"n_neighbors": 15}. Keys may also use the pipeline form
                "model__n_neighbors".

    Output:
        Returns an untrained scikit-learn Pipeline.
    """
    steps = []

    if model_name in SCALED_MODELS:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", create_estimator(model_name)))
    pipeline = Pipeline(steps=steps)

    if params:
        pipeline_params = {
            key if key.startswith("model__") else f"model__{key}": value
            for key, value in params.items()
        }
        pipeline.set_params(**pipeline_params)

    return pipeline


def build_models(params_by_model=None):
    """
    Description:
        Builds all five models in the order given by config.MODEL_NAMES.

    Input:
        params_by_model: Optional dictionary mapping a model name to its
                         settings, for example
                         {"k-Nearest Neighbours": {"n_neighbors": 15}}.
                         Models not listed keep their default settings.

    Output:
        Returns a dictionary of model names and untrained Pipelines.
    """
    params_by_model = params_by_model or {}

    unknown = set(params_by_model) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Settings given for unknown models: {unknown}")

    return {
        model_name: build_model(model_name, params_by_model.get(model_name))
        for model_name in MODEL_NAMES
    }


def describe_models(models):
    """
    Description:
        Creates a short text summary of each model's pipeline steps and
        main settings.

    Input:
        models: Dictionary returned by build_models().

    Output:
        Returns a list of text lines.
    """
    lines = []

    for model_name, pipeline in models.items():
        steps = " -> ".join(name for name, _ in pipeline.steps)
        estimator = pipeline.named_steps["model"]
        seed = estimator.get_params().get("random_state", "not used")
        lines.append(f"{model_name}: {steps} | {type(estimator).__name__} | random_state={seed}")

    return lines


def main():
    models = build_models()

    print("COMP903 model registry")
    print("-" * 22)
    for line in describe_models(models):
        print(line)


if __name__ == "__main__":
    main()