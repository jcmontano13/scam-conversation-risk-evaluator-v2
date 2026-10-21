"""
tune_five_models.py

Description:
    Tunes all five COMP903 models with grid search cross-validation on the
    saved training split only. Each model is built by model_registry.py and
    searched over a grid of similar size (20 to 24 setting combinations), so
    no model looks better simply because it was tuned more.

    Every grid is searched with the same stratified 5-fold cross-validation
    and the same scoring. The best settings are chosen by F1-score (the
    primary metric for RQ1) and saved for the later scripts (repeated
    cross-validation, held-out test, calibration and ablation).

Inputs:
    - data/processed/scam_dialogue_thread_features.csv
    - data/processed/split_indices.json
    - outputs/reports/cv_results_five_models.csv (optional, for the
      before-and-after tuning comparison)

Outputs:
    - outputs/reports/grid_search_results_five_models.csv
    - outputs/reports/selected_hyperparams_five_models.json

Run:
    - python src/models/tune_five_models.py

Important:
    - Only train_row_indices from split_indices.json are used.
    - The held-out test set is not used in this script.
"""

import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import (
    CV_FOLDS,
    CV_RESULTS_FIVE_MODELS,
    FEATURE_COLUMNS,
    GRID_SEARCH_RESULTS_FIVE_MODELS,
    MODEL_NAMES,
    PRIMARY_METRIC,
    PROCESSED_DATA,
    SEED,
    SELECTED_HYPERPARAMS_FIVE_MODELS,
    SPLIT_INDICES,
    TARGET_COLUMN,
)
from src.models.model_registry import build_model

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

# Grids of similar size for every model (20 to 24 combinations).
# Keys use the pipeline form "model__<setting>" from model_registry.py.
PARAM_GRIDS = {
    "Logistic Regression": {
        "model__solver": ["liblinear"],
        "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "model__l1_ratio": [0.0, 1.0],
        "model__class_weight": [None, "balanced"],
    },
    "Decision Tree": {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [3, 5, 7, None],
        "model__min_samples_leaf": [1, 5, 10],
    },
    "Gaussian Naive Bayes": {
        "model__var_smoothing": [float(value) for value in np.logspace(-12, 0, 20)],
    },
    "Random Forest": {
        "model__n_estimators": [100, 300],
        "model__max_depth": [3, 5, None],
        "model__min_samples_leaf": [1, 5],
        "model__max_features": ["sqrt", None],
    },
    "k-Nearest Neighbours": {
        "model__n_neighbors": [3, 5, 9, 15, 21, 31],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
}


def load_feature_data(file_path):
    """
    Description:
        Loads the processed thread-level feature dataset.

    Input:
        file_path: Path to the processed feature CSV file.

    Output:
        Returns a pandas DataFrame containing the thread-level features.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Feature file not found: {file_path}")

    return pd.read_csv(file_path)


def load_split_indices(file_path):
    """
    Description:
        Loads the saved train/test split indices from JSON.

    Input:
        file_path: Path to split_indices.json.

    Output:
        Returns a dictionary containing train and test row indices.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Split index file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_inputs(features_df, split_payload):
    """
    Description:
        Checks the required columns, that the training and test rows do not
        overlap, and that a grid exists for every model.

    Input:
        features_df: DataFrame containing processed thread-level features.
        split_payload: Dictionary loaded from split_indices.json.

    Output:
        Raises an error if a check fails. Returns None if all checks pass.
    """
    missing_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(features_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    overlap = set(split_payload["train_row_indices"]) & set(split_payload["test_row_indices"])
    if overlap:
        raise ValueError(f"{len(overlap)} rows appear in both the training and test split")

    missing_grids = set(MODEL_NAMES) - set(PARAM_GRIDS)
    if missing_grids:
        raise ValueError(f"No parameter grid defined for: {missing_grids}")


def get_training_data(features_df, split_payload):
    """
    Description:
        Selects only the training rows using the saved train row indices.

    Input:
        features_df: DataFrame containing all processed feature rows.
        split_payload: Dictionary loaded from split_indices.json.

    Output:
        Returns X_train and y_train.
    """
    train_df = features_df.iloc[split_payload["train_row_indices"]].copy()
    return train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]


def count_combinations(param_grid):
    """
    Description:
        Counts how many setting combinations a grid contains.

    Input:
        param_grid: Dictionary of setting names and candidate values.

    Output:
        Returns the number of combinations as an integer.
    """
    return len(list(product(*param_grid.values())))


def to_json_value(value):
    """
    Description:
        Converts numpy values to plain Python values so they can be saved
        as JSON.

    Input:
        value: Any setting or score value.

    Output:
        Returns a JSON-safe value.
    """
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_grid_search(X_train, y_train):
    """
    Description:
        Runs GridSearchCV for every model using the same stratified folds.
        The best settings are chosen by the primary metric (F1).

    Input:
        X_train: Training feature matrix.
        y_train: Training target labels.

    Output:
        Returns a DataFrame of every grid result and a dictionary of the
        selected settings and scores for each model.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    all_results = []
    selected = {}

    for model_name in MODEL_NAMES:
        param_grid = PARAM_GRIDS[model_name]
        grid_size = count_combinations(param_grid)
        print(f"Tuning {model_name} ({grid_size} combinations)...")

        search = GridSearchCV(
            estimator=build_model(model_name),
            param_grid=param_grid,
            scoring=SCORING,
            refit=PRIMARY_METRIC,
            cv=cv,
            n_jobs=1,
        )
        search.fit(X_train, y_train)

        results_df = pd.DataFrame(search.cv_results_)
        results_df.insert(0, "model", model_name)
        all_results.append(results_df)

        best_index = search.best_index_
        metrics = {}
        for metric in SCORING:
            metrics[f"{metric}_mean"] = float(results_df.loc[best_index, f"mean_test_{metric}"])
            metrics[f"{metric}_std"] = float(results_df.loc[best_index, f"std_test_{metric}"])

        selected[model_name] = {
            "grid_size": grid_size,
            "best_params": {key: to_json_value(value) for key, value in search.best_params_.items()},
            f"best_{PRIMARY_METRIC}": float(search.best_score_),
            "metrics": metrics,
        }

    return pd.concat(all_results, ignore_index=True), selected


def add_untuned_comparison(selected):
    """
    Description:
        Adds each model's untuned F1 (from cv_results_five_models.csv) so the
        effect of tuning can be seen. Skipped if that file does not exist.

    Input:
        selected: Dictionary of selected settings and scores.

    Output:
        Updates the dictionary in place. Returns None.
    """
    if not Path(CV_RESULTS_FIVE_MODELS).exists():
        print("Note: cv_results_five_models.csv not found; skipping untuned comparison.")
        return

    untuned_df = pd.read_csv(CV_RESULTS_FIVE_MODELS).set_index("model")

    for model_name, details in selected.items():
        untuned_f1 = float(untuned_df.loc[model_name, "f1_mean"])
        details["untuned_f1"] = untuned_f1
        details["f1_change_from_tuning"] = details[f"best_{PRIMARY_METRIC}"] - untuned_f1


def save_outputs(results_df, selected):
    """
    Description:
        Saves every grid result to CSV and the selected settings to JSON.

    Input:
        results_df: DataFrame of all grid search results.
        selected: Dictionary of selected settings and scores.

    Output:
        Creates grid_search_results_five_models.csv and
        selected_hyperparams_five_models.json.
    """
    GRID_SEARCH_RESULTS_FIVE_MODELS.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(GRID_SEARCH_RESULTS_FIVE_MODELS, index=False)

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "selection_metric": PRIMARY_METRIC,
        "cv_strategy": f"Stratified {CV_FOLDS}-fold cross-validation, random_state={SEED}",
        "held_out_test_used": False,
        "selected_models": selected,
    }

    with open(SELECTED_HYPERPARAMS_FIVE_MODELS, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)


def print_summary(selected):
    """
    Description:
        Prints the grid size, best F1 and selected settings for each model.

    Input:
        selected: Dictionary of selected settings and scores.

    Output:
        Displays the summary in the terminal.
    """
    print("")
    print("Grid Search Results (training set, stratified 5-fold, selected by F1)")
    print("=" * 70)

    for model_name, details in selected.items():
        line = f"{model_name}: grid {details['grid_size']} | best F1 {details['best_f1']:.4f}"
        if "untuned_f1" in details:
            line += f" (untuned {details['untuned_f1']:.4f}, change {details['f1_change_from_tuning']:+.4f})"
        print(line)

        settings = ", ".join(
            f"{key.replace('model__', '')}={value}" for key, value in details["best_params"].items()
        )
        print(f"    {settings}")

    print("")
    print(f"Saved grid results to: {GRID_SEARCH_RESULTS_FIVE_MODELS}")
    print(f"Saved selected settings to: {SELECTED_HYPERPARAMS_FIVE_MODELS}")


def main():
    """
    Description:
        Runs grid search for all five models on the training split and
        saves the full results and the selected settings.

    Input:
        No direct function input. Reads settings and paths from config.py.

    Output:
        Creates the grid search CSV and selected settings JSON.
    """
    features_df = load_feature_data(PROCESSED_DATA)
    split_payload = load_split_indices(SPLIT_INDICES)

    validate_inputs(features_df, split_payload)

    X_train, y_train = get_training_data(features_df, split_payload)

    results_df, selected = run_grid_search(X_train, y_train)

    add_untuned_comparison(selected)

    save_outputs(results_df, selected)

    print_summary(selected)


if __name__ == "__main__":
    main()