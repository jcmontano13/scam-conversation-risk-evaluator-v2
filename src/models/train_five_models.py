"""
train_five_models.py

Description:
    Trains and compares all five COMP903 models using stratified 5-fold
    cross-validation on the saved training split only. The models come from
    model_registry.py, so they are built exactly the same way in every script.

    Every model uses the same folds, the same four structural features and
    the same metrics, so the three new models (Gaussian Naive Bayes, Random
    Forest, k-Nearest Neighbours) are compared fairly with the COMP902
    baselines (Logistic Regression, Decision Tree).

    As a reproducibility check, the Logistic Regression and Decision Tree
    results are compared with the COMP902 cross-validation results.

Inputs:
    - data/processed/scam_dialogue_thread_features.csv
    - data/processed/split_indices.json

Output:
    - outputs/reports/cv_results_five_models.csv

Run:
    - python src/models/train_five_models.py

Important:
    - Only train_row_indices from split_indices.json are used.
    - The held-out test set is not used in this script.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import (
    CV_FOLDS,
    CV_RESULTS_FIVE_MODELS,
    FEATURE_COLUMNS,
    PROCESSED_DATA,
    SEED,
    SPLIT_INDICES,
    TARGET_COLUMN,
)
from src.models.model_registry import build_models

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

# COMP902 cross-validation results (outputs/reports/cv_results.csv,
# COMP902 commit a282ecf), used to confirm the baselines are reproduced.
COMP902_CV_REFERENCE = {
    "Logistic Regression": {"accuracy_mean": 0.8875, "f1_mean": 0.8841331523157716},
    "Decision Tree": {"accuracy_mean": 0.8703125, "f1_mean": 0.8636752499232723},
}
REFERENCE_TOLERANCE = 1e-9


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
        Checks that the required columns exist and that the training and
        test row indices do not overlap.

    Input:
        features_df: DataFrame containing processed thread-level features.
        split_payload: Dictionary loaded from split_indices.json.

    Output:
        Raises an error if a check fails. Returns None if all checks pass.
    """
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns - set(features_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    overlap = set(split_payload["train_row_indices"]) & set(split_payload["test_row_indices"])

    if overlap:
        raise ValueError(f"{len(overlap)} rows appear in both the training and test split")


def get_training_data(features_df, split_payload):
    """
    Description:
        Selects only the training rows using the saved train row indices.
        The held-out test rows are intentionally not used.

    Input:
        features_df: DataFrame containing all processed feature rows.
        split_payload: Dictionary loaded from split_indices.json.

    Output:
        Returns X_train and y_train for cross-validation.
    """
    train_df = features_df.iloc[split_payload["train_row_indices"]].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    return X_train, y_train


def run_cross_validation(models, X_train, y_train):
    """
    Description:
        Runs stratified 5-fold cross-validation for each model. The same
        folds are used for every model because the splitter has a fixed
        random seed.

    Input:
        models: Dictionary of model names and pipelines from build_models().
        X_train: Training feature matrix.
        y_train: Training target labels.

    Output:
        Returns a DataFrame with the mean and standard deviation of each
        metric for each model.
    """
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    result_rows = []

    for model_name, model in models.items():
        scores = cross_validate(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=SCORING,
            return_train_score=False,
        )

        row = {"model": model_name}

        for metric_name in SCORING:
            score_key = f"test_{metric_name}"
            row[f"{metric_name}_mean"] = scores[score_key].mean()
            row[f"{metric_name}_std"] = scores[score_key].std()

        result_rows.append(row)

    return pd.DataFrame(result_rows)


def check_comp902_reproduction(results_df):
    """
    Description:
        Compares the Logistic Regression and Decision Tree results with the
        COMP902 cross-validation results.

    Input:
        results_df: DataFrame returned by run_cross_validation().

    Output:
        Returns a list of text lines describing each comparison, and a
        boolean that is True if every value matched.
    """
    lines = []
    all_match = True

    for model_name, reference in COMP902_CV_REFERENCE.items():
        row = results_df.loc[results_df["model"] == model_name].iloc[0]

        for metric, expected in reference.items():
            actual = row[metric]
            match = abs(actual - expected) <= REFERENCE_TOLERANCE
            all_match = all_match and match
            lines.append(
                f"{model_name} {metric}: {actual:.6f} "
                f"(COMP902 {expected:.6f}) {'match' if match else 'DIFFERENT'}"
            )

    return lines, all_match


def save_cv_results(results_df, output_path):
    """
    Description:
        Saves cross-validation results to a CSV file.

    Input:
        results_df: DataFrame containing cross-validation summary results.
        output_path: Path where the CSV file will be saved.

    Output:
        Creates or overwrites outputs/reports/cv_results_five_models.csv.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)


def print_cv_summary(results_df, reproduction_lines):
    """
    Description:
        Prints the cross-validation results and the COMP902 reproduction
        check to the terminal.

    Input:
        results_df: DataFrame containing cross-validation summary results.
        reproduction_lines: Text lines from check_comp902_reproduction().

    Output:
        Displays the results in the terminal.
    """
    display_columns = ["model"] + [f"{metric}_mean" for metric in SCORING]

    print("Five-Model Cross-Validation Results (training set, stratified 5-fold)")
    print("=" * 70)
    print(results_df[display_columns].round(4).to_string(index=False))
    print("")
    print("COMP902 reproduction check")
    print("-" * 26)
    for line in reproduction_lines:
        print(line)
    print("")
    print(f"Saved CV results to: {CV_RESULTS_FIVE_MODELS}")


def main():
    """
    Description:
        Runs the five-model cross-validation pipeline. It loads the processed
        features and saved split, selects only the training rows, runs
        stratified 5-fold cross-validation for all five models, checks the
        baselines against COMP902 and saves the results.

    Input:
        No direct function input. Reads settings and paths from config.py.

    Output:
        Creates cv_results_five_models.csv and prints a model comparison.
        Exits with an error if the COMP902 baselines are not reproduced.
    """
    features_df = load_feature_data(PROCESSED_DATA)
    split_payload = load_split_indices(SPLIT_INDICES)

    validate_inputs(features_df, split_payload)

    X_train, y_train = get_training_data(features_df, split_payload)

    models = build_models()

    results_df = run_cross_validation(models, X_train, y_train)

    save_cv_results(results_df, CV_RESULTS_FIVE_MODELS)

    reproduction_lines, all_match = check_comp902_reproduction(results_df)

    print_cv_summary(results_df, reproduction_lines)

    if not all_match:
        sys.exit("COMP902 baseline results were not reproduced. Check the model registry settings.")


if __name__ == "__main__":
    main()