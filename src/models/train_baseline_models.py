"""
train_baseline_models.py

Description:
    Trains and compares baseline machine learning models using only the saved
    training split. The script performs stratified 5-fold cross-validation on
    Logistic Regression and Decision Tree models using the four structural
    features extracted from scam dialogue threads.

Inputs:
    - data/processed/scam_dialogue_thread_features.csv
    - data/processed/split_indices.json

Output:
    - outputs/reports/cv_results.csv

Run:
    - python src/models/train_baseline_models.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import CV_RESULTS, PROCESSED_DATA, SEED, SPLIT_INDICES


FEATURE_COLUMNS = [
    "turn_count",
    "aggregate_num_urls",
    "aggregate_money_terms",
    "ratio_request_messages",
]

TARGET_COLUMN = "thread_label"

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
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


def get_training_data(features_df, split_payload):
    """
    Description:
        Selects only the training rows from the processed feature dataset using
        the saved train row indices. The held-out test row indices are
        intentionally not used in this script.

    Input:
        features_df: DataFrame containing all processed feature rows.
        split_payload: Dictionary loaded from split_indices.json.

    Output:
        Returns X_train and y_train for cross-validation.
    """
    train_indices = split_payload["train_row_indices"]

    train_df = features_df.iloc[train_indices].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    return X_train, y_train


def validate_training_data(features_df):
    """
    Description:
        Validates that required feature and target columns exist before model
        training begins.

    Input:
        features_df: DataFrame containing processed thread-level features.

    Output:
        Raises an error if required columns are missing.
        Returns None if validation passes.
    """
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns - set(features_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def build_models():
    """
    Description:
        Creates the baseline models used for comparison.

    Input:
        No direct function input.

    Output:
        Returns a dictionary of model names and sklearn estimators.
    """
    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        random_state=SEED,
                        max_iter=1000,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=SEED,
        ),
    }

    return models


def run_cross_validation(models, X_train, y_train):
    """
    Description:
        Runs stratified 5-fold cross-validation for each baseline model using
        the training split only.

    Input:
        models: Dictionary of model names and sklearn estimators.
        X_train: Training feature matrix.
        y_train: Training target labels.

    Output:
        Returns a DataFrame containing mean and standard deviation for each
        evaluation metric.
    """
    cv = StratifiedKFold(
        n_splits=5,
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


def save_cv_results(results_df, output_path):
    """
    Description:
        Saves cross-validation results to a CSV file.

    Input:
        results_df: DataFrame containing cross-validation summary results.
        output_path: Path where cv_results.csv will be saved.

    Output:
        Creates or overwrites outputs/reports/cv_results.csv.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)


def print_cv_summary(results_df):
    """
    Description:
        Prints the cross-validation results to the terminal.

    Input:
        results_df: DataFrame containing cross-validation summary results.

    Output:
        Displays model comparison results in the terminal.
    """
    print("Baseline Model Cross-Validation Results")
    print("=" * 45)
    print(results_df.to_string(index=False))
    print("")
    print(f"Saved CV results to: {CV_RESULTS}")


def main():
    """
    Description:
        Runs the baseline model cross-validation pipeline. It loads the processed
        features, loads the saved train/test split indices, selects only the
        training rows, trains baseline models using stratified 5-fold
        cross-validation, and saves the results.

    Input:
        No direct function input.
        Reads PROCESSED_DATA, SPLIT_INDICES, CV_RESULTS, and SEED from config.py.

    Output:
        Creates cv_results.csv and prints a model comparison summary.
    """
    features_df = load_feature_data(PROCESSED_DATA)
    split_payload = load_split_indices(SPLIT_INDICES)

    validate_training_data(features_df)

    X_train, y_train = get_training_data(features_df, split_payload)

    models = build_models()

    results_df = run_cross_validation(models, X_train, y_train)

    save_cv_results(results_df, CV_RESULTS)

    print_cv_summary(results_df)


if __name__ == "__main__":
    main()