"""
tune_baseline_models.py

Description:
    Runs grid search cross-validation for Logistic Regression and Decision Tree
    using only the saved training row indices. This script selects baseline model
    hyperparameters and documents cross-validation variance without using the
    held-out test set.

Inputs:
    - data/processed/scam_dialogue_thread_features.csv
    - data/processed/split_indices.json

Output:
    - outputs/reports/grid_search_results.csv
    - outputs/reports/selected_hyperparams.json
    - outputs/reports/cv_variance_report.md

Run:
    - python src/models/tune_baseline_models.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import (
    CV_VARIANCE_REPORT,
    GRID_SEARCH_RESULTS,
    PROCESSED_DATA,
    SEED,
    SELECTED_HYPERPARAMS,
    SPLIT_INDICES,
)


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
        Returns a pandas DataFrame containing thread-level features.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Feature file not found: {file_path}")

    return pd.read_csv(file_path)


def load_split_indices(file_path):
    """
    Description:
        Loads saved train/test split indices from JSON.

    Input:
        file_path: Path to split_indices.json.

    Output:
        Returns a dictionary containing saved split information.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Split index file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_inputs(features_df, split_payload):
    """
    Description:
        Checks that required feature columns, target column, and training indices
        exist before grid search runs.

    Input:
        features_df: DataFrame containing processed feature rows.
        split_payload: Dictionary loaded from split_indices.json.

    Output:
        Raises an error if required data is missing.
        Returns None if validation passes.
    """
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns - set(features_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if "train_row_indices" not in split_payload:
        raise ValueError("Missing train_row_indices in split_indices.json")


def get_training_data(features_df, split_payload):
    """
    Description:
        Selects only the saved training rows for grid search cross-validation.
        The held-out test rows are intentionally not used.

    Input:
        features_df: DataFrame containing all processed feature rows.
        split_payload: Dictionary containing train_row_indices.

    Output:
        Returns X_train and y_train.
    """
    train_indices = split_payload["train_row_indices"]
    train_df = features_df.iloc[train_indices].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    return X_train, y_train


def build_model_grids():
    """
    Description:
        Builds baseline model pipelines and hyperparameter grids.

    Input:
        No direct function input.

    Output:
        Returns a dictionary containing model estimators and parameter grids.
    """
    logistic_regression = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    random_state=SEED,
                    max_iter=1000,
                ),
            ),
        ]
    )

    decision_tree = DecisionTreeClassifier(random_state=SEED)

    return {
        "Logistic Regression": {
            "estimator": logistic_regression,
            "param_grid": {
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__l1_ratio": [0.0, 1.0],
                "model__class_weight": [None, "balanced"],
            },
        },
        "Decision Tree": {
            "estimator": decision_tree,
            "param_grid": {
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 3, 5, 7, 10],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5],
                "class_weight": [None, "balanced"],
            },
        },
    }


def run_grid_search(model_grids, X_train, y_train):
    """
    Description:
        Runs stratified 5-fold GridSearchCV for each baseline model.

    Input:
        model_grids: Dictionary containing estimators and parameter grids.
        X_train: Training feature matrix.
        y_train: Training target labels.

    Output:
        Returns grid search result rows and selected hyperparameter summaries.
    """
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    all_result_frames = []
    selected_models = {}

    for model_name, config in model_grids.items():
        print(f"Running grid search for {model_name}...")

        grid_search = GridSearchCV(
            estimator=config["estimator"],
            param_grid=config["param_grid"],
            scoring=SCORING,
            refit="f1",
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
        )

        grid_search.fit(X_train, y_train)

        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.insert(0, "model", model_name)
        all_result_frames.append(results_df)

        best_index = grid_search.best_index_
        best_row = results_df.loc[best_index]

        selected_models[model_name] = {
            "best_params": grid_search.best_params_,
            "best_f1": float(grid_search.best_score_),
            "metrics": {
                "accuracy_mean": float(best_row["mean_test_accuracy"]),
                "accuracy_std": float(best_row["std_test_accuracy"]),
                "precision_mean": float(best_row["mean_test_precision"]),
                "precision_std": float(best_row["std_test_precision"]),
                "recall_mean": float(best_row["mean_test_recall"]),
                "recall_std": float(best_row["std_test_recall"]),
                "f1_mean": float(best_row["mean_test_f1"]),
                "f1_std": float(best_row["std_test_f1"]),
                "roc_auc_mean": float(best_row["mean_test_roc_auc"]),
                "roc_auc_std": float(best_row["std_test_roc_auc"]),
            },
        }

    combined_results = pd.concat(all_result_frames, ignore_index=True)

    return combined_results, selected_models


def save_grid_search_results(results_df, output_path):
    """
    Description:
        Saves full GridSearchCV results to CSV.

    Input:
        results_df: DataFrame containing all grid search results.
        output_path: Path where grid_search_results.csv will be saved.

    Output:
        Creates or overwrites grid_search_results.csv.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)


def save_selected_hyperparams(selected_models, output_path):
    """
    Description:
        Saves selected hyperparameters and best cross-validation scores to JSON.

    Input:
        selected_models: Dictionary containing best parameters and metrics.
        output_path: Path where selected_hyperparams.json will be saved.

    Output:
        Creates or overwrites selected_hyperparams.json.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selection_metric": "f1",
        "cv_strategy": "Stratified 5-fold cross-validation",
        "held_out_test_used": False,
        "selected_models": selected_models,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)


def format_metric(mean_value, std_value):
    """
    Description:
        Formats a cross-validation metric with mean and standard deviation.

    Input:
        mean_value: Mean score.
        std_value: Standard deviation score.

    Output:
        Returns a formatted string for Markdown display.
    """
    return f"{mean_value:.4f} (+/- {std_value:.4f})"


def build_variance_report(selected_models):
    """
    Description:
        Builds a Markdown report summarising selected hyperparameters and
        cross-validation variance.

    Input:
        selected_models: Dictionary containing best parameters and metrics.

    Output:
        Returns Markdown report content as a string.
    """
    lines = [
        "# Cross-Validation Variance Report",
        "",
        "## Description",
        "",
        "This report summarises grid search cross-validation results for the baseline models.",
        "",
        "Only the saved training rows were used. The held-out test set was not used during grid search.",
        "",
        "The main selection metric was F1 score.",
        "",
        "## Selected Model Results",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for model_name, details in selected_models.items():
        metrics = details["metrics"]

        lines.append(
            f"| {model_name} | "
            f"{format_metric(metrics['accuracy_mean'], metrics['accuracy_std'])} | "
            f"{format_metric(metrics['precision_mean'], metrics['precision_std'])} | "
            f"{format_metric(metrics['recall_mean'], metrics['recall_std'])} | "
            f"{format_metric(metrics['f1_mean'], metrics['f1_std'])} | "
            f"{format_metric(metrics['roc_auc_mean'], metrics['roc_auc_std'])} |"
        )

    lines.extend(
        [
            "",
            "## Selected Hyperparameters",
            "",
        ]
    )

    for model_name, details in selected_models.items():
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(details["best_params"], indent=4))
        lines.append("```")
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "Lower standard deviation indicates more stable cross-validation performance across folds.",
            "The selected hyperparameters should be compared with earlier baseline results before deciding whether tuning provides meaningful improvement.",
            "",
            f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        ]
    )

    return "\n".join(lines)


def save_variance_report(report, output_path):
    """
    Description:
        Saves the cross-validation variance report as a Markdown file.

    Input:
        report: Markdown report content.
        output_path: Path where cv_variance_report.md will be saved.

    Output:
        Creates or overwrites cv_variance_report.md.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(report)


def print_summary(selected_models):
    """
    Description:
        Prints selected hyperparameters and F1 scores to the terminal.

    Input:
        selected_models: Dictionary containing best parameters and metrics.

    Output:
        Displays grid search summary in the terminal.
    """
    print("")
    print("Grid Search Summary")
    print("=" * 40)

    for model_name, details in selected_models.items():
        print(f"{model_name}")
        print(f"Best F1: {details['best_f1']:.4f}")
        print(f"Best params: {details['best_params']}")
        print("")

    print(f"Saved grid search results to: {GRID_SEARCH_RESULTS}")
    print(f"Saved selected hyperparameters to: {SELECTED_HYPERPARAMS}")
    print(f"Saved CV variance report to: {CV_VARIANCE_REPORT}")


def main():
    """
    Description:
        Runs the full grid search tuning process using training data only.

    Input:
        No direct function input.
        Reads PROCESSED_DATA, SPLIT_INDICES, SEED, GRID_SEARCH_RESULTS,
        SELECTED_HYPERPARAMS, and CV_VARIANCE_REPORT from config.py.

    Output:
        Creates grid_search_results.csv, selected_hyperparams.json, and
        cv_variance_report.md.
    """
    features_df = load_feature_data(PROCESSED_DATA)
    split_payload = load_split_indices(SPLIT_INDICES)

    validate_inputs(features_df, split_payload)

    X_train, y_train = get_training_data(features_df, split_payload)

    model_grids = build_model_grids()

    results_df, selected_models = run_grid_search(model_grids, X_train, y_train)

    save_grid_search_results(results_df, GRID_SEARCH_RESULTS)
    save_selected_hyperparams(selected_models, SELECTED_HYPERPARAMS)

    report = build_variance_report(selected_models)
    save_variance_report(report, CV_VARIANCE_REPORT)

    print_summary(selected_models)


if __name__ == "__main__":
    main()