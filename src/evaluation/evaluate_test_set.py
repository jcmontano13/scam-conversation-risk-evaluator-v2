"""
evaluate_test_set.py

Description:
    Trains baseline models using the saved training row indices and evaluates
    them once on the held-out test row indices. This script is used for final
    unbiased baseline evaluation after cross-validation.

Inputs:
    - data/processed/scam_dialogue_thread_features.csv
    - data/processed/split_indices.json

Output:
    - outputs/reports/test_results.csv
    - outputs/reports/test_evaluation_report.md

Run:
    - python src/evaluation/evaluate_test_set.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import (
    PROCESSED_DATA,
    SEED,
    SPLIT_INDICES,
    TEST_EVALUATION_REPORT,
    TEST_RESULTS,
)


FEATURE_COLUMNS = [
    "turn_count",
    "aggregate_num_urls",
    "aggregate_money_terms",
    "ratio_request_messages",
]

TARGET_COLUMN = "thread_label"


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
        Validates that required feature columns, target column, and split index
        keys are available before model evaluation.

    Input:
        features_df: DataFrame containing processed feature rows.
        split_payload: Dictionary loaded from split_indices.json.

    Output:
        Raises an error if required columns or split keys are missing.
        Returns None if validation passes.
    """
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns - set(features_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    required_split_keys = {"train_row_indices", "test_row_indices"}
    missing_split_keys = required_split_keys - set(split_payload.keys())

    if missing_split_keys:
        raise ValueError(f"Missing split index keys: {missing_split_keys}")


def get_train_test_data(features_df, split_payload):
    """
    Description:
        Creates training and held-out test datasets using the saved row indices.

    Input:
        features_df: DataFrame containing all processed feature rows.
        split_payload: Dictionary containing train_row_indices and test_row_indices.

    Output:
        Returns X_train, X_test, y_train, and y_test.
    """
    train_indices = split_payload["train_row_indices"]
    test_indices = split_payload["test_row_indices"]

    train_df = features_df.iloc[train_indices].copy()
    test_df = features_df.iloc[test_indices].copy()

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def build_models():
    """
    Description:
        Creates the baseline models used for held-out test evaluation.

    Input:
        No direct function input.

    Output:
        Returns a dictionary of model names and sklearn estimators.
    """
    return {
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


def get_positive_class_probabilities(model, X_test):
    """
    Description:
        Gets predicted probabilities for the positive scam class.

    Input:
        model: Trained sklearn estimator.
        X_test: Test feature matrix.

    Output:
        Returns probability scores for class 1.
    """
    probabilities = model.predict_proba(X_test)
    classes = list(model.classes_)

    positive_class_index = classes.index(1)

    return probabilities[:, positive_class_index]


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    """
    Description:
        Trains one model on the training set and evaluates it on the held-out
        test set.

    Input:
        model_name: Name of the model being evaluated.
        model: sklearn estimator.
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training labels.
        y_test: Test labels.

    Output:
        Returns a dictionary containing metrics, confusion matrix values, and
        classification report details.
    """
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = get_positive_class_probabilities(model, X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        ),
    }


def run_test_evaluation(models, X_train, X_test, y_train, y_test):
    """
    Description:
        Runs held-out test evaluation for all baseline models.

    Input:
        models: Dictionary of model names and sklearn estimators.
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training labels.
        y_test: Test labels.

    Output:
        Returns a list of evaluation result dictionaries.
    """
    results = []

    for model_name, model in models.items():
        result = evaluate_model(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        results.append(result)

    return results


def build_results_dataframe(results):
    """
    Description:
        Converts model evaluation results into a CSV-friendly DataFrame.

    Input:
        results: List of evaluation result dictionaries.

    Output:
        Returns a pandas DataFrame containing model metrics.
    """
    rows = []

    for result in results:
        rows.append(
            {
                "model": result["model"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "roc_auc": result["roc_auc"],
                "true_negative": result["true_negative"],
                "false_positive": result["false_positive"],
                "false_negative": result["false_negative"],
                "true_positive": result["true_positive"],
            }
        )

    return pd.DataFrame(rows)


def build_results_table(results_df):
    """
    Description:
        Builds a Markdown table from held-out test results.

    Input:
        results_df: DataFrame containing model evaluation metrics.

    Output:
        Returns a Markdown table as a string.
    """
    lines = [
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in results_df.iterrows():
        lines.append(
            f"| {row['model']} | "
            f"{row['accuracy']:.4f} | "
            f"{row['precision']:.4f} | "
            f"{row['recall']:.4f} | "
            f"{row['f1']:.4f} | "
            f"{row['roc_auc']:.4f} | "
            f"{int(row['true_negative'])} | "
            f"{int(row['false_positive'])} | "
            f"{int(row['false_negative'])} | "
            f"{int(row['true_positive'])} |"
        )

    return "\n".join(lines)


def build_report(results_df):
    """
    Description:
        Builds the held-out test evaluation report in Markdown format.

    Input:
        results_df: DataFrame containing held-out test metrics.

    Output:
        Returns the Markdown report as a string.
    """
    best_f1_row = results_df.loc[results_df["f1"].idxmax()]
    best_auc_row = results_df.loc[results_df["roc_auc"].idxmax()]

    return f"""# Held-Out Test Evaluation Report

## Description

This report summarises the final held-out test evaluation for the baseline models.

The models are trained using only `train_row_indices` from `split_indices.json` and evaluated using only `test_row_indices`.

## Features Used

- `turn_count`
- `aggregate_num_urls`
- `aggregate_money_terms`
- `ratio_request_messages`

## Results

{build_results_table(results_df)}

## Best Model Summary

- Best model by F1 score: `{best_f1_row["model"]}` ({best_f1_row["f1"]:.4f})
- Best model by ROC-AUC: `{best_auc_row["model"]}` ({best_auc_row["roc_auc"]:.4f})

## Interpretation

This evaluation uses the held-out test set for final baseline comparison. The cross-validation stage was used earlier for training-set model comparison, while this report provides the final test-set performance.

For scam detection, recall and F1 are important because missed scam cases may be more harmful than false positives. ROC-AUC is also useful because it summarises ranking performance across possible classification thresholds.

Generated at: {datetime.now().isoformat(timespec="seconds")}
"""


def save_results(results_df, output_path):
    """
    Description:
        Saves held-out test metrics to a CSV file.

    Input:
        results_df: DataFrame containing held-out test metrics.
        output_path: Path where test_results.csv will be saved.

    Output:
        Creates or overwrites test_results.csv.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)


def save_report(report, output_path):
    """
    Description:
        Saves the held-out test evaluation report to a Markdown file.

    Input:
        report: Markdown report content as a string.
        output_path: Path where test_evaluation_report.md will be saved.

    Output:
        Creates or overwrites test_evaluation_report.md.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(report)


def print_summary(results_df):
    """
    Description:
        Prints held-out test evaluation results to the terminal.

    Input:
        results_df: DataFrame containing held-out test metrics.

    Output:
        Displays the model comparison summary in the terminal.
    """
    print("Held-Out Test Evaluation Results")
    print("=" * 40)
    print(results_df.to_string(index=False))
    print("")
    print(f"Saved test results to: {TEST_RESULTS}")
    print(f"Saved test evaluation report to: {TEST_EVALUATION_REPORT}")


def main():
    """
    Description:
        Runs the held-out test evaluation pipeline.

    Input:
        No direct function input.
        Reads PROCESSED_DATA, SPLIT_INDICES, TEST_RESULTS,
        TEST_EVALUATION_REPORT, and SEED from config.py.

    Output:
        Creates test_results.csv and test_evaluation_report.md.
    """
    features_df = load_feature_data(PROCESSED_DATA)
    split_payload = load_split_indices(SPLIT_INDICES)

    validate_inputs(features_df, split_payload)

    X_train, X_test, y_train, y_test = get_train_test_data(
        features_df,
        split_payload,
    )

    models = build_models()
    results = run_test_evaluation(models, X_train, X_test, y_train, y_test)

    results_df = build_results_dataframe(results)
    report = build_report(results_df)

    save_results(results_df, TEST_RESULTS)
    save_report(report, TEST_EVALUATION_REPORT)

    print_summary(results_df)


if __name__ == "__main__":
    main()