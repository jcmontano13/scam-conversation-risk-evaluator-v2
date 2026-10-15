"""
generate_cv_summary_report.py

Description:
    Generates a readable Markdown summary report from the baseline model
    cross-validation results.

Inputs:
    - outputs/reports/cv_results.csv

Output:
    - outputs/reports/cv_summary_report.md

Run:
    - python src/evaluation/generate_cv_summary_report.py
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import CV_RESULTS, CV_SUMMARY_REPORT


METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]


def load_cv_results(file_path):
    """
    Description:
        Loads the cross-validation results CSV.

    Input:
        file_path: Path to cv_results.csv.

    Output:
        Returns a pandas DataFrame containing cross-validation results.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"CV results file not found: {file_path}")

    return pd.read_csv(file_path)


def validate_cv_results(results_df):
    """
    Description:
        Validates that the expected model and metric columns exist.

    Input:
        results_df: DataFrame containing cross-validation results.

    Output:
        Raises an error if required columns are missing.
        Returns None if validation passes.
    """
    required_columns = {"model"}

    for metric in METRICS:
        required_columns.add(f"{metric}_mean")
        required_columns.add(f"{metric}_std")

    missing_columns = required_columns - set(results_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def format_score(mean_value, std_value):
    """
    Description:
        Formats a metric mean and standard deviation for report display.

    Input:
        mean_value: Mean score from cross-validation.
        std_value: Standard deviation score from cross-validation.

    Output:
        Returns a formatted string.
    """
    return f"{mean_value:.4f} (+/- {std_value:.4f})"


def get_best_model(results_df, metric):
    """
    Description:
        Identifies the model with the highest mean score for a selected metric.

    Input:
        results_df: DataFrame containing cross-validation results.
        metric: Metric name such as f1 or roc_auc.

    Output:
        Returns the best model name and best score.
    """
    metric_column = f"{metric}_mean"
    best_index = results_df[metric_column].idxmax()
    best_row = results_df.loc[best_index]

    return best_row["model"], float(best_row[metric_column])


def build_results_table(results_df):
    """
    Description:
        Builds a Markdown table from the cross-validation results.

    Input:
        results_df: DataFrame containing cross-validation results.

    Output:
        Returns a Markdown table as a string.
    """
    lines = [
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in results_df.iterrows():
        lines.append(
            "| "
            f"{row['model']} | "
            f"{format_score(row['accuracy_mean'], row['accuracy_std'])} | "
            f"{format_score(row['precision_mean'], row['precision_std'])} | "
            f"{format_score(row['recall_mean'], row['recall_std'])} | "
            f"{format_score(row['f1_mean'], row['f1_std'])} | "
            f"{format_score(row['roc_auc_mean'], row['roc_auc_std'])} |"
        )

    return "\n".join(lines)


def build_interpretation(results_df):
    """
    Description:
        Builds a short interpretation of the cross-validation results.

    Input:
        results_df: DataFrame containing cross-validation results.

    Output:
        Returns an interpretation paragraph as a string.
    """
    best_f1_model, best_f1_score = get_best_model(results_df, "f1")
    best_auc_model, best_auc_score = get_best_model(results_df, "roc_auc")

    interpretation = (
        f"The best model by mean F1 score is {best_f1_model} "
        f"with an F1 score of {best_f1_score:.4f}. "
        f"The best model by mean ROC-AUC is {best_auc_model} "
        f"with a ROC-AUC score of {best_auc_score:.4f}. "
        "These cross-validation results suggest which baseline model performs "
        "better on the training split before final held-out test evaluation. "
        "For scam detection, recall and F1 are especially important because "
        "missed scam cases may be more harmful than false alarms."
    )

    return interpretation


def build_report(results_df):
    """
    Description:
        Builds the full Markdown cross-validation summary report.

    Input:
        results_df: DataFrame containing cross-validation results.

    Output:
        Returns the full Markdown report as a string.
    """
    best_f1_model, best_f1_score = get_best_model(results_df, "f1")
    best_auc_model, best_auc_score = get_best_model(results_df, "roc_auc")

    report = f"""# Cross-Validation Summary Report

## Description

This report summarises the baseline model cross-validation results for the Scam Conversation Risk Evaluator project.

The models compared are:

- Logistic Regression
- Decision Tree

The models use only the four structural thread-level features:

- `turn_count`
- `aggregate_num_urls`
- `aggregate_money_terms`
- `ratio_request_messages`

## Evaluation Method

Stratified 5-fold cross-validation was performed using the training split only.

The held-out test set was not used in this stage. It remains reserved for final model evaluation.

## Results

{build_results_table(results_df)}

## Best Model Summary

- Best model by F1 score: `{best_f1_model}` ({best_f1_score:.4f})
- Best model by ROC-AUC: `{best_auc_model}` ({best_auc_score:.4f})

## Interpretation

{build_interpretation(results_df)}

## Reproducibility Note

These results were generated from `cv_results.csv`, which was created by `train_baseline_models.py`.

Generated at: {datetime.now().isoformat(timespec="seconds")}
"""

    return report


def save_report(report, output_path):
    """
    Description:
        Saves the Markdown report to disk.

    Input:
        report: Markdown report content as a string.
        output_path: Path where the report will be saved.

    Output:
        Creates or overwrites cv_summary_report.md.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(report)


def main():
    """
    Description:
        Runs the full cross-validation report generation process.

    Input:
        No direct function input.
        Reads CV_RESULTS and CV_SUMMARY_REPORT from config.py.

    Output:
        Creates cv_summary_report.md and prints the saved report path.
    """
    results_df = load_cv_results(CV_RESULTS)

    validate_cv_results(results_df)

    report = build_report(results_df)

    save_report(report, CV_SUMMARY_REPORT)

    print(f"Saved CV summary report to: {CV_SUMMARY_REPORT}")


if __name__ == "__main__":
    main()