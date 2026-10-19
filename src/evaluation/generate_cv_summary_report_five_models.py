"""
generate_cv_summary_report_five_models.py

Description:
    Generates a readable markdown summary of the five-model cross-validation
    results created by train_five_models.py. The report ranks the models by
    F1-score, shows the fold-to-fold spread of every metric, names the best
    model for each metric and compares each model with the reference model
    (Logistic Regression).

    The report only describes the results. Whether differences between
    models are statistically significant is tested later with repeated
    cross-validation and significance tests.

Inputs:
    - outputs/reports/cv_results_five_models.csv

Output:
    - outputs/reports/cv_summary_report_five_models.md

Run:
    - python src/evaluation/generate_cv_summary_report_five_models.py
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import (
    CV_FOLDS,
    CV_RESULTS_FIVE_MODELS,
    CV_SUMMARY_REPORT_FIVE_MODELS,
    FEATURE_COLUMNS,
    MODEL_NAMES,
    NEW_MODELS,
    REFERENCE_MODEL,
)

METRICS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
}


def load_cv_results(file_path):
    """
    Description:
        Loads the five-model cross-validation results.

    Input:
        file_path: Path to cv_results_five_models.csv.

    Output:
        Returns a pandas DataFrame with one row per model.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"CV results not found: {file_path}. Run src/models/train_five_models.py first."
        )

    return pd.read_csv(file_path)


def validate_cv_results(results_df):
    """
    Description:
        Checks that the results contain all five models and every metric
        column.

    Input:
        results_df: DataFrame loaded from cv_results_five_models.csv.

    Output:
        Raises an error if a check fails. Returns None if all checks pass.
    """
    missing_models = set(MODEL_NAMES) - set(results_df["model"])
    if missing_models:
        raise ValueError(f"Missing models in CV results: {missing_models}")

    required_columns = [f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")]
    missing_columns = set(required_columns) - set(results_df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in CV results: {missing_columns}")


def format_score(mean_value, std_value):
    """
    Description:
        Formats a mean score with its standard deviation.

    Input:
        mean_value: Mean score across folds.
        std_value: Standard deviation across folds.

    Output:
        Returns a string such as "0.8841 (+/- 0.0177)".
    """
    return f"{mean_value:.4f} (+/- {std_value:.4f})"


def get_best_model(results_df, metric):
    """
    Description:
        Finds the model with the highest mean score for one metric.

    Input:
        results_df: DataFrame of CV results.
        metric: Metric name, for example "f1".

    Output:
        Returns a tuple of (model name, mean score).
    """
    best_row = results_df.loc[results_df[f"{metric}_mean"].idxmax()]
    return best_row["model"], best_row[f"{metric}_mean"]


def build_results_table(results_df):
    """
    Description:
        Builds the main results table, ranked from highest to lowest F1.

    Input:
        results_df: DataFrame of CV results.

    Output:
        Returns the table as a list of markdown lines.
    """
    ranked_df = results_df.sort_values("f1_mean", ascending=False).reset_index(drop=True)

    header = "| Rank | Model | " + " | ".join(METRICS.values()) + " |"
    divider = "|---:|---|" + "---:|" * len(METRICS)
    lines = [header, divider]

    for rank, row in ranked_df.iterrows():
        scores = [format_score(row[f"{m}_mean"], row[f"{m}_std"]) for m in METRICS]
        lines.append(f"| {rank + 1} | {row['model']} | " + " | ".join(scores) + " |")

    return lines


def build_best_model_lines(results_df):
    """
    Description:
        Lists the best model for each metric.

    Input:
        results_df: DataFrame of CV results.

    Output:
        Returns a list of markdown bullet lines.
    """
    lines = []

    for metric, label in METRICS.items():
        model_name, score = get_best_model(results_df, metric)
        lines.append(f"- Best model by {label}: `{model_name}` ({score:.4f})")

    return lines


def build_reference_comparison(results_df):
    """
    Description:
        Compares the F1, precision and recall of every other model with the
        reference model. The F1 difference is also compared with the
        reference model's fold-to-fold standard deviation, as a simple
        indication of whether the gap is larger than normal variation.

    Input:
        results_df: DataFrame of CV results.

    Output:
        Returns the comparison table as a list of markdown lines.
    """
    reference = results_df.loc[results_df["model"] == REFERENCE_MODEL].iloc[0]
    reference_f1_std = reference["f1_std"]

    lines = [
        f"| Model | F1 difference | Precision difference | Recall difference | "
        f"F1 gap compared with {REFERENCE_MODEL} spread |",
        "|---|---:|---:|---:|---|",
    ]

    for _, row in results_df.iterrows():
        if row["model"] == REFERENCE_MODEL:
            continue

        f1_diff = row["f1_mean"] - reference["f1_mean"]
        precision_diff = row["precision_mean"] - reference["precision_mean"]
        recall_diff = row["recall_mean"] - reference["recall_mean"]
        within_spread = "within one standard deviation" if abs(f1_diff) <= reference_f1_std \
            else "larger than one standard deviation"

        lines.append(
            f"| {row['model']} | {f1_diff:+.4f} | {precision_diff:+.4f} | "
            f"{recall_diff:+.4f} | {within_spread} |"
        )

    return lines


def build_interpretation(results_df):
    """
    Description:
        Writes a short, factual interpretation of the results.

    Input:
        results_df: DataFrame of CV results.

    Output:
        Returns the interpretation as a list of paragraphs.
    """
    f1_model, f1_score = get_best_model(results_df, "f1")
    precision_model, precision_score = get_best_model(results_df, "precision")
    recall_model, recall_score = get_best_model(results_df, "recall")

    f1_range = results_df["f1_mean"].max() - results_df["f1_mean"].min()
    reference_f1_std = results_df.loc[results_df["model"] == REFERENCE_MODEL, "f1_std"].iloc[0]

    new_models_df = results_df[results_df["model"].isin(NEW_MODELS)]
    best_new_model, best_new_f1 = get_best_model(new_models_df, "f1")

    return [
        f"The highest mean F1-score was achieved by {f1_model} ({f1_score:.4f}). "
        f"Across all five models, mean F1 ranged by {f1_range:.4f}, compared with a "
        f"fold-to-fold standard deviation of {reference_f1_std:.4f} for {REFERENCE_MODEL}.",
        f"The best of the new models by F1 was {best_new_model} ({best_new_f1:.4f}).",
        f"The models made different trade-offs between precision and recall. "
        f"{precision_model} had the highest precision ({precision_score:.4f}), meaning its "
        f"scam predictions were most often correct, while {recall_model} had the highest "
        f"recall ({recall_score:.4f}), meaning it missed the fewest scam conversations.",
        "These results use default model settings and a single 5-fold cross-validation. "
        "They describe the baseline comparison only; tuning, repeated cross-validation and "
        "significance tests are needed before concluding that one model is better than another.",
    ]


def build_report(results_df):
    """
    Description:
        Builds the full markdown report.

    Input:
        results_df: DataFrame of CV results.

    Output:
        Returns the report as a single string.
    """
    lines = [
        "# Five-Model Cross-Validation Summary Report",
        "",
        "## Description",
        "",
        "This report summarises the baseline cross-validation results for the COMP903 "
        "comparison of five machine learning models for scam conversation classification.",
        "",
        "The models compared are:",
        "",
        *[f"- {name}{' (reference model)' if name == REFERENCE_MODEL else ''}" for name in MODEL_NAMES],
        "",
        "The models use only the four structural thread-level features:",
        "",
        *[f"- `{feature}`" for feature in FEATURE_COLUMNS],
        "",
        "## Evaluation Method",
        "",
        f"Stratified {CV_FOLDS}-fold cross-validation was performed on the training split only, "
        "using the same folds for every model. All models used their default settings from "
        "`model_registry.py`.",
        "",
        "The held-out test set was not used in this stage. It remains reserved for final model evaluation.",
        "",
        "Scores are shown as the mean across folds with the standard deviation in brackets.",
        "",
        "## Results (ranked by F1)",
        "",
        *build_results_table(results_df),
        "",
        "## Best Model by Metric",
        "",
        *build_best_model_lines(results_df),
        "",
        f"## Comparison with {REFERENCE_MODEL}",
        "",
        f"Differences are each model's mean score minus the {REFERENCE_MODEL} mean score. "
        "A positive value means the model scored higher.",
        "",
        *build_reference_comparison(results_df),
        "",
        "## Interpretation",
        "",
    ]

    for paragraph in build_interpretation(results_df):
        lines += [paragraph, ""]

    lines += [
        "## Reproducibility Note",
        "",
        "These results were generated from `cv_results_five_models.csv`, which was created by "
        "`train_five_models.py`.",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
    ]

    return "\n".join(lines) + "\n"


def save_report(report, output_path):
    """
    Description:
        Saves the markdown report.

    Input:
        report: Report text.
        output_path: Path where the report will be saved.

    Output:
        Creates or overwrites outputs/reports/cv_summary_report_five_models.md.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def main():
    """
    Description:
        Loads the five-model CV results, builds the summary report and
        saves it.

    Input:
        No direct function input. Reads paths and settings from config.py.

    Output:
        Creates cv_summary_report_five_models.md.
    """
    results_df = load_cv_results(CV_RESULTS_FIVE_MODELS)
    validate_cv_results(results_df)

    report = build_report(results_df)
    save_report(report, CV_SUMMARY_REPORT_FIVE_MODELS)

    print(f"Saved CV summary report to: {CV_SUMMARY_REPORT_FIVE_MODELS}")


if __name__ == "__main__":
    main()