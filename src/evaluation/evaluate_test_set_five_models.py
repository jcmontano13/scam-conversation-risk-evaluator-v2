"""
evaluate_test_set_five_models.py

Description:
    Final evaluation of the five tuned COMP903 models on the saved held-out
    test set. Each model is built with the settings chosen by grid search,
    trained once on all 1,280 training conversations, and evaluated once on
    the 320 test conversations it has never seen.

    Each new model is then compared with Logistic Regression using McNemar's
    exact test (Dietterich, 1998), which looks only at the test conversations
    where the two models disagree. The three p-values are adjusted with the
    Holm correction.

    The results are reported as the final measurement. They are not used to
    change any model setting.

Inputs:
    - data/processed/scam_dialogue_thread_features.csv
    - data/processed/split_indices.json
    - outputs/reports/selected_hyperparams_five_models.json

Outputs:
    - outputs/reports/test_results_five_models.csv
    - outputs/reports/mcnemar_results.csv
    - outputs/reports/test_predictions_five_models.csv
    - outputs/reports/test_evaluation_report_five_models.md

Run:
    - python src/evaluation/evaluate_test_set_five_models.py

Important:
    - Models are trained on train_row_indices only.
    - test_row_indices are used once, for this final evaluation.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import (
    FEATURE_COLUMNS,
    MCNEMAR_RESULTS,
    MODEL_NAMES,
    NEW_MODELS,
    PROCESSED_DATA,
    REFERENCE_MODEL,
    SELECTED_HYPERPARAMS_FIVE_MODELS,
    SIGNIFICANCE_ALPHA,
    SPLIT_INDICES,
    TARGET_COLUMN,
    TEST_EVALUATION_REPORT_FIVE_MODELS,
    TEST_PREDICTIONS_FIVE_MODELS,
    TEST_RESULTS_FIVE_MODELS,
)
from src.evaluation.significance_tests import holm_correction
from src.models.model_registry import build_models


def load_json(file_path, description):
    """
    Description:
        Loads a JSON file.

    Input:
        file_path: Path to the JSON file.
        description: Short name of the file, used in the error message.

    Output:
        Returns the parsed JSON content.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"{description} not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_train_and_test():
    """
    Description:
        Loads the features and the saved split, checks the split is clean,
        and returns the training and test rows separately.

    Output:
        Returns train_df and test_df.
    """
    if not Path(PROCESSED_DATA).exists():
        raise FileNotFoundError(f"Feature file not found: {PROCESSED_DATA}")

    features_df = pd.read_csv(PROCESSED_DATA)
    split_payload = load_json(SPLIT_INDICES, "Split index file")

    missing_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(features_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    train_rows = split_payload["train_row_indices"]
    test_rows = split_payload["test_row_indices"]

    overlap = set(train_rows) & set(test_rows)
    if overlap:
        raise ValueError(f"{len(overlap)} rows appear in both the training and test split")

    return features_df.iloc[train_rows], features_df.iloc[test_rows]


def load_tuned_models():
    """
    Description:
        Builds the five models with the settings selected by grid search.

    Output:
        Returns a dictionary of model names and untrained tuned pipelines.
    """
    selected = load_json(SELECTED_HYPERPARAMS_FIVE_MODELS, "Selected settings file")["selected_models"]

    missing = set(MODEL_NAMES) - set(selected)
    if missing:
        raise ValueError(f"No tuned settings found for: {missing}. Run tune_five_models.py first.")

    return build_models({name: selected[name]["best_params"] for name in MODEL_NAMES})


def evaluate_models(models, train_df, test_df):
    """
    Description:
        Trains each tuned model on all training rows and scores it once on
        the test rows.

    Input:
        models: Dictionary of model names and tuned pipelines.
        train_df: Training rows.
        test_df: Held-out test rows.

    Output:
        Returns a DataFrame of test metrics (one row per model) and a
        DataFrame of predictions (one row per test conversation).
    """
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    predictions_df = test_df[["thread_id", "type", TARGET_COLUMN]].copy().reset_index(drop=True)
    result_rows = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        result_rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "brier_score": brier_score_loss(y_test, y_prob),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            }
        )

        predictions_df[f"{model_name} | prediction"] = y_pred
        predictions_df[f"{model_name} | scam_probability"] = y_prob

    return pd.DataFrame(result_rows), predictions_df


def run_mcnemar_tests(predictions_df):
    """
    Description:
        Compares each new model with the reference model using McNemar's
        exact test on the test conversations where they disagree, then
        applies the Holm correction.

    Input:
        predictions_df: DataFrame of test predictions for every model.

    Output:
        Returns a DataFrame with one row per new model.
    """
    y_true = predictions_df[TARGET_COLUMN]
    reference_correct = predictions_df[f"{REFERENCE_MODEL} | prediction"] == y_true

    rows = []

    for model_name in NEW_MODELS:
        model_correct = predictions_df[f"{model_name} | prediction"] == y_true

        only_reference_correct = int((reference_correct & ~model_correct).sum())
        only_model_correct = int((~reference_correct & model_correct).sum())
        disagreements = only_reference_correct + only_model_correct

        if disagreements == 0:
            p_value = 1.0
        else:
            p_value = binomtest(only_model_correct, disagreements, 0.5).pvalue

        rows.append(
            {
                "model": model_name,
                "reference_model": REFERENCE_MODEL,
                "both_correct": int((reference_correct & model_correct).sum()),
                "both_wrong": int((~reference_correct & ~model_correct).sum()),
                "only_reference_correct": only_reference_correct,
                "only_model_correct": only_model_correct,
                "p_value": p_value,
            }
        )

    mcnemar_df = pd.DataFrame(rows)
    mcnemar_df["p_value_holm"] = holm_correction(mcnemar_df["p_value"].tolist())
    mcnemar_df["significant"] = mcnemar_df["p_value_holm"] < SIGNIFICANCE_ALPHA

    return mcnemar_df


def build_recall_by_type(predictions_df):
    """
    Description:
        Calculates each model's recall (share of scams caught) for each scam
        type in the test set.

    Input:
        predictions_df: DataFrame of test predictions for every model.

    Output:
        Returns a DataFrame with one row per scam type and one column per model.
    """
    scams_df = predictions_df[predictions_df[TARGET_COLUMN] == 1]
    table = {"scam type": [], "test conversations": []}

    for scam_type, group in scams_df.groupby("type"):
        table["scam type"].append(scam_type)
        table["test conversations"].append(len(group))

    recall_df = pd.DataFrame(table)

    for model_name in MODEL_NAMES:
        recall_df[model_name] = [
            (group[f"{model_name} | prediction"] == 1).mean()
            for _, group in scams_df.groupby("type")
        ]

    return recall_df


def build_report(results_df, mcnemar_df, recall_df, test_size):
    """
    Description:
        Builds the readable markdown report of the final test evaluation.

    Input:
        results_df: Test metrics per model.
        mcnemar_df: McNemar test results.
        recall_df: Recall by scam type.
        test_size: Number of test conversations.

    Output:
        Returns the report as a single string.
    """
    ranked_df = results_df.sort_values("f1", ascending=False)

    lines = [
        "# Held-Out Test Evaluation Report (Five Models)",
        "",
        "## Method",
        "",
        "- Each model used the settings selected by grid search (`selected_hyperparams_five_models.json`).",
        "- Each model was trained once on all 1,280 training conversations.",
        f"- Each model was evaluated once on the {test_size} held-out test conversations.",
        "- These results are the final measurement and were not used to change any setting.",
        "",
        "## Test Results (ranked by F1)",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in ranked_df.iterrows():
        lines.append(
            f"| {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['roc_auc']:.4f} | {row['brier_score']:.4f} |"
        )

    lines += [
        "",
        "## Confusion Matrices",
        "",
        "| Model | True negatives | False positives (false alarms) | False negatives (missed scams) | True positives |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in results_df.iterrows():
        lines.append(
            f"| {row['model']} | {row['true_negative']} | {row['false_positive']} | "
            f"{row['false_negative']} | {row['true_positive']} |"
        )

    lines += [
        "",
        f"## McNemar's Test Against {REFERENCE_MODEL}",
        "",
        "McNemar's exact test uses only the test conversations where the two models disagree. "
        "If one model is right noticeably more often in those cases, the difference is significant. "
        f"p-values are Holm-corrected across the three comparisons (alpha = {SIGNIFICANCE_ALPHA}).",
        "",
        f"| Model | Only {REFERENCE_MODEL} correct | Only this model correct | p-value | Holm p-value | Significant |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for _, row in mcnemar_df.iterrows():
        lines.append(
            f"| {row['model']} | {row['only_reference_correct']} | {row['only_model_correct']} | "
            f"{row['p_value']:.4f} | {row['p_value_holm']:.4f} | {'Yes' if row['significant'] else 'No'} |"
        )

    lines += [
        "",
        "## Recall by Scam Type",
        "",
        "Share of each scam type in the test set that each model correctly flagged as a scam.",
        "",
        "| Scam type | Test conversations | " + " | ".join(MODEL_NAMES) + " |",
        "|---|---:|" + "---:|" * len(MODEL_NAMES),
    ]

    for _, row in recall_df.iterrows():
        values = " | ".join(f"{row[name]:.3f}" for name in MODEL_NAMES)
        lines.append(f"| {row['scam type']} | {row['test conversations']} | {values} |")

    lines += [
        "",
        "## Reproducibility Note",
        "",
        "Generated by `evaluate_test_set_five_models.py`. Per-conversation predictions are saved in "
        "`test_predictions_five_models.csv`.",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
    ]

    return "\n".join(lines) + "\n"


def print_summary(results_df, mcnemar_df):
    """
    Description:
        Prints the test results and McNemar tests to the terminal.

    Input:
        results_df: Test metrics per model.
        mcnemar_df: McNemar test results.

    Output:
        Displays the summary in the terminal.
    """
    print("Held-Out Test Results (tuned models, 320 unseen conversations)")
    print("=" * 64)
    for _, row in results_df.iterrows():
        print(
            f"{row['model']:22s} F1 {row['f1']:.4f} | recall {row['recall']:.4f} | "
            f"precision {row['precision']:.4f} | missed scams {row['false_negative']:>2} | "
            f"false alarms {row['false_positive']:>2}"
        )

    print("")
    print(f"McNemar's test against {REFERENCE_MODEL} (Holm-corrected)")
    print("-" * 52)
    for _, row in mcnemar_df.iterrows():
        print(
            f"{row['model']:22s} disagreements {row['only_reference_correct']} vs {row['only_model_correct']} "
            f"| Holm p {row['p_value_holm']:.4f} | {'SIGNIFICANT' if row['significant'] else 'not significant'}"
        )

    print("")
    print(f"Saved test results to: {TEST_RESULTS_FIVE_MODELS}")
    print(f"Saved report to: {TEST_EVALUATION_REPORT_FIVE_MODELS}")


def main():
    """
    Description:
        Trains the tuned models on the training rows, evaluates them once on
        the held-out test rows, runs McNemar's tests and saves all outputs.

    Input:
        No direct function input. Reads paths and settings from config.py.

    Output:
        Creates the four output files listed at the top of this script.
    """
    train_df, test_df = load_train_and_test()
    models = load_tuned_models()

    results_df, predictions_df = evaluate_models(models, train_df, test_df)
    mcnemar_df = run_mcnemar_tests(predictions_df)
    recall_df = build_recall_by_type(predictions_df)

    TEST_RESULTS_FIVE_MODELS.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(TEST_RESULTS_FIVE_MODELS, index=False)
    mcnemar_df.to_csv(MCNEMAR_RESULTS, index=False)
    predictions_df.to_csv(TEST_PREDICTIONS_FIVE_MODELS, index=False)
    TEST_EVALUATION_REPORT_FIVE_MODELS.write_text(
        build_report(results_df, mcnemar_df, recall_df, len(test_df)), encoding="utf-8"
    )

    print_summary(results_df, mcnemar_df)


if __name__ == "__main__":
    main()