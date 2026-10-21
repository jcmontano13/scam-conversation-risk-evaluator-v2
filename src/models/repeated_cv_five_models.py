"""
repeated_cv_five_models.py

Description:
    Evaluates the five tuned COMP903 models with repeated stratified
    cross-validation (10 repeats x 5 folds = 50 folds) on the saved training
    split only. Every model is scored on exactly the same 50 folds, so the
    results are paired: for each fold there is one score per model. These
    paired results are the input for the significance tests.

    The folds use REPEATED_CV_SEED rather than SEED. With SEED, the first
    repeat would reproduce the exact folds used by grid search, and the
    models would be scored partly on the folds their settings were chosen on.

Inputs:
    - data/processed/scam_dialogue_thread_features.csv
    - data/processed/split_indices.json
    - outputs/reports/selected_hyperparams_five_models.json

Output:
    - outputs/reports/repeated_cv_results.csv
      (one row per model per fold: 5 models x 50 folds = 250 rows)

Run:
    - python src/models/repeated_cv_five_models.py

Important:
    - Only train_row_indices from split_indices.json are used.
    - The held-out test set is not used in this script.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import (
    CV_FOLDS,
    CV_REPEATS,
    FEATURE_COLUMNS,
    MODEL_NAMES,
    PROCESSED_DATA,
    REPEATED_CV_RESULTS,
    REPEATED_CV_SEED,
    SEED,
    SELECTED_HYPERPARAMS_FIVE_MODELS,
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
    "brier_score": "neg_brier_score",
}


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


def load_training_data():
    """
    Description:
        Loads the features and the saved split, checks them, and returns the
        training rows only.

    Output:
        Returns X_train and y_train.
    """
    if not Path(PROCESSED_DATA).exists():
        raise FileNotFoundError(f"Feature file not found: {PROCESSED_DATA}")

    features_df = pd.read_csv(PROCESSED_DATA)
    split_payload = load_json(SPLIT_INDICES, "Split index file")

    missing_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(features_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    overlap = set(split_payload["train_row_indices"]) & set(split_payload["test_row_indices"])
    if overlap:
        raise ValueError(f"{len(overlap)} rows appear in both the training and test split")

    train_df = features_df.iloc[split_payload["train_row_indices"]]
    return train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]


def load_tuned_models():
    """
    Description:
        Builds the five models with the settings selected by grid search.

    Output:
        Returns a dictionary of model names and untrained tuned pipelines.
    """
    payload = load_json(SELECTED_HYPERPARAMS_FIVE_MODELS, "Selected settings file")
    selected = payload["selected_models"]

    missing = set(MODEL_NAMES) - set(selected)
    if missing:
        raise ValueError(f"No tuned settings found for: {missing}. Run tune_five_models.py first.")

    return build_models({name: selected[name]["best_params"] for name in MODEL_NAMES})


def fold_signature(test_indices):
    """
    Description:
        Creates a short fingerprint of the rows in one validation fold, so it
        can be confirmed that every model was scored on the same folds.

    Input:
        test_indices: Array of row positions in the validation fold.

    Output:
        Returns a 12-character hexadecimal string.
    """
    return hashlib.sha256(np.sort(test_indices).tobytes()).hexdigest()[:12]


def check_folds_differ_from_tuning(splitter, X_train, y_train):
    """
    Description:
        Confirms that none of the 50 evaluation folds is identical to one of
        the 5 folds used by grid search.

    Input:
        splitter: The RepeatedStratifiedKFold used for evaluation.
        X_train: Training feature matrix.
        y_train: Training target labels.

    Output:
        Returns the number of evaluation folds that match a tuning fold.
    """
    tuning_cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    tuning_folds = {fold_signature(test) for _, test in tuning_cv.split(X_train, y_train)}
    evaluation_folds = [fold_signature(test) for _, test in splitter.split(X_train, y_train)]

    return sum(signature in tuning_folds for signature in evaluation_folds)


def run_repeated_cv(models, X_train, y_train):
    """
    Description:
        Scores every tuned model on the same 50 repeated stratified folds.

    Input:
        models: Dictionary of model names and tuned pipelines.
        X_train: Training feature matrix.
        y_train: Training target labels.

    Output:
        Returns a DataFrame with one row per model per fold.
    """
    splitter = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS,
        n_repeats=CV_REPEATS,
        random_state=REPEATED_CV_SEED,
    )

    reused = check_folds_differ_from_tuning(splitter, X_train, y_train)
    if reused:
        raise ValueError(f"{reused} evaluation folds are identical to grid search folds")

    signatures = [fold_signature(test) for _, test in splitter.split(X_train, y_train)]
    rows = []

    for model_name, model in models.items():
        print(f"Evaluating {model_name} on {len(signatures)} folds...")

        scores = cross_validate(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=splitter,
            scoring=SCORING,
            return_train_score=False,
        )

        for fold_index, signature in enumerate(signatures):
            row = {
                "model": model_name,
                "fold_index": fold_index + 1,
                "repeat": fold_index // CV_FOLDS + 1,
                "fold": fold_index % CV_FOLDS + 1,
                "fold_signature": signature,
            }
            for metric in SCORING:
                value = scores[f"test_{metric}"][fold_index]
                # neg_brier_score is negative so that higher is better; store the usual positive value
                row[metric] = -value if metric == "brier_score" else value
            rows.append(row)

    return pd.DataFrame(rows)


def check_pairing(results_df):
    """
    Description:
        Confirms every model has 50 folds and that all models used the same
        fold in each position.

    Input:
        results_df: DataFrame returned by run_repeated_cv().

    Output:
        Raises an error if a check fails. Returns None if all checks pass.
    """
    expected = CV_FOLDS * CV_REPEATS
    counts = results_df.groupby("model").size()

    if not (counts == expected).all():
        raise ValueError(f"Expected {expected} folds per model, got:\n{counts}")

    signatures_per_fold = results_df.groupby("fold_index")["fold_signature"].nunique()
    if not (signatures_per_fold == 1).all():
        raise ValueError("Models were not scored on identical folds")


def print_summary(results_df):
    """
    Description:
        Prints the mean and standard deviation of each metric per model.

    Input:
        results_df: DataFrame of per-fold results.

    Output:
        Displays the summary in the terminal.
    """
    metrics = list(SCORING)
    summary = results_df.groupby("model")[metrics].agg(["mean", "std"])
    summary = summary.reindex(MODEL_NAMES)

    print("")
    print(f"Repeated Cross-Validation ({CV_REPEATS} repeats x {CV_FOLDS} folds, tuned models, training set)")
    print("=" * 78)
    for model_name, row in summary.iterrows():
        print(
            f"{model_name:22s} F1 {row[('f1', 'mean')]:.4f} (+/- {row[('f1', 'std')]:.4f})"
            f" | recall {row[('recall', 'mean')]:.4f} | ROC-AUC {row[('roc_auc', 'mean')]:.4f}"
            f" | Brier {row[('brier_score', 'mean')]:.4f}"
        )
    print("")
    print(f"All {CV_FOLDS * CV_REPEATS} folds identical across models: yes")
    print(f"Saved per-fold results to: {REPEATED_CV_RESULTS}")


def main():
    """
    Description:
        Runs repeated cross-validation for the five tuned models, checks the
        results are paired, and saves one row per model per fold.

    Input:
        No direct function input. Reads settings and paths from config.py.

    Output:
        Creates repeated_cv_results.csv.
    """
    X_train, y_train = load_training_data()
    models = load_tuned_models()

    results_df = run_repeated_cv(models, X_train, y_train)
    check_pairing(results_df)

    REPEATED_CV_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(REPEATED_CV_RESULTS, index=False)

    print_summary(results_df)


if __name__ == "__main__":
    main()