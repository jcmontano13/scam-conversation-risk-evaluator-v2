"""
Generate Threshold Trade-Off Curve

Description:
    This script generates a threshold trade-off curve for the selected
    Logistic Regression model using calibrated probability scores.

    The curve shows how precision, recall, and F1-score change when the
    scam probability threshold is adjusted.

    The script uses the saved training split only and creates out-of-fold
    calibrated probability predictions. The held-out test set is not used
    for threshold exploration.

    This script does not introduce a new dataset, new features, or new models.
    It is only used to generate final report evidence.

Input:
    - data/processed/scam_dialogue_thread_features.csv
    - data/processed/split_indices.json
    - outputs/reports/selected_hyperparams.json, if available

Output:
    - outputs/figures/threshold_tradeoff_curve.png
    - outputs/figures/threshold_metrics.csv
    - outputs/figures/threshold_tradeoff_summary.txt
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Allow script to import config.py from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import PROCESSED_DATA, SEED, SPLIT_INDICES  # noqa: E402


FEATURE_COLUMNS = [
    "turn_count",
    "aggregate_num_urls",
    "aggregate_money_terms",
    "ratio_request_messages",
]

LABEL_COLUMN = "thread_label"

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

SELECTED_HYPERPARAMS_PATH = REPORT_DIR / "selected_hyperparams.json"

THRESHOLD_FIGURE_PATH = FIGURE_DIR / "threshold_tradeoff_curve.png"
THRESHOLD_METRICS_PATH = FIGURE_DIR / "threshold_metrics.csv"
THRESHOLD_SUMMARY_PATH = FIGURE_DIR / "threshold_tradeoff_summary.txt"


def load_split_indices():
    """
    Description:
        Load the saved train/test split indices.

    Input:
        split_indices.json

    Output:
        train_row_indices and test_row_indices lists
    """
    with open(SPLIT_INDICES, "r", encoding="utf-8") as file:
        split_payload = json.load(file)

    train_indices = split_payload.get("train_row_indices")
    test_indices = split_payload.get("test_row_indices")

    if train_indices is None or test_indices is None:
        raise KeyError(
            "Expected split_indices.json to contain "
            "'train_row_indices' and 'test_row_indices'."
        )

    return train_indices, test_indices


def load_selected_hyperparams():
    """
    Description:
        Load selected hyperparameters if the file exists.

    Output:
        Dictionary containing selected hyperparameters, or empty dictionary.
    """
    if not SELECTED_HYPERPARAMS_PATH.exists():
        print("selected_hyperparams.json not found. Using default Logistic Regression settings.")
        return {}

    with open(SELECTED_HYPERPARAMS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_logistic_regression_params(payload):
    """
    Description:
        Extract Logistic Regression parameters from selected_hyperparams.json
        in a flexible way.

    Input:
        payload: selected hyperparameter JSON payload

    Output:
        Dictionary of Logistic Regression best parameters
    """
    model_payload = payload.get("Logistic Regression", {})

    if isinstance(model_payload, dict):
        return (
            model_payload.get("best_params")
            or model_payload.get("selected_params")
            or model_payload.get("params")
            or {}
        )

    return {}


def build_logistic_regression_model():
    """
    Description:
        Build the existing Logistic Regression model using selected
        hyperparameters when available.

    Output:
        Logistic Regression pipeline
    """
    selected_payload = load_selected_hyperparams()
    selected_params = get_logistic_regression_params(selected_payload)

    lr_params = {
        "max_iter": 5000,
        "random_state": SEED,
    }

    if "model__C" in selected_params:
        lr_params["C"] = selected_params["model__C"]

    if "model__class_weight" in selected_params:
        lr_params["class_weight"] = selected_params["model__class_weight"]

    if "model__l1_ratio" in selected_params:
        lr_params["l1_ratio"] = selected_params["model__l1_ratio"]
        lr_params["penalty"] = "elasticnet"
        lr_params["solver"] = "saga"

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**lr_params)),
        ]
    )


def build_calibrated_classifier(base_model, method, inner_cv):
    """
    Description:
        Build calibrated classifier with compatibility for different sklearn versions.

    Input:
        base_model: Base estimator
        method: Calibration method
        inner_cv: Inner cross-validation object

    Output:
        CalibratedClassifierCV object
    """
    try:
        return CalibratedClassifierCV(
            estimator=base_model,
            method=method,
            cv=inner_cv,
        )
    except TypeError:
        return CalibratedClassifierCV(
            base_estimator=base_model,
            method=method,
            cv=inner_cv,
        )


def generate_out_of_fold_probabilities(X_train, y_train):
    """
    Description:
        Generate out-of-fold calibrated probability scores using Logistic
        Regression with isotonic calibration.

    Input:
        X_train: Training feature matrix
        y_train: Training labels

    Output:
        Out-of-fold predicted scam probabilities
    """
    base_model = build_logistic_regression_model()

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    out_of_fold_probabilities = np.zeros(len(X_train))

    for fold_number, (fit_index, validation_index) in enumerate(
        outer_cv.split(X_train, y_train),
        start=1,
    ):
        print(f"Generating calibrated probabilities for fold {fold_number}...")

        X_fit = X_train.iloc[fit_index]
        y_fit = y_train.iloc[fit_index]
        X_validation = X_train.iloc[validation_index]

        inner_cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=SEED,
        )

        calibrated_model = build_calibrated_classifier(
            base_model=clone(base_model),
            method="isotonic",
            inner_cv=inner_cv,
        )

        calibrated_model.fit(X_fit, y_fit)

        validation_probabilities = calibrated_model.predict_proba(X_validation)[:, 1]
        out_of_fold_probabilities[validation_index] = validation_probabilities

    return out_of_fold_probabilities


def calculate_threshold_metrics(y_true, y_probability):
    """
    Description:
        Calculate precision, recall, and F1-score across probability thresholds.

    Input:
        y_true: Actual labels
        y_probability: Predicted scam probabilities

    Output:
        DataFrame containing threshold metrics
    """
    records = []

    thresholds = np.round(np.arange(0.00, 1.01, 0.01), 2)

    for threshold in thresholds:
        y_pred = (y_probability >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        records.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

    return pd.DataFrame(records)


def main():
    """
    Description:
        Generate threshold trade-off curve, threshold metrics CSV,
        and summary file.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading processed feature data...")
    feature_df = pd.read_csv(PROCESSED_DATA)

    print("Loading saved train/test split...")
    train_indices, test_indices = load_split_indices()

    missing_features = [col for col in FEATURE_COLUMNS if col not in feature_df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    if LABEL_COLUMN not in feature_df.columns:
        raise ValueError(f"Missing required label column: {LABEL_COLUMN}")

    train_df = feature_df.iloc[train_indices].copy().reset_index(drop=True)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[LABEL_COLUMN]

    print(f"Training rows used for threshold trade-off curve: {len(train_df)}")
    print(f"Held-out test rows not used: {len(test_indices)}")

    print("Generating out-of-fold calibrated probabilities...")
    y_probability = generate_out_of_fold_probabilities(X_train, y_train)

    print("Calculating threshold metrics...")
    threshold_metrics_df = calculate_threshold_metrics(y_train, y_probability)

    best_f1_row = threshold_metrics_df.loc[
        threshold_metrics_df["f1_score"].idxmax()
    ]

    print("Saving threshold metrics CSV...")
    threshold_metrics_df.to_csv(THRESHOLD_METRICS_PATH, index=False)

    print("Saving threshold trade-off curve figure...")
    plt.figure(figsize=(8, 6))

    plt.plot(
        threshold_metrics_df["threshold"],
        threshold_metrics_df["precision"],
        linewidth=2,
        label="Precision",
    )

    plt.plot(
        threshold_metrics_df["threshold"],
        threshold_metrics_df["recall"],
        linewidth=2,
        label="Recall",
    )

    plt.plot(
        threshold_metrics_df["threshold"],
        threshold_metrics_df["f1_score"],
        linewidth=2,
        label="F1-score",
    )

    plt.axvline(
        x=best_f1_row["threshold"],
        linestyle="--",
        linewidth=1,
        label=f"Best F1 threshold = {best_f1_row['threshold']:.2f}",
    )

    plt.title("Threshold Trade-Off Curve")
    plt.xlabel("Scam Probability Threshold")
    plt.ylabel("Metric Score")
    plt.xlim(0, 1.00)
    plt.ylim(0, 1.02)
    plt.legend(loc="lower center")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    plt.savefig(THRESHOLD_FIGURE_PATH, dpi=300)
    plt.close()

    print("Saving threshold trade-off summary...")
    with open(THRESHOLD_SUMMARY_PATH, "w", encoding="utf-8") as file:
        file.write("Threshold Trade-Off Curve Summary\n")
        file.write("=================================\n\n")
        file.write("Dataset: Training split only\n")
        file.write("Model: Logistic Regression\n")
        file.write("Calibration method: Isotonic\n")
        file.write("Evaluation method: 5-fold out-of-fold calibrated probabilities\n")
        file.write("Inner calibration CV: 3-fold\n")
        file.write(f"Training rows: {len(train_df)}\n")
        file.write(f"Held-out test rows used: 0 of {len(test_indices)}\n")
        file.write(f"Features used: {', '.join(FEATURE_COLUMNS)}\n\n")
        file.write("Best threshold by F1-score:\n")
        file.write(f"Threshold: {best_f1_row['threshold']:.2f}\n")
        file.write(f"Precision: {best_f1_row['precision']:.4f}\n")
        file.write(f"Recall: {best_f1_row['recall']:.4f}\n")
        file.write(f"F1-score: {best_f1_row['f1_score']:.4f}\n\n")
        file.write(
            "Interpretation:\n"
            "The threshold trade-off curve shows how precision, recall, and F1-score "
            "change as the scam probability threshold is adjusted. Lower thresholds "
            "usually increase recall because more scam conversations are detected, "
            "but this may reduce precision by increasing false positives. Higher "
            "thresholds usually improve precision but may reduce recall by missing "
            "more scam conversations.\n\n"
        )
        file.write(
            "Note:\n"
            "This output is for final report evidence and threshold behaviour analysis. "
            "It does not introduce new data, new features, or new model scope.\n"
        )

    print("Threshold trade-off curve generation completed.")
    print(f"Figure saved to: {THRESHOLD_FIGURE_PATH}")
    print(f"Threshold metrics saved to: {THRESHOLD_METRICS_PATH}")
    print(f"Summary saved to: {THRESHOLD_SUMMARY_PATH}")


if __name__ == "__main__":
    main()