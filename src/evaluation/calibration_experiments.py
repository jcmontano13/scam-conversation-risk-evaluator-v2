"""
calibration_experiments.py

Description:
    Runs probability calibration experiments for the scam conversation
    classification project.

    This script compares sigmoid/Platt calibration and isotonic calibration
    for Logistic Regression and Decision Tree models. The purpose is to check
    whether predicted scam probabilities are reliable enough to support
    operational risk thresholds.

Inputs:
    - data/processed/scam_dialogue_thread_features.csv
    - data/processed/split_indices.json
    - outputs/reports/selected_hyperparams.json, if available

Outputs:
    - outputs/reports/calibration_results.csv
    - outputs/reports/calibration_report.md
    - outputs/reports/proposed_risk_thresholds.csv

Run:
    - python src/evaluation/calibration_experiments.py

Important:
    - Only train_row_indices from split_indices.json are used.
    - The held-out test set is not used during calibration experiments.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------
# Project import setup
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config  # noqa: E402


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

FEATURE_COLUMNS = [
    "turn_count",
    "aggregate_num_urls",
    "aggregate_money_terms",
    "ratio_request_messages",
]

TARGET_COLUMN = "thread_label"

N_OUTER_SPLITS = 5
N_INNER_SPLITS = 3

CALIBRATION_METHODS = ["sigmoid", "isotonic"]


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------

def get_output_path(config_name: str, fallback_filename: str) -> Path:
    """
    Description:
        Reads an output path from config.py if it exists. If it does not
        exist yet, a fallback path inside outputs/reports is used.

    Input:
        config_name: Name of the config variable.
        fallback_filename: Default filename to use if config variable is missing.

    Output:
        Path object for the target output file.
    """
    return getattr(
        config,
        config_name,
        PROJECT_ROOT / "outputs" / "reports" / fallback_filename,
    )


CALIBRATION_RESULTS = get_output_path(
    "CALIBRATION_RESULTS",
    "calibration_results.csv",
)

CALIBRATION_REPORT = get_output_path(
    "CALIBRATION_REPORT",
    "calibration_report.md",
)

PROPOSED_RISK_THRESHOLDS = get_output_path(
    "PROPOSED_RISK_THRESHOLDS",
    "proposed_risk_thresholds.csv",
)

SELECTED_HYPERPARAMS = getattr(
    config,
    "SELECTED_HYPERPARAMS",
    PROJECT_ROOT / "outputs" / "reports" / "selected_hyperparams.json",
)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_feature_data() -> pd.DataFrame:
    """
    Description:
        Loads the processed thread-level feature dataset.

    Input:
        None. Uses PROCESSED_DATA from config.py.

    Output:
        Pandas DataFrame containing thread-level features and labels.
    """
    feature_path = Path(config.PROCESSED_DATA)

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    data = pd.read_csv(feature_path)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return data


def load_train_indices() -> List[int]:
    """
    Description:
        Loads the saved training row indices from split_indices.json.

    Input:
        None. Uses SPLIT_INDICES from config.py.

    Output:
        List of training row indices.
    """
    split_path = Path(config.SPLIT_INDICES)

    if not split_path.exists():
        raise FileNotFoundError(f"Split indices file not found: {split_path}")

    with split_path.open("r", encoding="utf-8") as file:
        split_payload = json.load(file)

    if "train_row_indices" not in split_payload:
        raise KeyError("Expected key 'train_row_indices' was not found.")

    return split_payload["train_row_indices"]


def load_selected_hyperparams() -> Dict:
    """
    Description:
        Loads selected hyperparameters if the grid search output exists.
        If not available, safe baseline defaults are returned.

    Input:
        None. Uses selected_hyperparams.json if available.

    Output:
        Dictionary of selected model hyperparameters.
    """
    if not Path(SELECTED_HYPERPARAMS).exists():
        return {}

    with Path(SELECTED_HYPERPARAMS).open("r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------

def build_logistic_regression_model(selected_params: Dict) -> Pipeline:
    """
    Description:
        Builds the Logistic Regression pipeline using selected
        hyperparameters if available.

    Input:
        selected_params: Dictionary loaded from selected_hyperparams.json.

    Output:
        Scikit-learn Pipeline containing StandardScaler and LogisticRegression.
    """
    lr_defaults = {
        "C": 1.0,
        "class_weight": None,
        "l1_ratio": None,
    }

    lr_params = (
        selected_params
        .get("selected_models", {})
        .get("Logistic Regression", {})
        .get("best_params", {})
    )

    c_value = lr_params.get("model__C", lr_defaults["C"])
    class_weight = lr_params.get("model__class_weight", lr_defaults["class_weight"])
    l1_ratio = lr_params.get("model__l1_ratio", lr_defaults["l1_ratio"])

    logistic_model = LogisticRegression(
        C=c_value,
        class_weight=class_weight,
        l1_ratio=l1_ratio,
        solver="liblinear",
        max_iter=1000,
        random_state=config.SEED,
    )

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", logistic_model),
        ]
    )


def build_decision_tree_model(selected_params: Dict) -> DecisionTreeClassifier:
    """
    Description:
        Builds the Decision Tree model using selected hyperparameters
        if available.

    Input:
        selected_params: Dictionary loaded from selected_hyperparams.json.

    Output:
        DecisionTreeClassifier model.
    """
    tree_params = (
        selected_params
        .get("selected_models", {})
        .get("Decision Tree", {})
        .get("best_params", {})
    )

    return DecisionTreeClassifier(
        criterion=tree_params.get("criterion", "gini"),
        max_depth=tree_params.get("max_depth", None),
        min_samples_leaf=tree_params.get("min_samples_leaf", 1),
        min_samples_split=tree_params.get("min_samples_split", 2),
        class_weight=tree_params.get("class_weight", None),
        random_state=config.SEED,
    )


def build_models(selected_params: Dict) -> Dict:
    """
    Description:
        Builds all models used in calibration experiments.

    Input:
        selected_params: Dictionary loaded from selected_hyperparams.json.

    Output:
        Dictionary of model names and model objects.
    """
    return {
        "Logistic Regression": build_logistic_regression_model(selected_params),
        "Decision Tree": build_decision_tree_model(selected_params),
    }


# ---------------------------------------------------------------------
# Calibration compatibility helper
# ---------------------------------------------------------------------

def build_calibrated_classifier(base_model, method: str, cv):
    """
    Description:
        Builds a CalibratedClassifierCV object. This helper supports
        different scikit-learn versions by trying estimator first and
        then falling back to base_estimator if needed.

    Input:
        base_model: Base classifier to calibrate.
        method: Calibration method, either sigmoid or isotonic.
        cv: Cross-validation strategy for calibration.

    Output:
        CalibratedClassifierCV object.
    """
    try:
        return CalibratedClassifierCV(
            estimator=base_model,
            method=method,
            cv=cv,
        )
    except TypeError:
        return CalibratedClassifierCV(
            base_estimator=base_model,
            method=method,
            cv=cv,
        )


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def calculate_metrics(y_true, y_probability, threshold: float = 0.5) -> Dict:
    """
    Description:
        Calculates classification and calibration metrics.

    Input:
        y_true: True labels.
        y_probability: Predicted scam probabilities.
        threshold: Classification threshold.

    Output:
        Dictionary of metric values.
    """
    y_pred = (y_probability >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probability),
        "brier_score": brier_score_loss(y_true, y_probability),
    }


def run_calibration_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    base_model,
    method: str,
) -> Tuple[List[Dict], np.ndarray]:
    """
    Description:
        Runs nested calibration evaluation using outer stratified folds.
        The outer fold evaluates performance, while the inner fold is used
        inside CalibratedClassifierCV for probability calibration.

    Input:
        X: Feature matrix from training rows only.
        y: Target labels from training rows only.
        model_name: Name of the base model.
        base_model: Scikit-learn classifier.
        method: Calibration method.

    Output:
        Tuple containing:
        - List of per-fold metric dictionaries.
        - Out-of-fold predicted probabilities.
    """
    outer_cv = StratifiedKFold(
        n_splits=N_OUTER_SPLITS,
        shuffle=True,
        random_state=config.SEED,
    )

    fold_results = []
    out_of_fold_probabilities = np.zeros(len(y))

    for fold_number, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train_fold = X.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]
        y_train_fold = y.iloc[train_idx]
        y_valid_fold = y.iloc[valid_idx]

        inner_cv = StratifiedKFold(
            n_splits=N_INNER_SPLITS,
            shuffle=True,
            random_state=config.SEED + fold_number,
        )

        calibrated_model = build_calibrated_classifier(
            base_model=clone(base_model),
            method=method,
            cv=inner_cv,
        )

        calibrated_model.fit(X_train_fold, y_train_fold)

        y_probability = calibrated_model.predict_proba(X_valid_fold)[:, 1]
        out_of_fold_probabilities[valid_idx] = y_probability

        metrics = calculate_metrics(y_valid_fold, y_probability)
        metrics.update(
            {
                "model": model_name,
                "calibration_method": method,
                "fold": fold_number,
            }
        )

        fold_results.append(metrics)

    return fold_results, out_of_fold_probabilities


def summarize_results(fold_results: List[Dict]) -> pd.DataFrame:
    """
    Description:
        Converts fold-level calibration results into a summary table.

    Input:
        fold_results: List of fold-level metric dictionaries.

    Output:
        Summary DataFrame with mean and standard deviation metrics.
    """
    results_df = pd.DataFrame(fold_results)

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "brier_score",
    ]

    summary_rows = []

    grouped = results_df.groupby(["model", "calibration_method"])

    for (model_name, method), group in grouped:
        row = {
            "model": model_name,
            "calibration_method": method,
        }

        for metric in metric_columns:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std()

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------

def find_threshold_for_precision(
    y_true: pd.Series,
    y_probability: np.ndarray,
    target_precision: float = 0.90,
) -> Dict:
    """
    Description:
        Finds a threshold that reaches the target precision where possible.

    Input:
        y_true: True labels.
        y_probability: Predicted probabilities.
        target_precision: Minimum desired precision.

    Output:
        Dictionary containing selected threshold and related metrics.
    """
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    candidates = []

    for threshold in thresholds:
        y_pred = (y_probability >= threshold).astype(int)

        if y_pred.sum() == 0:
            continue

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if precision >= target_precision:
            candidates.append(
                {
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "basis": f"precision >= {target_precision}",
                }
            )

    if candidates:
        return max(candidates, key=lambda item: (item["f1"], item["recall"]))

    return find_threshold_for_best_f1(y_true, y_probability)


def find_threshold_for_recall(
    y_true: pd.Series,
    y_probability: np.ndarray,
    target_recall: float = 0.90,
) -> Dict:
    """
    Description:
        Finds a threshold that reaches the target recall where possible.

    Input:
        y_true: True labels.
        y_probability: Predicted probabilities.
        target_recall: Minimum desired recall.

    Output:
        Dictionary containing selected threshold and related metrics.
    """
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    candidates = []

    for threshold in thresholds:
        y_pred = (y_probability >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if recall >= target_recall:
            candidates.append(
                {
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "basis": f"recall >= {target_recall}",
                }
            )

    if candidates:
        return max(candidates, key=lambda item: (item["f1"], item["precision"]))

    return find_threshold_for_best_f1(y_true, y_probability)


def find_threshold_for_best_f1(
    y_true: pd.Series,
    y_probability: np.ndarray,
) -> Dict:
    """
    Description:
        Finds the threshold with the best F1 score.

    Input:
        y_true: True labels.
        y_probability: Predicted probabilities.

    Output:
        Dictionary containing selected threshold and related metrics.
    """
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    candidates = []

    for threshold in thresholds:
        y_pred = (y_probability >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        candidates.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "basis": "best F1 fallback",
            }
        )

    return max(candidates, key=lambda item: item["f1"])


def create_risk_thresholds(
    y_true: pd.Series,
    y_probability: np.ndarray,
    selected_model: str,
    selected_method: str,
) -> pd.DataFrame:
    """
    Description:
        Creates proposed Low, Medium, and High scam-risk thresholds.

    Input:
        y_true: True labels from training rows only.
        y_probability: Out-of-fold predicted probabilities.
        selected_model: Selected model name.
        selected_method: Selected calibration method.

    Output:
        DataFrame containing proposed risk thresholds.
    """
    medium_threshold_info = find_threshold_for_recall(
        y_true=y_true,
        y_probability=y_probability,
        target_recall=0.90,
    )

    high_threshold_info = find_threshold_for_precision(
        y_true=y_true,
        y_probability=y_probability,
        target_precision=0.90,
    )

    medium_threshold = float(medium_threshold_info["threshold"])
    high_threshold = float(high_threshold_info["threshold"])

    if medium_threshold >= high_threshold:
        medium_threshold = 0.33
        high_threshold = 0.66
        threshold_basis = (
            "fallback fixed bands because validation-derived thresholds overlapped"
        )
    else:
        threshold_basis = (
            f"medium threshold based on {medium_threshold_info['basis']}; "
            f"high threshold based on {high_threshold_info['basis']}"
        )

    threshold_rows = [
        {
            "risk_band": "Low",
            "probability_min": 0.00,
            "probability_max_exclusive": medium_threshold,
            "recommended_action": "Low scam-risk score; no immediate alert, but still keep record for analysis.",
            "selected_model": selected_model,
            "calibration_method": selected_method,
            "threshold_basis": threshold_basis,
        },
        {
            "risk_band": "Medium",
            "probability_min": medium_threshold,
            "probability_max_exclusive": high_threshold,
            "recommended_action": "Medium scam-risk score; conversation should be reviewed or monitored.",
            "selected_model": selected_model,
            "calibration_method": selected_method,
            "threshold_basis": threshold_basis,
        },
        {
            "risk_band": "High",
            "probability_min": high_threshold,
            "probability_max_exclusive": 1.01,
            "recommended_action": "High scam-risk score; prioritize this conversation for warning or further review.",
            "selected_model": selected_model,
            "calibration_method": selected_method,
            "threshold_basis": threshold_basis,
        },
    ]

    return pd.DataFrame(threshold_rows)


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def format_metric(value: float) -> str:
    """
    Description:
        Formats metric values for Markdown output.

    Input:
        value: Numeric metric value.

    Output:
        Formatted string.
    """
    return f"{value:.4f}"


def write_calibration_report(
    summary_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    selected_model: str,
    selected_method: str,
) -> None:
    """
    Description:
        Writes a Markdown report summarising calibration results.

    Input:
        summary_df: Calibration summary results.
        thresholds_df: Proposed risk thresholds.
        selected_model: Selected model name.
        selected_method: Selected calibration method.

    Output:
        Saves calibration_report.md.
    """
    CALIBRATION_REPORT.parent.mkdir(parents=True, exist_ok=True)

    report_lines = []

    report_lines.append("# Calibration Report")
    report_lines.append("")
    report_lines.append("## Description")
    report_lines.append("")
    report_lines.append(
        "This report summarises probability calibration experiments for the "
        "scam conversation classification project."
    )
    report_lines.append("")
    report_lines.append(
        "Only the saved training rows from `split_indices.json` were used. "
        "The held-out test set was not used during calibration experiments."
    )
    report_lines.append("")
    report_lines.append("## Calibration Methods")
    report_lines.append("")
    report_lines.append("- Sigmoid calibration, also known as Platt scaling")
    report_lines.append("- Isotonic calibration")
    report_lines.append("")
    report_lines.append("## Summary Results")
    report_lines.append("")
    report_lines.append(
        "| Model | Method | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier Score |"
    )
    report_lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|"
    )

    for _, row in summary_df.iterrows():
        report_lines.append(
            "| "
            f"{row['model']} | "
            f"{row['calibration_method']} | "
            f"{format_metric(row['accuracy_mean'])} | "
            f"{format_metric(row['precision_mean'])} | "
            f"{format_metric(row['recall_mean'])} | "
            f"{format_metric(row['f1_mean'])} | "
            f"{format_metric(row['roc_auc_mean'])} | "
            f"{format_metric(row['brier_score_mean'])} |"
        )

    report_lines.append("")
    report_lines.append("## Selected Calibration Setup")
    report_lines.append("")
    report_lines.append(
        f"The selected calibration setup is **{selected_model} with "
        f"{selected_method} calibration**."
    )
    report_lines.append("")
    report_lines.append(
        "The selection is based mainly on the lowest mean Brier score, "
        "because Brier score directly measures the quality of probability estimates."
    )
    report_lines.append("")
    report_lines.append("## Proposed Risk Thresholds")
    report_lines.append("")
    report_lines.append(
        "| Risk Band | Probability Minimum | Probability Maximum | Recommended Action |"
    )
    report_lines.append(
        "|---|---:|---:|---|"
    )

    for _, row in thresholds_df.iterrows():
        report_lines.append(
            "| "
            f"{row['risk_band']} | "
            f"{row['probability_min']:.2f} | "
            f"{row['probability_max_exclusive']:.2f} | "
            f"{row['recommended_action']} |"
        )

    report_lines.append("")
    report_lines.append("## Interpretation")
    report_lines.append("")
    report_lines.append(
        "The calibration experiment provides a way to convert model probability "
        "outputs into practical risk bands. These thresholds are proposed using "
        "training-fold validation results only and should be reviewed again after "
        "the final held-out test evaluation."
    )
    report_lines.append("")
    report_lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    report_lines.append("")

    CALIBRATION_REPORT.write_text("\n".join(report_lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """
    Description:
        Main script controller.

    Input:
        None.

    Output:
        Saves calibration result files and prints a short summary.
    """
    print("Loading feature data...")
    data = load_feature_data()

    print("Loading training indices...")
    train_indices = load_train_indices()

    train_data = data.iloc[train_indices].copy()

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN].astype(int)

    print(f"Training rows used for calibration: {len(train_data)}")
    print("Held-out test rows are not used.")

    print("Loading selected hyperparameters...")
    selected_params = load_selected_hyperparams()

    models = build_models(selected_params)

    all_fold_results = []
    out_of_fold_store = {}

    for model_name, base_model in models.items():
        for method in CALIBRATION_METHODS:
            print(f"Running calibration: {model_name} with {method}...")

            fold_results, oof_probabilities = run_calibration_cv(
                X=X_train,
                y=y_train,
                model_name=model_name,
                base_model=base_model,
                method=method,
            )

            all_fold_results.extend(fold_results)
            out_of_fold_store[(model_name, method)] = oof_probabilities

    print("Summarising calibration results...")
    summary_df = summarize_results(all_fold_results)

    summary_df = summary_df.sort_values(
        by=["brier_score_mean", "f1_mean"],
        ascending=[True, False],
    ).reset_index(drop=True)

    best_row = summary_df.iloc[0]
    selected_model = best_row["model"]
    selected_method = best_row["calibration_method"]

    selected_probabilities = out_of_fold_store[(selected_model, selected_method)]

    print("Creating proposed risk thresholds...")
    thresholds_df = create_risk_thresholds(
        y_true=y_train,
        y_probability=selected_probabilities,
        selected_model=selected_model,
        selected_method=selected_method,
    )

    print("Saving output files...")
    CALIBRATION_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    PROPOSED_RISK_THRESHOLDS.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(CALIBRATION_RESULTS, index=False)
    thresholds_df.to_csv(PROPOSED_RISK_THRESHOLDS, index=False)

    write_calibration_report(
        summary_df=summary_df,
        thresholds_df=thresholds_df,
        selected_model=selected_model,
        selected_method=selected_method,
    )

    print("")
    print("Calibration experiments completed.")
    print(f"Selected setup: {selected_model} with {selected_method}")
    print(f"Saved: {CALIBRATION_RESULTS}")
    print(f"Saved: {CALIBRATION_REPORT}")
    print(f"Saved: {PROPOSED_RISK_THRESHOLDS}")


if __name__ == "__main__":
    main()