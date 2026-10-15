"""
alablation_test.py

Description:
    Runs a leave-one-feature-out ablation test for the scam conversation
    classification project.

    The script compares the full structural feature set against versions
    where one feature is removed at a time. This helps identify which
    engineered features contribute most to model performance.

Inputs:
    - data/processed/scam_dialogue_thread_features.csv
    - data/processed/split_indices.json
    - outputs/reports/selected_hyperparams.json, if available

Outputs:
    - outputs/reports/ablation_results.csv
    - outputs/reports/ablation_report.md

Run:
    - python src/evaluation/ablation_test.py

Important:
    - Only train_row_indices from split_indices.json are used.
    - The held-out test set is not used for the ablation test.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
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

N_SPLITS = 5


# ---------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------

def get_output_path(config_name: str, fallback_filename: str) -> Path:
    """
    Description:
        Gets an output path from config.py. If the config variable does not
        exist yet, a fallback output path is used.

    Input:
        config_name: Name of the config variable.
        fallback_filename: Default output filename.

    Output:
        Path object for the output file.
    """
    return getattr(
        config,
        config_name,
        PROJECT_ROOT / "outputs" / "reports" / fallback_filename,
    )


ABLATION_RESULTS = get_output_path(
    "ABLATION_RESULTS",
    "ablation_results.csv",
)

ABLATION_REPORT = get_output_path(
    "ABLATION_REPORT",
    "ablation_report.md",
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
        Pandas DataFrame containing features and target labels.
    """
    feature_path = Path(config.PROCESSED_DATA)

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    data = pd.read_csv(feature_path)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return data


def load_train_indices() -> List[int]:
    """
    Description:
        Loads saved training row indices from split_indices.json.

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
        Loads selected hyperparameters from the grid search output if available.

    Input:
        None.

    Output:
        Dictionary containing selected hyperparameters.
    """
    selected_path = Path(SELECTED_HYPERPARAMS)

    if not selected_path.exists():
        return {}

    with selected_path.open("r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------

def build_logistic_regression_model(selected_params: Dict) -> Pipeline:
    """
    Description:
        Builds the Logistic Regression pipeline using selected hyperparameters
        if available.

    Input:
        selected_params: Dictionary from selected_hyperparams.json.

    Output:
        Scikit-learn Pipeline containing StandardScaler and LogisticRegression.
    """
    lr_params = (
        selected_params
        .get("selected_models", {})
        .get("Logistic Regression", {})
        .get("best_params", {})
    )

    c_value = lr_params.get("model__C", 1.0)
    class_weight = lr_params.get("model__class_weight", None)
    l1_ratio = lr_params.get("model__l1_ratio", None)

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
        selected_params: Dictionary from selected_hyperparams.json.

    Output:
        DecisionTreeClassifier object.
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
        Builds the models used for the ablation test.

    Input:
        selected_params: Dictionary from selected_hyperparams.json.

    Output:
        Dictionary of model names and model objects.
    """
    return {
        "Logistic Regression": build_logistic_regression_model(selected_params),
        "Decision Tree": build_decision_tree_model(selected_params),
    }


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------

def calculate_metrics(y_true, y_pred, y_probability) -> Dict:
    """
    Description:
        Calculates classification metrics for one validation fold.

    Input:
        y_true: True labels.
        y_pred: Predicted labels.
        y_probability: Predicted scam probabilities.

    Output:
        Dictionary containing metric values.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probability),
    }


def evaluate_model_cv(
    model,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    feature_set_name: str,
    removed_feature: str,
    features_used: List[str],
) -> Dict:
    """
    Description:
        Evaluates one model and one feature set using stratified 5-fold
        cross-validation.

    Input:
        model: Scikit-learn model or pipeline.
        model_name: Name of the model.
        X: Feature matrix.
        y: Target labels.
        feature_set_name: Name of the feature set.
        removed_feature: Removed feature name, or None for full feature set.
        features_used: List of feature columns used.

    Output:
        Dictionary containing mean and standard deviation metrics.
    """
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=config.SEED,
    )

    fold_rows = []

    for fold_number, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        X_train_fold = X.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]
        y_train_fold = y.iloc[train_idx]
        y_valid_fold = y.iloc[valid_idx]

        fold_model = clone(model)
        fold_model.fit(X_train_fold, y_train_fold)

        y_pred = fold_model.predict(X_valid_fold)
        y_probability = fold_model.predict_proba(X_valid_fold)[:, 1]

        metrics = calculate_metrics(y_valid_fold, y_pred, y_probability)
        metrics["fold"] = fold_number

        fold_rows.append(metrics)

    fold_df = pd.DataFrame(fold_rows)

    result = {
        "model": model_name,
        "feature_set": feature_set_name,
        "removed_feature": removed_feature,
        "features_used": ", ".join(features_used),
        "n_features": len(features_used),
    }

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    for metric in metric_columns:
        result[f"{metric}_mean"] = fold_df[metric].mean()
        result[f"{metric}_std"] = fold_df[metric].std()

    return result


def create_feature_sets() -> Dict[str, Dict]:
    """
    Description:
        Creates the full feature set and each leave-one-feature-out set.

    Input:
        None.

    Output:
        Dictionary of feature set definitions.
    """
    feature_sets = {
        "full_feature_set": {
            "removed_feature": "None",
            "features": FEATURE_COLUMNS,
        }
    }

    for feature in FEATURE_COLUMNS:
        remaining_features = [item for item in FEATURE_COLUMNS if item != feature]

        feature_sets[f"without_{feature}"] = {
            "removed_feature": feature,
            "features": remaining_features,
        }

    return feature_sets


def add_performance_drops(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Description:
        Calculates performance drop compared with the full feature set
        for each model.

    Input:
        results_df: DataFrame containing ablation results.

    Output:
        DataFrame with performance drop columns added.
    """
    metric_columns = [
        "accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "roc_auc_mean",
    ]

    results_df = results_df.copy()

    for metric in metric_columns:
        results_df[f"{metric}_drop_from_full"] = 0.0

    for model_name in results_df["model"].unique():
        model_mask = results_df["model"] == model_name
        full_row = results_df[
            model_mask & (results_df["feature_set"] == "full_feature_set")
        ]

        if full_row.empty:
            continue

        full_row = full_row.iloc[0]

        for index, row in results_df[model_mask].iterrows():
            for metric in metric_columns:
                drop_value = full_row[metric] - row[metric]
                results_df.loc[index, f"{metric}_drop_from_full"] = drop_value

    return results_df


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def format_metric(value: float) -> str:
    """
    Description:
        Formats metric values for report output.

    Input:
        value: Numeric metric value.

    Output:
        Formatted string.
    """
    return f"{value:.4f}"


def get_largest_f1_drop(results_df: pd.DataFrame, model_name: str) -> pd.Series:
    """
    Description:
        Finds the removed feature that caused the largest F1 drop for a model.

    Input:
        results_df: Ablation results DataFrame.
        model_name: Name of the model.

    Output:
        DataFrame row with the largest F1 drop.
    """
    model_rows = results_df[
        (results_df["model"] == model_name)
        & (results_df["feature_set"] != "full_feature_set")
    ]

    return model_rows.sort_values(
        by="f1_mean_drop_from_full",
        ascending=False,
    ).iloc[0]


def write_ablation_report(results_df: pd.DataFrame) -> None:
    """
    Description:
        Writes the ablation report as a Markdown file.

    Input:
        results_df: DataFrame containing ablation results.

    Output:
        Saves ablation_report.md.
    """
    ABLATION_REPORT.parent.mkdir(parents=True, exist_ok=True)

    report_lines = []

    report_lines.append("# Ablation Report")
    report_lines.append("")
    report_lines.append("## Description")
    report_lines.append("")
    report_lines.append(
        "This report summarises the leave-one-feature-out ablation test for "
        "the scam conversation classification project."
    )
    report_lines.append("")
    report_lines.append(
        "Only the saved training rows from `split_indices.json` were used. "
        "The held-out test set was not used for feature importance decisions."
    )
    report_lines.append("")
    report_lines.append("## Feature Set")
    report_lines.append("")
    report_lines.append("The full feature set contains:")
    report_lines.append("")
    for feature in FEATURE_COLUMNS:
        report_lines.append(f"- `{feature}`")
    report_lines.append("")
    report_lines.append("## Summary Results")
    report_lines.append("")
    report_lines.append(
        "| Model | Feature Set | Removed Feature | Accuracy | Precision | Recall | F1 | ROC-AUC | F1 Drop |"
    )
    report_lines.append(
        "|---|---|---|---:|---:|---:|---:|---:|---:|"
    )

    for _, row in results_df.iterrows():
        report_lines.append(
            "| "
            f"{row['model']} | "
            f"{row['feature_set']} | "
            f"{row['removed_feature']} | "
            f"{format_metric(row['accuracy_mean'])} | "
            f"{format_metric(row['precision_mean'])} | "
            f"{format_metric(row['recall_mean'])} | "
            f"{format_metric(row['f1_mean'])} | "
            f"{format_metric(row['roc_auc_mean'])} | "
            f"{format_metric(row['f1_mean_drop_from_full'])} |"
        )

    report_lines.append("")
    report_lines.append("## Interpretation")
    report_lines.append("")

    for model_name in results_df["model"].unique():
        largest_drop_row = get_largest_f1_drop(results_df, model_name)

        report_lines.append(
            f"For **{model_name}**, removing `{largest_drop_row['removed_feature']}` "
            f"caused the largest F1 drop of "
            f"{format_metric(largest_drop_row['f1_mean_drop_from_full'])}."
        )

    report_lines.append("")
    report_lines.append(
        "A positive drop means that model performance became worse after the "
        "feature was removed. A small or negative drop means the model did not "
        "depend strongly on that feature during cross-validation."
    )
    report_lines.append("")
    report_lines.append(
        "These results should be used together with model coefficients, decision "
        "tree rules, and final test results before making final conclusions."
    )
    report_lines.append("")
    report_lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    report_lines.append("")

    ABLATION_REPORT.write_text("\n".join(report_lines), encoding="utf-8")


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
        Saves ablation_results.csv and ablation_report.md.
    """
    print("Loading feature data...")
    data = load_feature_data()

    print("Loading training indices...")
    train_indices = load_train_indices()

    train_data = data.iloc[train_indices].copy()

    y_train = train_data[TARGET_COLUMN].astype(int)

    print(f"Training rows used for ablation test: {len(train_data)}")
    print("Held-out test rows are not used.")

    print("Loading selected hyperparameters...")
    selected_params = load_selected_hyperparams()

    models = build_models(selected_params)
    feature_sets = create_feature_sets()

    all_results = []

    for model_name, model in models.items():
        for feature_set_name, feature_set_info in feature_sets.items():
            features_used = feature_set_info["features"]
            removed_feature = feature_set_info["removed_feature"]

            print(
                f"Evaluating {model_name} using {feature_set_name}..."
            )

            X_train = train_data[features_used]

            result = evaluate_model_cv(
                model=model,
                model_name=model_name,
                X=X_train,
                y=y_train,
                feature_set_name=feature_set_name,
                removed_feature=removed_feature,
                features_used=features_used,
            )

            all_results.append(result)

    print("Calculating performance drops...")
    results_df = pd.DataFrame(all_results)
    results_df = add_performance_drops(results_df)

    results_df = results_df.sort_values(
        by=["model", "feature_set"],
        ascending=[True, True],
    ).reset_index(drop=True)

    print("Saving output files...")
    ABLATION_RESULTS.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(ABLATION_RESULTS, index=False)
    write_ablation_report(results_df)

    print("")
    print("Ablation test completed.")
    print(f"Saved: {ABLATION_RESULTS}")
    print(f"Saved: {ABLATION_REPORT}")


if __name__ == "__main__":
    main()