"""
interpretability_artifacts.py

Description:
    Generates model interpretability artifacts for the scam conversation
    classification project.

    This script trains the selected Logistic Regression and Decision Tree
    models using only the saved training rows. It then creates simple
    interpretability outputs that can be used in the final report.

Inputs:
    - data/processed/scam_dialogue_thread_features.csv
    - data/processed/split_indices.json
    - outputs/reports/selected_hyperparams.json, if available

Outputs:
    - outputs/reports/coefficients_table.csv
    - outputs/reports/tree_feature_importance.csv
    - outputs/reports/decision_tree_rules.txt
    - outputs/reports/interpretability_report.md

Run:
    - python src/evaluation/interpretability_artifacts.py

Important:
    - Only train_row_indices from split_indices.json are used.
    - The held-out test set is not used to create interpretability artifacts.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


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


# ---------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------

def get_output_path(config_name: str, fallback_filename: str) -> Path:
    """
    Description:
        Gets an output path from config.py. If the config variable does not
        exist yet, a fallback path inside outputs/reports is used.

    Input:
        config_name: Name of the config variable.
        fallback_filename: Default filename if the config variable is missing.

    Output:
        Path object for the output file.
    """
    return getattr(
        config,
        config_name,
        PROJECT_ROOT / "outputs" / "reports" / fallback_filename,
    )


COEFFICIENTS_TABLE = get_output_path(
    "COEFFICIENTS_TABLE",
    "coefficients_table.csv",
)

TREE_FEATURE_IMPORTANCE = get_output_path(
    "TREE_FEATURE_IMPORTANCE",
    "tree_feature_importance.csv",
)

TREE_RULES = get_output_path(
    "TREE_RULES",
    "decision_tree_rules.txt",
)

INTERPRETABILITY_REPORT = get_output_path(
    "INTERPRETABILITY_REPORT",
    "interpretability_report.md",
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
        Loads selected hyperparameters from selected_hyperparams.json if
        the file exists.

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
        Builds the Decision Tree model using selected hyperparameters if
        available.

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


# ---------------------------------------------------------------------
# Interpretability outputs
# ---------------------------------------------------------------------

def create_coefficients_table(
    logistic_pipeline: Pipeline,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Description:
        Creates a coefficient table for the trained Logistic Regression model.

    Input:
        logistic_pipeline: Trained Logistic Regression pipeline.
        feature_columns: List of feature names.

    Output:
        DataFrame containing coefficients and interpretation direction.
    """
    logistic_model = logistic_pipeline.named_steps["model"]
    coefficients = logistic_model.coef_[0]

    rows = []

    for feature_name, coefficient in zip(feature_columns, coefficients):
        if coefficient > 0:
            direction = "Increases predicted scam probability"
        elif coefficient < 0:
            direction = "Decreases predicted scam probability"
        else:
            direction = "No effect in the fitted model"

        rows.append(
            {
                "feature": feature_name,
                "coefficient": coefficient,
                "absolute_coefficient": abs(coefficient),
                "direction": direction,
                "note": "Coefficient is based on standardized feature values.",
            }
        )

    coefficients_df = pd.DataFrame(rows)

    coefficients_df = coefficients_df.sort_values(
        by="absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)

    return coefficients_df


def create_tree_feature_importance_table(
    decision_tree_model: DecisionTreeClassifier,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Description:
        Creates a feature importance table for the trained Decision Tree model.

    Input:
        decision_tree_model: Trained Decision Tree classifier.
        feature_columns: List of feature names.

    Output:
        DataFrame containing Decision Tree feature importance values.
    """
    rows = []

    for feature_name, importance in zip(
        feature_columns,
        decision_tree_model.feature_importances_,
    ):
        rows.append(
            {
                "feature": feature_name,
                "importance": importance,
            }
        )

    importance_df = pd.DataFrame(rows)

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    ).reset_index(drop=True)

    return importance_df


def create_decision_tree_rules(
    decision_tree_model: DecisionTreeClassifier,
    feature_columns: List[str],
) -> str:
    """
    Description:
        Exports the trained Decision Tree as readable text rules.

    Input:
        decision_tree_model: Trained Decision Tree classifier.
        feature_columns: List of feature names.

    Output:
        String containing text version of the decision tree rules.
    """
    return export_text(
        decision_tree_model,
        feature_names=feature_columns,
        decimals=4,
    )


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def write_interpretability_report(
    coefficients_df: pd.DataFrame,
    tree_importance_df: pd.DataFrame,
) -> None:
    """
    Description:
        Writes a Markdown report summarising the model interpretability outputs.

    Input:
        coefficients_df: Logistic Regression coefficient table.
        tree_importance_df: Decision Tree feature importance table.

    Output:
        Saves interpretability_report.md.
    """
    INTERPRETABILITY_REPORT.parent.mkdir(parents=True, exist_ok=True)

    top_lr_feature = coefficients_df.iloc[0]["feature"]
    top_lr_direction = coefficients_df.iloc[0]["direction"]
    top_tree_feature = tree_importance_df.iloc[0]["feature"]

    report_lines = []

    report_lines.append("# Interpretability Report")
    report_lines.append("")
    report_lines.append("## Description")
    report_lines.append("")
    report_lines.append(
        "This report summarises simple interpretability artifacts for the "
        "Logistic Regression and Decision Tree models."
    )
    report_lines.append("")
    report_lines.append(
        "Only the saved training rows from `split_indices.json` were used. "
        "The held-out test set was not used to create these interpretability artifacts."
    )
    report_lines.append("")
    report_lines.append("## Logistic Regression Coefficients")
    report_lines.append("")
    report_lines.append(
        "The Logistic Regression coefficients show how each standardized structural "
        "feature affects the predicted scam probability."
    )
    report_lines.append("")
    report_lines.append("| Feature | Coefficient | Absolute Coefficient | Direction |")
    report_lines.append("|---|---:|---:|---|")

    for _, row in coefficients_df.iterrows():
        report_lines.append(
            "| "
            f"{row['feature']} | "
            f"{row['coefficient']:.6f} | "
            f"{row['absolute_coefficient']:.6f} | "
            f"{row['direction']} |"
        )

    report_lines.append("")
    report_lines.append("## Decision Tree Feature Importance")
    report_lines.append("")
    report_lines.append(
        "The Decision Tree feature importance values show which features contributed "
        "most to the tree splits."
    )
    report_lines.append("")
    report_lines.append("| Feature | Importance |")
    report_lines.append("|---|---:|")

    for _, row in tree_importance_df.iterrows():
        report_lines.append(
            "| "
            f"{row['feature']} | "
            f"{row['importance']:.6f} |"
        )

    report_lines.append("")
    report_lines.append("## Main Interpretation")
    report_lines.append("")
    report_lines.append(
        f"For Logistic Regression, the strongest feature by absolute coefficient is "
        f"`{top_lr_feature}`. Its direction is: {top_lr_direction.lower()}."
    )
    report_lines.append("")
    report_lines.append(
        f"For the Decision Tree, the most important feature is `{top_tree_feature}` "
        f"based on feature importance."
    )
    report_lines.append("")
    report_lines.append(
        "These outputs support the use of compact structural features because the "
        "model decisions can be explained using feature coefficients, feature "
        "importance values, and readable tree rules."
    )
    report_lines.append("")
    report_lines.append(
        "The Decision Tree rules are saved separately in `decision_tree_rules.txt`."
    )
    report_lines.append("")
    report_lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    report_lines.append("")

    INTERPRETABILITY_REPORT.write_text("\n".join(report_lines), encoding="utf-8")


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
        Saves interpretability artifact files and prints a short summary.
    """
    print("Loading feature data...")
    data = load_feature_data()

    print("Loading training indices...")
    train_indices = load_train_indices()

    train_data = data.iloc[train_indices].copy()

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN].astype(int)

    print(f"Training rows used for interpretability artifacts: {len(train_data)}")
    print("Held-out test rows are not used.")

    print("Loading selected hyperparameters...")
    selected_params = load_selected_hyperparams()

    print("Training Logistic Regression model...")
    logistic_pipeline = build_logistic_regression_model(selected_params)
    logistic_pipeline.fit(X_train, y_train)

    print("Training Decision Tree model...")
    decision_tree_model = build_decision_tree_model(selected_params)
    decision_tree_model.fit(X_train, y_train)

    print("Creating Logistic Regression coefficient table...")
    coefficients_df = create_coefficients_table(
        logistic_pipeline=logistic_pipeline,
        feature_columns=FEATURE_COLUMNS,
    )

    print("Creating Decision Tree feature importance table...")
    tree_importance_df = create_tree_feature_importance_table(
        decision_tree_model=decision_tree_model,
        feature_columns=FEATURE_COLUMNS,
    )

    print("Exporting Decision Tree rules...")
    tree_rules_text = create_decision_tree_rules(
        decision_tree_model=decision_tree_model,
        feature_columns=FEATURE_COLUMNS,
    )

    print("Saving output files...")
    COEFFICIENTS_TABLE.parent.mkdir(parents=True, exist_ok=True)
    TREE_FEATURE_IMPORTANCE.parent.mkdir(parents=True, exist_ok=True)
    TREE_RULES.parent.mkdir(parents=True, exist_ok=True)

    coefficients_df.to_csv(COEFFICIENTS_TABLE, index=False)
    tree_importance_df.to_csv(TREE_FEATURE_IMPORTANCE, index=False)
    TREE_RULES.write_text(tree_rules_text, encoding="utf-8")

    write_interpretability_report(
        coefficients_df=coefficients_df,
        tree_importance_df=tree_importance_df,
    )

    print("")
    print("Interpretability artifacts completed.")
    print(f"Saved: {COEFFICIENTS_TABLE}")
    print(f"Saved: {TREE_FEATURE_IMPORTANCE}")
    print(f"Saved: {TREE_RULES}")
    print(f"Saved: {INTERPRETABILITY_REPORT}")


if __name__ == "__main__":
    main()