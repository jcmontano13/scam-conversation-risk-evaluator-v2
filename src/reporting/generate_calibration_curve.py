"""
Generate Calibration Reliability Curve

Description:
    This script generates a calibration reliability curve for the existing
    Logistic Regression and Decision Tree models using sigmoid and isotonic
    calibration.

    The script uses the saved training split only and creates out-of-fold
    calibrated probability predictions. The held-out test set is not used.

    This script does not introduce a new dataset, new features, or new models.
    It is only used to generate final report evidence.

Input:
    - data/processed/scam_dialogue_thread_features.csv
    - data/processed/split_indices.json
    - outputs/reports/selected_hyperparams.json, if available

Output:
    - outputs/figures/calibration_reliability_curve.png
    - outputs/figures/calibration_curve_points.csv
    - outputs/figures/calibration_curve_summary.txt
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
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


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

CALIBRATION_FIGURE_PATH = FIGURE_DIR / "calibration_reliability_curve.png"
CALIBRATION_POINTS_PATH = FIGURE_DIR / "calibration_curve_points.csv"
CALIBRATION_SUMMARY_PATH = FIGURE_DIR / "calibration_curve_summary.txt"


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
        print("selected_hyperparams.json not found. Using default model settings.")
        return {}

    with open(SELECTED_HYPERPARAMS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_best_params(payload, model_name):
    """
    Description:
        Extract best parameters from selected_hyperparams.json in a flexible way.

    Input:
        payload: selected hyperparameter JSON payload
        model_name: model name

    Output:
        Dictionary of best parameters
    """
    model_payload = payload.get(model_name, {})

    if isinstance(model_payload, dict):
        return (
            model_payload.get("best_params")
            or model_payload.get("selected_params")
            or model_payload.get("params")
            or {}
        )

    return {}


def build_logistic_regression(selected_params):
    """
    Description:
        Build Logistic Regression model using selected parameters when available.

    Input:
        selected_params: Dictionary from selected_hyperparams.json

    Output:
        Logistic Regression pipeline
    """
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


def build_decision_tree(selected_params):
    """
    Description:
        Build Decision Tree model using selected parameters when available.

    Input:
        selected_params: Dictionary from selected_hyperparams.json

    Output:
        Decision Tree classifier
    """
    dt_params = {
        "random_state": SEED,
    }

    allowed_params = {
        "criterion",
        "max_depth",
        "min_samples_leaf",
        "min_samples_split",
        "class_weight",
    }

    for key, value in selected_params.items():
        cleaned_key = key.replace("model__", "")
        if cleaned_key in allowed_params:
            dt_params[cleaned_key] = value

    return DecisionTreeClassifier(**dt_params)


def build_models():
    """
    Description:
        Build the two existing project models.

    Output:
        Dictionary of model names and model objects
    """
    selected_payload = load_selected_hyperparams()

    lr_params = get_best_params(selected_payload, "Logistic Regression")
    dt_params = get_best_params(selected_payload, "Decision Tree")

    return {
        "Logistic Regression": build_logistic_regression(lr_params),
        "Decision Tree": build_decision_tree(dt_params),
    }


def build_calibrated_classifier(base_model, method, inner_cv):
    """
    Description:
        Build calibrated classifier with compatibility for different sklearn versions.

    Input:
        base_model: Base estimator
        method: Calibration method, either sigmoid or isotonic
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


def calculate_calibration_bins(y_true, y_probability, n_bins=10):
    """
    Description:
        Calculate calibration reliability points manually so sample counts
        can also be saved for Excel recreation.

    Input:
        y_true: Actual labels
        y_probability: Predicted probabilities
        n_bins: Number of probability bins

    Output:
        DataFrame containing calibration points
    """
    calibration_df = pd.DataFrame(
        {
            "actual_label": y_true,
            "predicted_probability": y_probability,
        }
    )

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    calibration_df["bin_index"] = pd.cut(
        calibration_df["predicted_probability"],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    )

    records = []

    for bin_index in range(n_bins):
        bin_data = calibration_df[calibration_df["bin_index"] == bin_index]

        if bin_data.empty:
            continue

        records.append(
            {
                "bin_index": bin_index,
                "bin_start": bin_edges[bin_index],
                "bin_end": bin_edges[bin_index + 1],
                "mean_predicted_probability": bin_data[
                    "predicted_probability"
                ].mean(),
                "observed_scam_rate": bin_data["actual_label"].mean(),
                "sample_count": len(bin_data),
            }
        )

    return pd.DataFrame(records)


def main():
    """
    Description:
        Generate calibration reliability curve, calibration points CSV,
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

    print(f"Training rows used for calibration curve: {len(train_df)}")
    print(f"Held-out test rows not used: {len(test_indices)}")

    models = build_models()
    calibration_methods = ["sigmoid", "isotonic"]

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    all_calibration_points = []
    summary_records = []

    plt.figure(figsize=(8, 6))

    for model_name, base_model in models.items():
        for method in calibration_methods:
            setup_name = f"{model_name} + {method.capitalize()}"
            print(f"Generating out-of-fold probabilities for: {setup_name}")

            out_of_fold_probabilities = np.zeros(len(train_df))

            for fold_number, (fit_index, validation_index) in enumerate(
                outer_cv.split(X_train, y_train),
                start=1,
            ):
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
                    method=method,
                    inner_cv=inner_cv,
                )

                calibrated_model.fit(X_fit, y_fit)

                validation_probabilities = calibrated_model.predict_proba(
                    X_validation
                )[:, 1]

                out_of_fold_probabilities[validation_index] = (
                    validation_probabilities
                )

                print(f"  Completed fold {fold_number}")

            brier_score = brier_score_loss(y_train, out_of_fold_probabilities)

            summary_records.append(
                {
                    "model": model_name,
                    "calibration_method": method,
                    "brier_score": round(brier_score, 6),
                    "training_rows": len(train_df),
                    "held_out_test_rows_used": 0,
                }
            )

            calibration_points_df = calculate_calibration_bins(
                y_true=y_train,
                y_probability=out_of_fold_probabilities,
                n_bins=10,
            )

            calibration_points_df.insert(0, "model", model_name)
            calibration_points_df.insert(1, "calibration_method", method)

            all_calibration_points.append(calibration_points_df)

            plt.plot(
                calibration_points_df["mean_predicted_probability"],
                calibration_points_df["observed_scam_rate"],
                marker="o",
                linewidth=2,
                label=f"{setup_name} (Brier = {brier_score:.4f})",
            )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Perfect calibration",
    )

    plt.title("Calibration Reliability Curve")
    plt.xlabel("Mean Predicted Scam Probability")
    plt.ylabel("Observed Scam Rate")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    print("Saving calibration reliability curve figure...")
    plt.savefig(CALIBRATION_FIGURE_PATH, dpi=300)
    plt.close()

    print("Saving calibration curve points CSV...")
    final_calibration_points_df = pd.concat(
        all_calibration_points,
        ignore_index=True,
    )
    final_calibration_points_df.to_csv(CALIBRATION_POINTS_PATH, index=False)

    print("Saving calibration curve summary...")
    summary_df = pd.DataFrame(summary_records)
    summary_df = summary_df.sort_values(by="brier_score", ascending=True)

    with open(CALIBRATION_SUMMARY_PATH, "w", encoding="utf-8") as file:
        file.write("Calibration Reliability Curve Summary\n")
        file.write("=====================================\n\n")
        file.write("Dataset: Training split only\n")
        file.write("Evaluation method: 5-fold out-of-fold calibrated probabilities\n")
        file.write("Inner calibration CV: 3-fold\n")
        file.write(f"Training rows: {len(train_df)}\n")
        file.write(f"Held-out test rows used: 0 of {len(test_indices)}\n")
        file.write(f"Features used: {', '.join(FEATURE_COLUMNS)}\n\n")
        file.write(summary_df.to_string(index=False))
        file.write("\n\n")
        file.write(
            "Interpretation:\n"
            "The calibration reliability curve compares predicted scam probabilities "
            "with the observed scam rate in each probability bin. Points closer to "
            "the diagonal reference line indicate better calibration. The Brier "
            "score is also reported, where a lower value indicates better "
            "probability calibration.\n"
        )

    print("Calibration curve generation completed.")
    print(f"Figure saved to: {CALIBRATION_FIGURE_PATH}")
    print(f"Curve points saved to: {CALIBRATION_POINTS_PATH}")
    print(f"Summary saved to: {CALIBRATION_SUMMARY_PATH}")


if __name__ == "__main__":
    main()