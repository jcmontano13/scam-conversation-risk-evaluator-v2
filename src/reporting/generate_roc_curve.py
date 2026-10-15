"""
Generate ROC Curve for Held-Out Test Set

Description:
    This script generates a ROC curve for the Logistic Regression and Decision Tree
    models using the existing held-out test set.

    The script does not introduce a new dataset, new features, or new models.
    It reuses the existing processed feature file and saved train/test split.

Input:
    - data/processed/scam_dialogue_thread_features.csv
    - data/processed/split_indices.json

Output:
    - outputs/figures/roc_curve_test_set.png
    - outputs/figures/roc_curve_points.csv
    - outputs/figures/roc_curve_summary.txt
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
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
ROC_FIGURE_PATH = FIGURE_DIR / "roc_curve_test_set.png"
ROC_POINTS_PATH = FIGURE_DIR / "roc_curve_points.csv"
ROC_SUMMARY_PATH = FIGURE_DIR / "roc_curve_summary.txt"


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


def build_models():
    """
    Description:
        Build the two existing project models.

    Output:
        Dictionary of model names and model objects
    """
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=SEED,
        ),
    }


def main():
    """
    Description:
        Generate ROC curve, ROC points CSV, and summary file.
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

    train_df = feature_df.iloc[train_indices].copy()
    test_df = feature_df.iloc[test_indices].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[LABEL_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[LABEL_COLUMN]

    models = build_models()

    roc_point_records = []
    summary_records = []

    plt.figure(figsize=(8, 6))

    for model_name, model in models.items():
        print(f"Training and evaluating ROC curve for: {model_name}")

        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = roc_auc_score(y_test, y_score)

        summary_records.append(
            {
                "model": model_name,
                "roc_auc": round(roc_auc, 6),
                "test_rows": len(test_df),
            }
        )

        for false_positive_rate, true_positive_rate, threshold in zip(
            fpr, tpr, thresholds
        ):
            roc_point_records.append(
                {
                    "model": model_name,
                    "false_positive_rate": false_positive_rate,
                    "true_positive_rate": true_positive_rate,
                    "threshold": threshold,
                }
            )

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model_name} (AUC = {roc_auc:.4f})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="No-skill baseline",
    )

    plt.title("ROC Curve on Held-Out Test Set")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    print("Saving ROC curve figure...")
    plt.savefig(ROC_FIGURE_PATH, dpi=300)
    plt.close()

    print("Saving ROC curve points CSV...")
    roc_points_df = pd.DataFrame(roc_point_records)
    roc_points_df.to_csv(ROC_POINTS_PATH, index=False)

    print("Saving ROC curve summary...")
    summary_df = pd.DataFrame(summary_records)

    with open(ROC_SUMMARY_PATH, "w", encoding="utf-8") as file:
        file.write("ROC Curve Summary\n")
        file.write("=================\n\n")
        file.write("Dataset: Held-out test set\n")
        file.write(f"Test rows: {len(test_df)}\n")
        file.write(f"Features used: {', '.join(FEATURE_COLUMNS)}\n\n")
        file.write(summary_df.to_string(index=False))
        file.write("\n\n")
        file.write(
            "Interpretation:\n"
            "The ROC curve shows how well each model separates scam and non-scam "
            "conversation threads across different classification thresholds. "
            "A higher ROC-AUC value indicates stronger discrimination performance.\n"
        )

    print("ROC curve generation completed.")
    print(f"Figure saved to: {ROC_FIGURE_PATH}")
    print(f"ROC points saved to: {ROC_POINTS_PATH}")
    print(f"Summary saved to: {ROC_SUMMARY_PATH}")


if __name__ == "__main__":
    main()