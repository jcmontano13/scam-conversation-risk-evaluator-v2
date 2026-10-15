"""
Generate Precision-Recall Curve for Held-Out Test Set

Description:
    This script generates a Precision-Recall curve for the Logistic Regression
    and Decision Tree models using the existing held-out test set.

    The script does not introduce a new dataset, new features, or new models.
    It reuses the existing processed feature file and saved train/test split.

Input:
    - data/processed/scam_dialogue_thread_features.csv
    - data/processed/split_indices.json

Output:
    - outputs/figures/precision_recall_curve_test_set.png
    - outputs/figures/precision_recall_curve_points.csv
    - outputs/figures/precision_recall_curve_summary.txt
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
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
PR_FIGURE_PATH = FIGURE_DIR / "precision_recall_curve_test_set.png"
PR_POINTS_PATH = FIGURE_DIR / "precision_recall_curve_points.csv"
PR_SUMMARY_PATH = FIGURE_DIR / "precision_recall_curve_summary.txt"


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
        Generate Precision-Recall curve, curve points CSV, and summary file.
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

    pr_point_records = []
    summary_records = []

    positive_rate = y_test.mean()

    plt.figure(figsize=(8, 6))

    for model_name, model in models.items():
        print(f"Training and evaluating Precision-Recall curve for: {model_name}")

        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_test, y_score)
        average_precision = average_precision_score(y_test, y_score)

        summary_records.append(
            {
                "model": model_name,
                "average_precision": round(average_precision, 6),
                "test_rows": len(test_df),
            }
        )

        # precision and recall have one more value than thresholds.
        # The final point does not have a threshold, so NaN is used.
        threshold_values = list(thresholds) + [float("nan")]

        for point_index, (precision_value, recall_value, threshold) in enumerate(
            zip(precision, recall, threshold_values)
        ):
            pr_point_records.append(
                {
                    "model": model_name,
                    "point_index": point_index,
                    "recall": recall_value,
                    "precision": precision_value,
                    "threshold": threshold,
                }
            )

        plt.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{model_name} (AP = {average_precision:.4f})",
        )

    plt.axhline(
        y=positive_rate,
        linestyle="--",
        linewidth=1,
        label=f"No-skill baseline = {positive_rate:.2f}",
    )

    plt.title("Precision-Recall Curve on Held-Out Test Set")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    print("Saving Precision-Recall curve figure...")
    plt.savefig(PR_FIGURE_PATH, dpi=300)
    plt.close()

    print("Saving Precision-Recall curve points CSV...")
    pr_points_df = pd.DataFrame(pr_point_records)
    pr_points_df.to_csv(PR_POINTS_PATH, index=False)

    print("Saving Precision-Recall curve summary...")
    summary_df = pd.DataFrame(summary_records)

    with open(PR_SUMMARY_PATH, "w", encoding="utf-8") as file:
        file.write("Precision-Recall Curve Summary\n")
        file.write("===============================\n\n")
        file.write("Dataset: Held-out test set\n")
        file.write(f"Test rows: {len(test_df)}\n")
        file.write(f"Positive class rate: {positive_rate:.4f}\n")
        file.write(f"Features used: {', '.join(FEATURE_COLUMNS)}\n\n")
        file.write(summary_df.to_string(index=False))
        file.write("\n\n")
        file.write(
            "Interpretation:\n"
            "The Precision-Recall curve shows the trade-off between correctly "
            "detecting scam conversations and avoiding false scam predictions. "
            "A higher Average Precision score indicates stronger performance "
            "across different classification thresholds.\n"
        )

    print("Precision-Recall curve generation completed.")
    print(f"Figure saved to: {PR_FIGURE_PATH}")
    print(f"Curve points saved to: {PR_POINTS_PATH}")
    print(f"Summary saved to: {PR_SUMMARY_PATH}")


if __name__ == "__main__":
    main()