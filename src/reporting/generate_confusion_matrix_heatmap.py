"""
Generate Confusion Matrix Heatmap

Description:
    This script generates confusion matrix heatmaps for the final report using
    the existing held-out test evaluation results.

    The script reads the completed test_results.csv file and converts the
    saved TN, FP, FN, and TP values into heatmap figures.

    This script does not introduce a new dataset, new features, new models,
    or new training process. It is only used to convert completed project
    outputs into final report evidence.

Input:
    - outputs/reports/test_results.csv

Output:
    - outputs/figures/confusion_matrix_heatmap.png
    - outputs/figures/confusion_matrix_values.csv
    - outputs/figures/confusion_matrix_summary.txt
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# Allow script to import config.py from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

TEST_RESULTS_PATH = REPORT_DIR / "test_results.csv"

CONFUSION_MATRIX_FIGURE_PATH = FIGURE_DIR / "confusion_matrix_heatmap.png"
CONFUSION_MATRIX_VALUES_PATH = FIGURE_DIR / "confusion_matrix_values.csv"
CONFUSION_MATRIX_SUMMARY_PATH = FIGURE_DIR / "confusion_matrix_summary.txt"


def normalise_column_names(df):
    """
    Description:
        Normalise column names so the script can handle minor naming differences.

    Input:
        DataFrame

    Output:
        DataFrame with cleaned column names
    """
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def check_file_exists(file_path):
    """
    Description:
        Check that a required file exists.

    Input:
        file_path

    Output:
        Raises FileNotFoundError if missing
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")


def find_column(df, possible_names):
    """
    Description:
        Find the first matching column from a list of possible column names.

    Input:
        df: DataFrame
        possible_names: List of possible column names

    Output:
        Matching column name

    Raises:
        ValueError if no matching column is found
    """
    for column_name in possible_names:
        if column_name in df.columns:
            return column_name

    raise ValueError(
        "Could not find required column. "
        f"Expected one of: {possible_names}. "
        f"Available columns are: {list(df.columns)}"
    )


def load_confusion_matrix_results():
    """
    Description:
        Load test results and extract confusion matrix values.

    Input:
        test_results.csv

    Output:
        DataFrame with model, TN, FP, FN, TP values
    """
    print("Loading held-out test results...")
    check_file_exists(TEST_RESULTS_PATH)

    test_results_df = pd.read_csv(TEST_RESULTS_PATH)
    test_results_df = normalise_column_names(test_results_df)

    print("Available test result columns:")
    print(list(test_results_df.columns))

    model_column = find_column(
        test_results_df,
        [
            "model",
            "model_name",
            "classifier",
        ],
    )

    tn_column = find_column(
        test_results_df,
        [
            "tn",
            "true_negative",
            "true_negatives",
            "confusion_tn",
            "true_negative_count",
        ],
    )

    fp_column = find_column(
        test_results_df,
        [
            "fp",
            "false_positive",
            "false_positives",
            "confusion_fp",
            "false_positive_count",
        ],
    )

    fn_column = find_column(
        test_results_df,
        [
            "fn",
            "false_negative",
            "false_negatives",
            "confusion_fn",
            "false_negative_count",
        ],
    )

    tp_column = find_column(
        test_results_df,
        [
            "tp",
            "true_positive",
            "true_positives",
            "confusion_tp",
            "true_positive_count",
        ],
    )

    confusion_df = test_results_df[
        [
            model_column,
            tn_column,
            fp_column,
            fn_column,
            tp_column,
        ]
    ].copy()

    confusion_df = confusion_df.rename(
        columns={
            model_column: "model",
            tn_column: "tn",
            fp_column: "fp",
            fn_column: "fn",
            tp_column: "tp",
        }
    )

    for column in ["tn", "fp", "fn", "tp"]:
        confusion_df[column] = pd.to_numeric(
            confusion_df[column],
            errors="coerce",
        )

    if confusion_df[["tn", "fp", "fn", "tp"]].isnull().any().any():
        raise ValueError(
            "Some confusion matrix values could not be converted to numbers."
        )

    return confusion_df


def create_long_format_confusion_values(confusion_df):
    """
    Description:
        Create long-format confusion matrix values for Excel recreation.

    Input:
        confusion_df

    Output:
        Long-format DataFrame
    """
    records = []

    for _, row in confusion_df.iterrows():
        model_name = row["model"]

        records.extend(
            [
                {
                    "model": model_name,
                    "actual_label": "Non-scam",
                    "predicted_label": "Non-scam",
                    "cell_type": "True Negative",
                    "count": int(row["tn"]),
                },
                {
                    "model": model_name,
                    "actual_label": "Non-scam",
                    "predicted_label": "Scam",
                    "cell_type": "False Positive",
                    "count": int(row["fp"]),
                },
                {
                    "model": model_name,
                    "actual_label": "Scam",
                    "predicted_label": "Non-scam",
                    "cell_type": "False Negative",
                    "count": int(row["fn"]),
                },
                {
                    "model": model_name,
                    "actual_label": "Scam",
                    "predicted_label": "Scam",
                    "cell_type": "True Positive",
                    "count": int(row["tp"]),
                },
            ]
        )

    return pd.DataFrame(records)


def plot_single_confusion_matrix(ax, model_name, tn, fp, fn, tp, max_value):
    """
    Description:
        Plot one confusion matrix heatmap.

    Input:
        ax: Matplotlib axis
        model_name: Model name
        tn, fp, fn, tp: Confusion matrix values
        max_value: Maximum value for consistent heatmap scale

    Output:
        Heatmap on the supplied axis
    """
    matrix = [
        [tn, fp],
        [fn, tp],
    ]

    image = ax.imshow(
        matrix,
        interpolation="nearest",
        vmin=0,
        vmax=max_value,
    )

    ax.set_title(model_name)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Non-scam", "Scam"])
    ax.set_yticklabels(["Non-scam", "Scam"])

    cell_labels = [
        ["TN", "FP"],
        ["FN", "TP"],
    ]

    for row_index in range(2):
        for column_index in range(2):
            value = matrix[row_index][column_index]
            label = cell_labels[row_index][column_index]

            ax.text(
                column_index,
                row_index,
                f"{label}\n{value}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )

    return image


def generate_confusion_matrix_heatmap(confusion_df):
    """
    Description:
        Generate confusion matrix heatmap figure.

    Input:
        confusion_df

    Output:
        confusion_matrix_heatmap.png
    """
    model_count = len(confusion_df)

    if model_count == 0:
        raise ValueError("No model rows found in confusion matrix results.")

    max_value = int(confusion_df[["tn", "fp", "fn", "tp"]].max().max())

    fig, axes = plt.subplots(
        1,
        model_count,
        figsize=(6 * model_count, 5),
        squeeze=False,
    )

    axes = axes[0]

    last_image = None

    for index, (_, row) in enumerate(confusion_df.iterrows()):
        last_image = plot_single_confusion_matrix(
            ax=axes[index],
            model_name=row["model"],
            tn=int(row["tn"]),
            fp=int(row["fp"]),
            fn=int(row["fn"]),
            tp=int(row["tp"]),
            max_value=max_value,
        )

    fig.suptitle("Held-Out Test Confusion Matrix Heatmap", fontsize=14)
    fig.tight_layout()

    if last_image is not None:
        fig.colorbar(
            last_image,
            ax=axes,
            fraction=0.046,
            pad=0.04,
            label="Conversation Count",
        )

    print("Saving confusion matrix heatmap...")
    plt.savefig(CONFUSION_MATRIX_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_summary(confusion_df):
    """
    Description:
        Save a short interpretation summary for the final report.

    Input:
        confusion_df

    Output:
        confusion_matrix_summary.txt
    """
    with open(CONFUSION_MATRIX_SUMMARY_PATH, "w", encoding="utf-8") as file:
        file.write("Confusion Matrix Heatmap Summary\n")
        file.write("================================\n\n")

        file.write("Input file:\n")
        file.write(f"- {TEST_RESULTS_PATH}\n\n")

        file.write("Generated files:\n")
        file.write(f"- {CONFUSION_MATRIX_FIGURE_PATH}\n")
        file.write(f"- {CONFUSION_MATRIX_VALUES_PATH}\n\n")

        file.write("Confusion matrix values:\n")
        file.write(confusion_df.to_string(index=False))
        file.write("\n\n")

        for _, row in confusion_df.iterrows():
            total = int(row["tn"] + row["fp"] + row["fn"] + row["tp"])

            file.write(f"{row['model']}:\n")
            file.write(f"- True negatives: {int(row['tn'])}\n")
            file.write(f"- False positives: {int(row['fp'])}\n")
            file.write(f"- False negatives: {int(row['fn'])}\n")
            file.write(f"- True positives: {int(row['tp'])}\n")
            file.write(f"- Total held-out test rows represented: {total}\n\n")

        file.write("Interpretation:\n")
        file.write(
            "The confusion matrix heatmap shows how each model classified the "
            "held-out test conversations. True positives represent scam "
            "conversations correctly classified as scam. True negatives represent "
            "non-scam conversations correctly classified as non-scam. False "
            "positives are non-scam conversations incorrectly classified as scam. "
            "False negatives are scam conversations incorrectly classified as "
            "non-scam. For scam detection, false negatives are important because "
            "they represent scam conversations that the model failed to detect.\n\n"
        )

        file.write("Note:\n")
        file.write(
            "This script only converts existing held-out test evaluation results "
            "into a final report figure. It does not perform model training, "
            "model tuning, or new evaluation.\n"
        )


def main():
    """
    Description:
        Generate confusion matrix heatmap, Excel recreation CSV, and summary file.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating confusion matrix heatmap...")

    confusion_df = load_confusion_matrix_results()

    print("Saving confusion matrix values for Excel recreation...")
    long_format_df = create_long_format_confusion_values(confusion_df)
    long_format_df.to_csv(CONFUSION_MATRIX_VALUES_PATH, index=False)

    generate_confusion_matrix_heatmap(confusion_df)

    print("Saving confusion matrix summary...")
    save_confusion_matrix_summary(confusion_df)

    print("Confusion matrix heatmap generation completed.")
    print(f"Figure saved to: {CONFUSION_MATRIX_FIGURE_PATH}")
    print(f"Values saved to: {CONFUSION_MATRIX_VALUES_PATH}")
    print(f"Summary saved to: {CONFUSION_MATRIX_SUMMARY_PATH}")


if __name__ == "__main__":
    main()