"""
create_split_indices.py

Description:
    Creates a fixed stratified 80/20 train/test split from the processed
    thread-level feature dataset.

Justification:
    This script saves the exact train and test row indices to support
    reproducibility, fair model comparison, and auditability. A fixed split
    ensures that all future modelling experiments use the same training and
    held-out test records. This prevents accidental changes in evaluation
    results caused by random re-splitting of the dataset.

    Stratification is used so that the scam and non-scam label distribution is
    preserved across the training and test sets. This is important because the
    final model evaluation should reflect a fair and stable comparison between
    Logistic Regression and Decision Tree models.

    The saved split_indices.json file also helps prevent test leakage by clearly
    separating the held-out test set before model training begins.

Input:
    - data/processed/scam_dialogue_thread_features.csv

Output:
    - data/processed/split_indices.json

Run: 
    - python src/preprocessing/create_split_indices.py

"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import PROCESSED_DATA, SPLIT_INDICES, SEED


LABEL_COLUMN = "thread_label"
TEST_SIZE = 0.20


def load_feature_data(file_path):
    """
    Description:
        Loads the processed thread-level feature dataset from CSV.

    Input:
        file_path: Path to the processed feature CSV file.

    Output:
        A pandas DataFrame containing the processed feature data.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Feature file not found: {file_path}")

    return pd.read_csv(file_path)


def validate_feature_data(df):
    """
    Description:
        Validates that the feature dataset is suitable for creating a
        stratified train/test split.

    Input:
        df: DataFrame containing the processed feature dataset.

    Output:
        None. Raises ValueError if validation fails.
    """
    if df.empty:
        raise ValueError("Feature dataset is empty.")

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing required label column: {LABEL_COLUMN}")

    missing_labels = df[LABEL_COLUMN].isnull().sum()

    if missing_labels > 0:
        raise ValueError(f"Label column contains missing values: {missing_labels}")

    label_count = df[LABEL_COLUMN].nunique()

    if label_count < 2:
        raise ValueError("Stratified split requires at least two label classes.")


def create_stratified_split(df, test_size, seed):
    """
    Description:
        Creates a stratified train/test split using the configured label column.

    Input:
        df: DataFrame containing the processed feature dataset.
        test_size: Proportion of the dataset to assign to the test split.
        seed: Random seed used to make the split reproducible.

    Output:
        train_indices: Sorted list of row indices assigned to the training set.
        test_indices: Sorted list of row indices assigned to the test set.
    """
    row_indices = df.index.tolist()
    labels = df[LABEL_COLUMN]

    train_indices, test_indices = train_test_split(
        row_indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    train_indices = sorted([int(index) for index in train_indices])
    test_indices = sorted([int(index) for index in test_indices])

    return train_indices, test_indices


def get_label_distribution(df, indices):
    """
    Description:
        Computes the label distribution for a selected group of row indices.

    Input:
        df: DataFrame containing the processed feature dataset.
        indices: List of row indices for either the train or test split.

    Output:
        A dictionary containing label counts.
    """
    distribution = df.loc[indices, LABEL_COLUMN].value_counts().sort_index()

    return {str(label): int(count) for label, count in distribution.items()}


def get_original_indices(df, row_indices):
    """
    Description:
        Retrieves original dataset indices if the orig_index column exists.

    Input:
        df: DataFrame containing the processed feature dataset.
        row_indices: List of row indices from the processed feature dataset.

    Output:
        A list of original dataset indices if available.
        Otherwise, returns an empty list.
    """
    if "orig_index" not in df.columns:
        return []

    return [int(value) for value in df.loc[row_indices, "orig_index"].tolist()]


def build_split_payload(df, train_indices, test_indices):
    """
    Description:
        Builds a JSON-ready dictionary containing split metadata, row indices,
        original indices, and label distributions.

    Input:
        df: DataFrame containing the processed feature dataset.
        train_indices: List of processed dataset row indices for training.
        test_indices: List of processed dataset row indices for testing.

    Output:
        A dictionary that can be saved as split_indices.json.
    """
    payload = {
        "seed": int(SEED),
        "test_size": TEST_SIZE,
        "label_column": LABEL_COLUMN,
        "total_rows": int(len(df)),
        "train_size": int(len(train_indices)),
        "test_size_rows": int(len(test_indices)),
        "train_row_indices": train_indices,
        "test_row_indices": test_indices,
        "train_original_indices": get_original_indices(df, train_indices),
        "test_original_indices": get_original_indices(df, test_indices),
        "overall_label_distribution": get_label_distribution(df, df.index.tolist()),
        "train_label_distribution": get_label_distribution(df, train_indices),
        "test_label_distribution": get_label_distribution(df, test_indices),
    }

    return payload


def save_split_indices(payload, output_path):
    """
    Description:
        Saves the split metadata and indices to a JSON file.

    Input:
        payload: Dictionary containing split metadata and indices.
        output_path: Path where the JSON file will be saved.

    Output:
        Creates or overwrites split_indices.json.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)


def print_split_summary(payload):
    """
    Description:
        Prints a readable summary of the generated train/test split.

    Input:
        payload: Dictionary containing split metadata and label distributions.

    Output:
        Prints split details to the terminal.
    """
    print("Train/Test Split Summary")
    print("=" * 30)
    print(f"Seed: {payload['seed']}")
    print(f"Total rows: {payload['total_rows']}")
    print(f"Train rows: {payload['train_size']}")
    print(f"Test rows: {payload['test_size_rows']}")
    print("")
    print("Overall label distribution:")
    print(payload["overall_label_distribution"])
    print("")
    print("Train label distribution:")
    print(payload["train_label_distribution"])
    print("")
    print("Test label distribution:")
    print(payload["test_label_distribution"])
    print("")
    print(f"Saved split indices to: {SPLIT_INDICES}")


def main():
    """
    Description:
        Runs the full split generation process. It loads the processed feature
        dataset, validates it, creates a stratified 80/20 split, saves the split
        indices to JSON, and prints a summary.

    Input:
        No direct function input.
        Reads PROCESSED_DATA, SPLIT_INDICES, and SEED from config.py.

    Output:
        Creates split_indices.json and prints split summary to the terminal.
    """
    features_df = load_feature_data(PROCESSED_DATA)

    validate_feature_data(features_df)

    train_indices, test_indices = create_stratified_split(
        features_df,
        test_size=TEST_SIZE,
        seed=SEED,
    )

    payload = build_split_payload(
        features_df,
        train_indices,
        test_indices,
    )

    save_split_indices(payload, SPLIT_INDICES)

    print_split_summary(payload)


if __name__ == "__main__":
    main()