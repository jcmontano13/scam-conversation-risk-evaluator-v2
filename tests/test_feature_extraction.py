"""
test_feature_extraction.py

Description:
    Runs automated unit checks on the thread-level feature output created by
    parse_and_features.py. These tests confirm that the extracted feature file
    exists, contains the required columns, has valid labels, and that numeric
    feature values are within expected ranges.

Inputs:
    No direct command-line input.
    The test file reads:
    - data/processed/scam_dialogue_thread_features.csv

Output:
    No output file is created.
    Test results are displayed in the terminal through pytest.

Run:
    - pytest tests/test_feature_extraction.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import PROCESSED_DATA


REQUIRED_COLUMNS = {
    "thread_id",
    "orig_index",
    "thread_label",
    "type",
    "turn_count",
    "aggregate_num_urls",
    "aggregate_money_terms",
    "ratio_request_messages",
}


def load_features():
    """
    Description:
        Loads the processed thread-level feature dataset.

    Input:
        No direct input. Reads the feature path from config.py.

    Output:
        A pandas DataFrame containing the extracted features.
    """
    return pd.read_csv(PROCESSED_DATA)


def test_feature_file_exists():
    assert PROCESSED_DATA.exists(), f"Feature file does not exist: {PROCESSED_DATA}"


def test_required_columns_exist():
    df = load_features()
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    assert not missing_columns, f"Missing columns: {missing_columns}"


def test_no_missing_labels():
    df = load_features()

    assert df["thread_label"].isnull().sum() == 0


def test_no_missing_numeric_features():
    df = load_features()

    numeric_features = [
        "turn_count",
        "aggregate_num_urls",
        "aggregate_money_terms",
        "ratio_request_messages",
    ]

    assert df[numeric_features].isnull().sum().sum() == 0


def test_turn_count_greater_than_zero():
    df = load_features()

    assert (df["turn_count"] > 0).all()


def test_url_count_non_negative():
    df = load_features()

    assert (df["aggregate_num_urls"] >= 0).all()


def test_money_terms_non_negative():
    df = load_features()

    assert (df["aggregate_money_terms"] >= 0).all()


def test_ratio_request_messages_valid_range():
    df = load_features()

    assert df["ratio_request_messages"].between(0, 1).all()


def test_label_values_are_binary():
    df = load_features()

    allowed_labels = {0, 1}
    actual_labels = set(df["thread_label"].unique())

    assert actual_labels.issubset(allowed_labels), f"Unexpected labels: {actual_labels}"