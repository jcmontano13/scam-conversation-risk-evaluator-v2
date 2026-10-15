"""
check_features.py

Description:
    Validates the parsed dialogue output and thread-level feature output created
    by parse_and_features.py.

Inputs:
    - data/processed/scam_dialogue_parsed.csv
    - data/processed/scam_dialogue_thread_features.csv

Output:
    - outputs/reports/feature_validation_report.txt

Run: 
    - python src/validation/check_features.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import PARSED_DATA, PROCESSED_DATA, VALIDATION_REPORT


REQUIRED_PARSED_COLUMNS = {
    "thread_id",
    "message_id",
    "turn_index",
    "speaker_role",
    "message_text",
    "thread_label",
    "type",
}

REQUIRED_FEATURE_COLUMNS = {
    "thread_id",
    "orig_index",
    "thread_label",
    "type",
    "turn_count",
    "aggregate_num_urls",
    "aggregate_money_terms",
    "ratio_request_messages",
}


def add_result(results, check_name, passed, detail):
    """
    Description:
        Adds one validation result to the results list.

    Input:
        results: A list that stores validation result dictionaries.
        check_name: The name or description of the validation check.
        passed: Boolean value showing whether the check passed.
        detail: Additional explanation about the validation result.

    Output:
        Updates the results list by appending one validation result.
    """
    results.append(
        {
            "check": check_name,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        }
    )


def check_file_exists(file_path):
    """
    Description:
        Checks whether a required file exists.

    Input:
        file_path: Path object pointing to the file being checked.

    Output:
        True if the file exists.
        False otherwise.
    """
    return Path(file_path).exists()


def check_required_columns(df, required_columns):
    """
    Description:
        Checks whether a dataframe contains all required columns.

    Input:
        df: The pandas DataFrame being checked.
        required_columns: A set of column names expected in the DataFrame.

    Output:
        A set of missing column names.
        Returns an empty set if no columns are missing.
    """
    return required_columns - set(df.columns)


def validate_parsed_data(parsed_df, results):
    """
    Description:
        Runs validation checks on the parsed turn-level dataset.

    Input:
        parsed_df: DataFrame containing parsed dialogue turns.
        results: A list used to store validation results.

    Output:
        Updates the results list with parsed dataset validation outcomes.
    """
    missing_columns = check_required_columns(parsed_df, REQUIRED_PARSED_COLUMNS)

    add_result(
        results,
        "Parsed data required columns",
        len(missing_columns) == 0,
        f"Missing columns: {missing_columns}" if missing_columns else "All required columns exist.",
    )

    add_result(
        results,
        "Parsed data row count",
        len(parsed_df) > 0,
        f"Parsed rows: {len(parsed_df)}",
    )

    if "thread_label" in parsed_df.columns:
        missing_labels = parsed_df["thread_label"].isnull().sum()

        add_result(
            results,
            "Parsed data missing labels",
            missing_labels == 0,
            f"Missing labels: {missing_labels}",
        )

    if "turn_index" in parsed_df.columns:
        invalid_turn_index = (parsed_df["turn_index"] < 0).sum()

        add_result(
            results,
            "Parsed data valid turn_index",
            invalid_turn_index == 0,
            f"Invalid turn_index rows: {invalid_turn_index}",
        )


def validate_feature_data(features_df, results):
    """
    Description:
        Runs validation checks on the thread-level feature dataset.

    Input:
        features_df: DataFrame containing thread-level extracted features.
        results: A list used to store validation results.

    Output:
        Updates the results list with feature dataset validation outcomes.
    """
    missing_columns = check_required_columns(features_df, REQUIRED_FEATURE_COLUMNS)

    add_result(
        results,
        "Feature data required columns",
        len(missing_columns) == 0,
        f"Missing columns: {missing_columns}" if missing_columns else "All required columns exist.",
    )

    add_result(
        results,
        "Feature data row count",
        len(features_df) > 0,
        f"Feature rows: {len(features_df)}",
    )

    if "thread_label" in features_df.columns:
        missing_labels = features_df["thread_label"].isnull().sum()

        add_result(
            results,
            "Feature data missing labels",
            missing_labels == 0,
            f"Missing labels: {missing_labels}",
        )

    numeric_features = [
        "turn_count",
        "aggregate_num_urls",
        "aggregate_money_terms",
        "ratio_request_messages",
    ]

    existing_numeric_features = [
        col for col in numeric_features if col in features_df.columns
    ]

    missing_values = features_df[existing_numeric_features].isnull().sum().sum()

    add_result(
        results,
        "Feature data missing numeric values",
        missing_values == 0,
        f"Missing numeric feature values: {missing_values}",
    )

    if "turn_count" in features_df.columns:
        invalid_turn_count = (features_df["turn_count"] <= 0).sum()

        add_result(
            results,
            "turn_count greater than 0",
            invalid_turn_count == 0,
            f"Invalid turn_count rows: {invalid_turn_count}",
        )

    if "aggregate_num_urls" in features_df.columns:
        invalid_url_count = (features_df["aggregate_num_urls"] < 0).sum()

        add_result(
            results,
            "aggregate_num_urls non-negative",
            invalid_url_count == 0,
            f"Invalid URL count rows: {invalid_url_count}",
        )

    if "aggregate_money_terms" in features_df.columns:
        invalid_money_count = (features_df["aggregate_money_terms"] < 0).sum()

        add_result(
            results,
            "aggregate_money_terms non-negative",
            invalid_money_count == 0,
            f"Invalid money term count rows: {invalid_money_count}",
        )

    if "ratio_request_messages" in features_df.columns:
        invalid_ratio = (
            (features_df["ratio_request_messages"] < 0)
            | (features_df["ratio_request_messages"] > 1)
        ).sum()

        add_result(
            results,
            "ratio_request_messages between 0 and 1",
            invalid_ratio == 0,
            f"Invalid ratio rows: {invalid_ratio}",
        )


def build_summary_report(results, features_df):
    """
    Description:
        Builds a text report containing validation results and basic feature
        summary statistics.

    Input:
        results: A list of validation result dictionaries.
        features_df: DataFrame containing thread-level extracted features.

    Output:
        A formatted validation report as a string.
    """
    report_lines = []

    report_lines.append("Feature Validation Report")
    report_lines.append("=" * 30)
    report_lines.append("")

    for result in results:
        report_lines.append(f"[{result['status']}] {result['check']}")
        report_lines.append(f"    {result['detail']}")
        report_lines.append("")

    numeric_features = [
        "turn_count",
        "aggregate_num_urls",
        "aggregate_money_terms",
        "ratio_request_messages",
    ]

    existing_numeric_features = [
        col for col in numeric_features if col in features_df.columns
    ]

    if existing_numeric_features:
        report_lines.append("Feature Summary Statistics")
        report_lines.append("-" * 30)
        report_lines.append(
            features_df[existing_numeric_features].describe().to_string()
        )
        report_lines.append("")

    if "thread_label" in features_df.columns:
        report_lines.append("Label Distribution")
        report_lines.append("-" * 30)
        report_lines.append(features_df["thread_label"].value_counts().to_string())
        report_lines.append("")

    return "\n".join(report_lines)


def save_report(report_text):
    """
    Description:
        Saves the validation report text to the configured output path.

    Input:
        report_text: The full validation report as a string.

    Output:
        Creates or overwrites the validation report text file.
    """
    VALIDATION_REPORT.parent.mkdir(parents=True, exist_ok=True)

    with open(VALIDATION_REPORT, "w", encoding="utf-8") as file:
        file.write(report_text)


def main():
    """
    Description:
        Executes all validation checks for parsed and feature datasets, prints
        the validation report, and saves it to the reports folder.

    Input:
        No direct function input.
        Reads processed dataset paths from config.py.

    Output:
        Prints validation results to the terminal.
        Saves feature_validation_report.txt.
    """
    results = []

    parsed_exists = check_file_exists(PARSED_DATA)
    features_exists = check_file_exists(PROCESSED_DATA)

    add_result(
        results,
        "Parsed CSV exists",
        parsed_exists,
        str(PARSED_DATA),
    )

    add_result(
        results,
        "Feature CSV exists",
        features_exists,
        str(PROCESSED_DATA),
    )

    if not parsed_exists or not features_exists:
        report_text = build_summary_report(results, pd.DataFrame())
        print(report_text)
        save_report(report_text)
        raise FileNotFoundError("Required processed file is missing.")

    parsed_df = pd.read_csv(PARSED_DATA)
    features_df = pd.read_csv(PROCESSED_DATA)

    validate_parsed_data(parsed_df, results)
    validate_feature_data(features_df, results)

    report_text = build_summary_report(results, features_df)

    print(report_text)
    save_report(report_text)

    failed_checks = [result for result in results if result["status"] == "FAIL"]

    if failed_checks:
        raise ValueError(f"Validation failed. Failed checks: {len(failed_checks)}")

    print(f"Validation report saved to: {VALIDATION_REPORT}")


if __name__ == "__main__":
    main()