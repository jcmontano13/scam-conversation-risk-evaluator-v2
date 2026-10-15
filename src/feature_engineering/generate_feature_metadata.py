"""
generate_feature_metadata.py

Description:
    Generates metadata for the feature extraction process. This file documents
    the dataset paths, output paths, extracted features, row counts, label
    distribution, redaction rules, and reproducibility settings.

Justification:
    The metadata file supports reproducibility and auditability by recording
    what inputs, outputs, features, and preprocessing assumptions were used
    during feature extraction. This is useful for project marking, future
    reruns, and final report documentation.

Input:
    - data/raw/scam-dialogue_all.csv
    - data/processed/scam_dialogue_parsed.csv
    - data/processed/scam_dialogue_thread_features.csv

Output:
    - data/processed/feature_extraction_metadata.json

Run: 
    - python src/feature_engineering/generate_feature_metadata.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import RAW_DATA, PARSED_DATA, PROCESSED_DATA, FEATURE_METADATA, SEED


STRUCTURAL_FEATURES = [
    "turn_count",
    "aggregate_num_urls",
    "aggregate_money_terms",
    "ratio_request_messages",
]

REDACTION_RULES = {
    "email": "Email-like patterns are replaced with [REDACTED_EMAIL].",
    "phone": "Phone-like number patterns are replaced with [REDACTED_PHONE].",
    "ssn": "SSN-like patterns are replaced with [REDACTED_SSN].",
    "amount": "Money amount patterns are replaced with [REDACTED_AMOUNT].",
}

FEATURE_DEFINITIONS = {
    "turn_count": "Number of parsed dialogue turns in a conversation thread.",
    "aggregate_num_urls": "Total number of URL patterns detected across all turns in a thread.",
    "aggregate_money_terms": "Total count of money-related keywords and amount patterns across a thread.",
    "ratio_request_messages": "Proportion of messages in a thread that appear to contain a request.",
}


def load_csv(file_path):
    """
    Description:
        Loads a CSV file into a pandas DataFrame.

    Input:
        file_path: Path object pointing to the CSV file.

    Output:
        A pandas DataFrame containing the CSV data.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)


def get_file_metadata(file_path):
    """
    Description:
        Collects basic file information such as path, existence, and file size.

    Input:
        file_path: Path object pointing to the file.

    Output:
        A dictionary containing file path, existence status, and file size.
    """
    path = Path(file_path)

    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def get_label_distribution(df, label_column):
    """
    Description:
        Computes the distribution of labels in a DataFrame.

    Input:
        df: DataFrame containing a label column.
        label_column: Name of the label column.

    Output:
        A dictionary containing label values and their counts.
    """
    if label_column not in df.columns:
        return {}

    distribution = df[label_column].value_counts().sort_index()

    return {str(label): int(count) for label, count in distribution.items()}


def get_feature_summary(features_df):
    """
    Description:
        Creates summary statistics for the structural feature columns.

    Input:
        features_df: DataFrame containing thread-level structural features.

    Output:
        A dictionary containing descriptive statistics for each feature.
    """
    existing_features = [
        feature for feature in STRUCTURAL_FEATURES if feature in features_df.columns
    ]

    if not existing_features:
        return {}

    summary = features_df[existing_features].describe().to_dict()

    cleaned_summary = {}

    for feature_name, stats in summary.items():
        cleaned_summary[feature_name] = {
            stat_name: float(stat_value) for stat_name, stat_value in stats.items()
        }

    return cleaned_summary


def build_metadata(raw_df, parsed_df, features_df):
    """
    Description:
        Builds a metadata dictionary describing the feature extraction process.

    Input:
        raw_df: DataFrame containing the original raw dataset.
        parsed_df: DataFrame containing parsed turn-level data.
        features_df: DataFrame containing thread-level feature data.

    Output:
        A dictionary containing feature extraction metadata.
    """
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_stage": "Week 1 - Feature extraction and reproducibility setup",
        "random_seed": int(SEED),
        "script_name": "generate_feature_metadata.py",
        "feature_extraction_script": "parse_and_features.py",
        "input_files": {
            "raw_data": get_file_metadata(RAW_DATA),
        },
        "output_files": {
            "parsed_data": get_file_metadata(PARSED_DATA),
            "thread_features": get_file_metadata(PROCESSED_DATA),
            "feature_metadata": str(FEATURE_METADATA),
        },
        "row_counts": {
            "raw_rows": int(len(raw_df)),
            "parsed_turn_rows": int(len(parsed_df)),
            "thread_feature_rows": int(len(features_df)),
        },
        "columns": {
            "raw_columns": list(raw_df.columns),
            "parsed_columns": list(parsed_df.columns),
            "feature_columns": list(features_df.columns),
        },
        "structural_features": STRUCTURAL_FEATURES,
        "feature_definitions": FEATURE_DEFINITIONS,
        "redaction_rules": REDACTION_RULES,
        "label_distribution": {
            "raw_data": get_label_distribution(raw_df, "label"),
            "parsed_data": get_label_distribution(parsed_df, "thread_label"),
            "feature_data": get_label_distribution(features_df, "thread_label"),
        },
        "feature_summary_statistics": get_feature_summary(features_df),
        "excluded_features": {
            "tf_idf": "Excluded to preserve interpretability and reduce computational complexity.",
            "deep_learning_embeddings": "Excluded because the project focuses on lightweight explainable structural features.",
            "llm_features": "Excluded because the project does not include LLM fine-tuning or deployment.",
        },
    }

    return metadata


def save_metadata(metadata, output_path):
    """
    Description:
        Saves the metadata dictionary as a JSON file.

    Input:
        metadata: Dictionary containing feature extraction metadata.
        output_path: Path where the JSON metadata file will be saved.

    Output:
        Creates or overwrites feature_extraction_metadata.json.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)


def print_metadata_summary(metadata):
    """
    Description:
        Prints a readable summary of the generated metadata.

    Input:
        metadata: Dictionary containing feature extraction metadata.

    Output:
        Prints key metadata values to the terminal.
    """
    print("Feature Extraction Metadata Summary")
    print("=" * 40)
    print(f"Generated at: {metadata['generated_at']}")
    print(f"Random seed: {metadata['random_seed']}")
    print("")
    print("Row counts:")
    print(metadata["row_counts"])
    print("")
    print("Label distribution:")
    print(metadata["label_distribution"]["feature_data"])
    print("")
    print("Structural features:")
    for feature in metadata["structural_features"]:
        print(f"- {feature}")
    print("")
    print(f"Saved metadata to: {FEATURE_METADATA}")


def main():
    """
    Description:
        Runs the full metadata generation process. It loads the raw, parsed,
        and thread-level feature datasets, builds metadata, saves the metadata
        JSON file, and prints a summary.

    Input:
        No direct function input.
        Reads paths from config.py.

    Output:
        Creates feature_extraction_metadata.json and prints a summary.
    """
    raw_df = load_csv(RAW_DATA)
    parsed_df = load_csv(PARSED_DATA)
    features_df = load_csv(PROCESSED_DATA)

    metadata = build_metadata(raw_df, parsed_df, features_df)

    save_metadata(metadata, FEATURE_METADATA)

    print_metadata_summary(metadata)


if __name__ == "__main__":
    main()