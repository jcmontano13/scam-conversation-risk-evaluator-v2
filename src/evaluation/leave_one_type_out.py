"""
leave_one_type_out.py

Description:
    Extension experiment (story 18). Tests whether the five tuned models can
    detect a type of scam they never saw during training, to show where the
    four structural features stop generalising.

    The training split contains four scam types (refund, reward, ssn,
    support). For each scam type and each model:

    - Unseen: the model is trained on all training conversations except that
      scam type, then asked to classify the held-out scam type. The share it
      flags as scam is its recall on an unseen scam type.

    - Seen (control): the same number of scam conversations is removed, but
      chosen at random from all four scam types, so the model still sees
      the scam type in training. Recall is measured on the removed
      conversations of that type. This is repeated LOTO_CONTROL_REPEATS
      times with different seeds and pooled.

    The control keeps the training size and class balance the same as in
    the unseen case, so the difference between the two recalls (the
    generalisation gap) reflects not having seen the scam type, rather than
    simply having less data.

Inputs:
    - data/processed/scam_dialogue_thread_features.csv
    - data/processed/split_indices.json
    - outputs/reports/selected_hyperparams_five_models.json

Outputs:
    - outputs/reports/leave_one_type_out_results.csv
    - outputs/reports/leave_one_type_out_report.md

Run:
    - python src/evaluation/leave_one_type_out.py

Important:
    - Only train_row_indices from split_indices.json are used. The held-out
      test set is not used.
    - Model settings are the ones selected by grid search and are not
      changed based on these results.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import (
    FEATURE_COLUMNS,
    LOTO_CONTROL_REPEATS,
    LOTO_REPORT,
    LOTO_RESULTS,
    MODEL_NAMES,
    PROCESSED_DATA,
    REPEATED_CV_SEED,
    SELECTED_HYPERPARAMS_FIVE_MODELS,
    SPLIT_INDICES,
    TARGET_COLUMN,
)
from src.models.model_registry import build_models


def load_json(file_path, description):
    """
    Description:
        Loads a JSON file.

    Input:
        file_path: Path to the JSON file.
        description: Short name of the file, used in the error message.

    Output:
        Returns the parsed JSON content.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"{description} not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_training_rows():
    """
    Description:
        Loads the features and saved split, checks them, and returns the
        training rows only.

    Output:
        Returns a DataFrame of training rows.
    """
    if not Path(PROCESSED_DATA).exists():
        raise FileNotFoundError(f"Feature file not found: {PROCESSED_DATA}")

    features_df = pd.read_csv(PROCESSED_DATA)
    split_payload = load_json(SPLIT_INDICES, "Split index file")

    overlap = set(split_payload["train_row_indices"]) & set(split_payload["test_row_indices"])
    if overlap:
        raise ValueError(f"{len(overlap)} rows appear in both the training and test split")

    return features_df.iloc[split_payload["train_row_indices"]].reset_index(drop=True)


def load_tuned_models():
    """
    Description:
        Builds the five models with the settings selected by grid search.

    Output:
        Returns a dictionary of model names and untrained tuned pipelines.
    """
    selected = load_json(SELECTED_HYPERPARAMS_FIVE_MODELS, "Selected settings file")["selected_models"]
    return build_models({name: selected[name]["best_params"] for name in MODEL_NAMES})


def recall_after_training(model, train_df, train_mask, evaluate_mask):
    """
    Description:
        Trains a model on the rows in train_mask and returns the share of
        the rows in evaluate_mask (all scams) that it flags as scam.

    Input:
        model: Untrained tuned pipeline.
        train_df: Training rows.
        train_mask: Boolean array of rows to train on.
        evaluate_mask: Boolean array of scam rows to evaluate.

    Output:
        Returns a tuple of (number flagged as scam, number evaluated).
    """
    model.fit(train_df.loc[train_mask, FEATURE_COLUMNS], train_df.loc[train_mask, TARGET_COLUMN])
    predictions = model.predict(train_df.loc[evaluate_mask, FEATURE_COLUMNS])
    return int(predictions.sum()), int(evaluate_mask.sum())


def run_experiment(models, train_df):
    """
    Description:
        Runs the unseen and seen (control) conditions for every scam type
        and every model.

    Input:
        models: Dictionary of tuned pipelines.
        train_df: Training rows.

    Output:
        Returns a DataFrame with one row per model and scam type.
    """
    is_scam = (train_df[TARGET_COLUMN] == 1).to_numpy()
    types = train_df["type"].to_numpy()
    scam_positions = np.where(is_scam)[0]
    scam_types = sorted(train_df.loc[is_scam, "type"].unique())

    rows = []

    for scam_type in scam_types:
        held_out = is_scam & (types == scam_type)
        n_removed = int(held_out.sum())
        print(f"Scam type '{scam_type}': {n_removed} training conversations")

        for model_name, model in models.items():
            # Unseen: train without this scam type, evaluate on it
            caught_unseen, total_unseen = recall_after_training(model, train_df, ~held_out, held_out)

            # Seen (control): remove the same number of random scams from all types
            caught_seen, total_seen = 0, 0
            for repeat in range(LOTO_CONTROL_REPEATS):
                rng = np.random.default_rng(REPEATED_CV_SEED + repeat)
                removed = np.zeros(len(train_df), dtype=bool)
                removed[rng.choice(scam_positions, size=n_removed, replace=False)] = True

                evaluate = removed & (types == scam_type)
                if evaluate.sum() == 0:
                    continue

                caught, total = recall_after_training(model, train_df, ~removed, evaluate)
                caught_seen += caught
                total_seen += total

            recall_unseen = caught_unseen / total_unseen
            recall_seen = caught_seen / total_seen

            rows.append(
                {
                    "scam_type": scam_type,
                    "model": model_name,
                    "conversations_held_out": total_unseen,
                    "recall_seen": recall_seen,
                    "recall_unseen": recall_unseen,
                    "generalisation_gap": recall_seen - recall_unseen,
                    "control_evaluations": total_seen,
                }
            )

    return pd.DataFrame(rows)


def build_report(results_df):
    """
    Description:
        Builds the readable markdown report of the experiment.

    Output:
        Returns the report as a single string.
    """
    scam_types = list(dict.fromkeys(results_df["scam_type"]))

    def table(column, title, note):
        lines = [
            f"## {title}",
            "",
            note,
            "",
            "| Scam type | " + " | ".join(MODEL_NAMES) + " | Average |",
            "|---|" + "---:|" * (len(MODEL_NAMES) + 1),
        ]
        for scam_type in scam_types:
            values = results_df[results_df["scam_type"] == scam_type].set_index("model")[column]
            cells = [f"{values[m]:.3f}" for m in MODEL_NAMES]
            lines.append(f"| {scam_type} | " + " | ".join(cells) + f" | {values.mean():.3f} |")
        return lines + [""]

    average_gap = results_df.groupby("scam_type")["generalisation_gap"].mean()
    average_unseen = results_df.groupby("scam_type")["recall_unseen"].mean()
    hardest = average_unseen.idxmin()

    lines = [
        "# Leave-One-Scam-Type-Out Report (Extension)",
        "",
        "## Purpose",
        "",
        "Tests whether the five tuned models can detect a scam type they never saw during training, "
        "to show where the four structural features stop generalising. This experiment was added "
        "after the significance tests found no significant difference between the models (see the "
        "challenges and decisions log).",
        "",
        "## Method",
        "",
        "- **Unseen:** each model is trained on the training set without one scam type, then asked "
        "to classify that scam type. Recall = share flagged as scam.",
        "- **Seen (control):** the same number of scam conversations is removed at random from all "
        f"four types, so the scam type is still seen in training. Repeated {LOTO_CONTROL_REPEATS} "
        "times and pooled. This keeps training size and class balance the same as the unseen case.",
        "- **Generalisation gap** = seen recall minus unseen recall. A large gap means the model "
        "relied on having seen that scam type.",
        "- Training split only; the held-out test set was not used. Tuned settings were not changed.",
        "",
    ]

    lines += table("recall_seen", "Recall When the Scam Type Was Seen (control)", "Share of that scam type flagged as scam.")
    lines += table("recall_unseen", "Recall When the Scam Type Was Never Seen", "Share of that scam type flagged as scam.")
    lines += table("generalisation_gap", "Generalisation Gap (seen minus unseen)", "Positive = recall lost because the type was never seen.")

    lines += [
        "## Summary",
        "",
        f"- Hardest scam type to detect when unseen: **{hardest}** "
        f"(average recall {average_unseen[hardest]:.3f} across the five models).",
        f"- Largest average generalisation gap: **{average_gap.idxmax()}** ({average_gap.max():+.3f}).",
        f"- Smallest average generalisation gap: **{average_gap.idxmin()}** ({average_gap.min():+.3f}).",
        "",
        "## Limitation",
        "",
        "Only recall on the held-out scam type is measured. False alarms on non-scam conversations "
        "are not measured here, because every non-scam conversation in the training split was used "
        "for training in this experiment.",
        "",
        "## Reproducibility Note",
        "",
        "Generated by `leave_one_type_out.py`.",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
    ]

    return "\n".join(lines) + "\n"


def main():
    """
    Description:
        Runs the leave-one-scam-type-out experiment and saves the results
        and report.
    """
    train_df = load_training_rows()
    models = load_tuned_models()

    results_df = run_experiment(models, train_df)

    LOTO_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(LOTO_RESULTS, index=False)
    LOTO_REPORT.write_text(build_report(results_df), encoding="utf-8")

    print("")
    print("Recall on each scam type: seen in training -> never seen (gap)")
    print("=" * 62)
    for scam_type, group in results_df.groupby("scam_type"):
        print(f"{scam_type}")
        for _, row in group.iterrows():
            print(
                f"  {row['model']:22s} {row['recall_seen']:.3f} -> {row['recall_unseen']:.3f} "
                f"(gap {row['generalisation_gap']:+.3f})"
            )

    print(f"\nSaved report to: {LOTO_REPORT}")


if __name__ == "__main__":
    main()