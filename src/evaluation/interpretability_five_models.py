"""
interpretability_five_models.py

Description:
    Produces simple interpretability outputs for the three new COMP903
    models, so each model's behaviour can be explained in the final report.
    Each tuned model is trained once on all training rows.

    - Random Forest: feature importance, showing how much each feature
      helped the trees separate scam from non-scam conversations.
    - Gaussian Naive Bayes: the average and spread of each feature for scam
      and non-scam conversations (what the model "learned"), plus a
      separation score showing how far apart the two classes are.
    - k-Nearest Neighbours: worked examples showing, for selected training
      conversations, the most similar other conversations and how they
      voted.

    The COMP902 models (Logistic Regression and Decision Tree) were
    explained in COMP902 and are not repeated here.

Inputs:
    - data/processed/scam_dialogue_thread_features.csv
    - data/processed/split_indices.json
    - outputs/reports/selected_hyperparams_five_models.json

Outputs:
    - outputs/reports/rf_feature_importance.csv
    - outputs/reports/nb_class_statistics.csv
    - outputs/reports/knn_neighbour_examples.csv
    - outputs/reports/interpretability_report_five_models.md

Run:
    - python src/evaluation/interpretability_five_models.py

Important:
    - Only train_row_indices from split_indices.json are used.
    - The held-out test set is not used in this script.
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
    INTERPRETABILITY_REPORT_FIVE_MODELS,
    KNN_NEIGHBOUR_EXAMPLES,
    NB_CLASS_STATISTICS,
    PROCESSED_DATA,
    RF_FEATURE_IMPORTANCE,
    SELECTED_HYPERPARAMS_FIVE_MODELS,
    SPLIT_INDICES,
    TARGET_COLUMN,
)
from src.models.model_registry import build_model

CLASS_NAMES = {0: "non-scam", 1: "scam"}


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
        training rows only (with thread_id and type for the k-NN examples).

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


def train_tuned_model(model_name, train_df):
    """
    Description:
        Builds one model with its grid search settings and trains it on all
        training rows.

    Input:
        model_name: Name of the model.
        train_df: Training rows.

    Output:
        Returns the trained pipeline.
    """
    selected = load_json(SELECTED_HYPERPARAMS_FIVE_MODELS, "Selected settings file")["selected_models"]
    model = build_model(model_name, selected[model_name]["best_params"])
    return model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])


def explain_random_forest(train_df):
    """
    Description:
        Extracts Random Forest feature importance: the share of the total
        improvement in the trees' splits that came from each feature.

    Input:
        train_df: Training rows.

    Output:
        Returns a DataFrame ranked from most to least important.
    """
    model = train_tuned_model("Random Forest", train_df)
    forest = model.named_steps["model"]

    importance_df = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": forest.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance_df["rank"] = range(1, len(importance_df) + 1)

    return importance_df.reset_index(drop=True)


def explain_naive_bayes(train_df):
    """
    Description:
        Extracts what Gaussian Naive Bayes learned: the average and standard
        deviation of each feature for each class. A separation score
        (difference between the class averages divided by their combined
        spread) shows how clearly each feature tells the classes apart.

    Input:
        train_df: Training rows.

    Output:
        Returns a DataFrame with one row per feature.
    """
    model = train_tuned_model("Gaussian Naive Bayes", train_df)
    nb = model.named_steps["model"]

    means = pd.DataFrame(nb.theta_, columns=FEATURE_COLUMNS, index=nb.classes_)
    stds = pd.DataFrame(np.sqrt(nb.var_), columns=FEATURE_COLUMNS, index=nb.classes_)

    rows = []
    for feature in FEATURE_COLUMNS:
        pooled_std = np.sqrt((stds.loc[0, feature] ** 2 + stds.loc[1, feature] ** 2) / 2)
        separation = abs(means.loc[1, feature] - means.loc[0, feature]) / pooled_std if pooled_std else 0.0
        rows.append(
            {
                "feature": feature,
                "non_scam_mean": means.loc[0, feature],
                "non_scam_std": stds.loc[0, feature],
                "scam_mean": means.loc[1, feature],
                "scam_std": stds.loc[1, feature],
                "separation": separation,
            }
        )

    stats_df = pd.DataFrame(rows).sort_values("separation", ascending=False).reset_index(drop=True)
    stats_df["rank"] = range(1, len(stats_df) + 1)

    return stats_df


def choose_knn_examples(train_df, vote_shares, predictions):
    """
    Description:
        Picks four training conversations that illustrate k-NN behaviour:
        a confident correct scam, a confident correct non-scam, the most
        borderline case, and a missed tech support scam (if any).

    Input:
        train_df: Training rows.
        vote_shares: Share of neighbours voting scam, per conversation.
        predictions: Predicted class per conversation.

    Output:
        Returns a list of (label, row position) pairs.
    """
    labels = train_df[TARGET_COLUMN].to_numpy()
    examples = []

    correct_scam = np.where((labels == 1) & (predictions == 1))[0]
    examples.append(("Confident correct scam", int(correct_scam[np.argmax(vote_shares[correct_scam])])))

    correct_non_scam = np.where((labels == 0) & (predictions == 0))[0]
    examples.append(("Confident correct non-scam", int(correct_non_scam[np.argmin(vote_shares[correct_non_scam])])))

    examples.append(("Most borderline case", int(np.argmin(np.abs(vote_shares - 0.5)))))

    missed_support = np.where(
        (labels == 1) & (predictions == 0) & (train_df["type"].to_numpy() == "support")
    )[0]
    if len(missed_support):
        examples.append(("Missed tech support scam", int(missed_support[np.argmin(vote_shares[missed_support])])))

    return examples


def explain_knn(train_df):
    """
    Description:
        For selected training conversations, finds their nearest neighbours
        among the other training conversations (excluding themselves) and
        shows how those neighbours voted.

    Input:
        train_df: Training rows.

    Output:
        Returns a DataFrame with one row per example conversation and the
        number of neighbours used.
    """
    model = train_tuned_model("k-Nearest Neighbours", train_df)
    knn = model.named_steps["model"]
    k = knn.n_neighbors

    X_scaled = model.named_steps["scaler"].transform(train_df[FEATURE_COLUMNS])

    # Ask for k + 1 neighbours and drop the first, which is the conversation itself
    _, neighbour_positions = knn.kneighbors(X_scaled, n_neighbors=k + 1)
    neighbour_positions = neighbour_positions[:, 1:]

    labels = train_df[TARGET_COLUMN].to_numpy()
    vote_shares = labels[neighbour_positions].mean(axis=1)
    predictions = (vote_shares > 0.5).astype(int)

    rows = []
    for label, position in choose_knn_examples(train_df, vote_shares, predictions):
        neighbours = train_df.iloc[neighbour_positions[position]]
        conversation = train_df.iloc[position]

        row = {
            "example": label,
            "thread_id": conversation["thread_id"],
            "type": conversation["type"],
            "true_label": CLASS_NAMES[int(conversation[TARGET_COLUMN])],
            "neighbours_voting_scam": int(labels[neighbour_positions[position]].sum()),
            "neighbours": k,
            "predicted_label": CLASS_NAMES[int(predictions[position])],
            "neighbour_types": ", ".join(
                f"{t} {c}" for t, c in neighbours["type"].value_counts().items()
            ),
        }
        for feature in FEATURE_COLUMNS:
            row[f"{feature} (conversation)"] = conversation[feature]
            row[f"{feature} (neighbour mean)"] = round(neighbours[feature].mean(), 3)
        rows.append(row)

    return pd.DataFrame(rows), k


def build_report(rf_df, nb_df, knn_df, k):
    """
    Description:
        Builds the readable markdown interpretability report.

    Output:
        Returns the report as a single string.
    """
    lines = [
        "# Interpretability Report (New Models)",
        "",
        "Each tuned model was trained once on all 1,280 training conversations. The held-out test "
        "set was not used.",
        "",
        "## Random Forest: Feature Importance",
        "",
        "Share of the trees' total splitting improvement that came from each feature (sums to 1).",
        "",
        "| Rank | Feature | Importance |",
        "|---:|---|---:|",
    ]
    for _, row in rf_df.iterrows():
        lines.append(f"| {row['rank']} | {row['feature']} | {row['importance']:.3f} |")

    lines += [
        "",
        "## Gaussian Naive Bayes: What the Model Learned",
        "",
        "Average (and standard deviation) of each feature for each class. Separation is the gap "
        "between the class averages divided by their combined spread; a higher value means the "
        "feature tells scam and non-scam conversations apart more clearly.",
        "",
        "| Rank | Feature | Non-scam mean (sd) | Scam mean (sd) | Separation |",
        "|---:|---|---:|---:|---:|",
    ]
    for _, row in nb_df.iterrows():
        lines.append(
            f"| {row['rank']} | {row['feature']} | {row['non_scam_mean']:.3f} ({row['non_scam_std']:.3f}) | "
            f"{row['scam_mean']:.3f} ({row['scam_std']:.3f}) | {row['separation']:.2f} |"
        )

    lines += [
        "",
        f"## k-Nearest Neighbours: Worked Examples ({k} neighbours)",
        "",
        "For each example, the model finds the most similar other training conversations and takes "
        "a majority vote.",
        "",
        "| Example | Type | True label | Neighbours voting scam | Prediction | Neighbour types |",
        "|---|---|---|---:|---|---|",
    ]
    for _, row in knn_df.iterrows():
        lines.append(
            f"| {row['example']} | {row['type']} | {row['true_label']} | "
            f"{row['neighbours_voting_scam']}/{row['neighbours']} | {row['predicted_label']} | "
            f"{row['neighbour_types']} |"
        )

    lines += [
        "",
        "Feature values for each example compared with the average of its neighbours:",
        "",
        "| Example | " + " | ".join(FEATURE_COLUMNS) + " |",
        "|---|" + "---|" * len(FEATURE_COLUMNS),
    ]
    for _, row in knn_df.iterrows():
        cells = [
            f"{row[f'{f} (conversation)']:.2f} (nbrs {row[f'{f} (neighbour mean)']:.2f})"
            for f in FEATURE_COLUMNS
        ]
        lines.append(f"| {row['example']} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Reproducibility Note",
        "",
        "Generated by `interpretability_five_models.py`.",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
    ]

    return "\n".join(lines) + "\n"


def main():
    """
    Description:
        Produces the interpretability outputs for Random Forest, Gaussian
        Naive Bayes and k-Nearest Neighbours, and saves them.
    """
    train_df = load_training_rows()

    rf_df = explain_random_forest(train_df)
    nb_df = explain_naive_bayes(train_df)
    knn_df, k = explain_knn(train_df)

    RF_FEATURE_IMPORTANCE.parent.mkdir(parents=True, exist_ok=True)
    rf_df.to_csv(RF_FEATURE_IMPORTANCE, index=False)
    nb_df.to_csv(NB_CLASS_STATISTICS, index=False)
    knn_df.to_csv(KNN_NEIGHBOUR_EXAMPLES, index=False)
    INTERPRETABILITY_REPORT_FIVE_MODELS.write_text(build_report(rf_df, nb_df, knn_df, k), encoding="utf-8")

    print("Random Forest feature importance")
    for _, row in rf_df.iterrows():
        print(f"  {row['rank']}. {row['feature']:24s} {row['importance']:.3f}")

    print("\nGaussian Naive Bayes separation (higher = clearer gap between classes)")
    for _, row in nb_df.iterrows():
        print(
            f"  {row['rank']}. {row['feature']:24s} non-scam {row['non_scam_mean']:.2f} | "
            f"scam {row['scam_mean']:.2f} | separation {row['separation']:.2f}"
        )

    print(f"\nk-Nearest Neighbours worked examples ({k} neighbours)")
    for _, row in knn_df.iterrows():
        print(
            f"  {row['example']:28s} {row['type']:13s} true {row['true_label']:9s} "
            f"votes {row['neighbours_voting_scam']}/{row['neighbours']} -> {row['predicted_label']}"
        )

    print(f"\nSaved report to: {INTERPRETABILITY_REPORT_FIVE_MODELS}")


if __name__ == "__main__":
    main()