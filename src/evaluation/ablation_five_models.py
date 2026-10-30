"""
ablation_five_models.py

Description:
    Runs a leave-one-feature-out ablation test for all five tuned COMP903
    models to answer RQ3: is the COMP902 feature ranking (turn_count first,
    aggregate_money_terms second) the same across every model?

    For each model, the full four-feature version and four reduced versions
    (each with one feature removed) are scored on the same 50 repeated
    cross-validation folds used for the significance tests, on the training
    set only. The drop in F1 when a feature is removed shows how much the
    model relies on that feature: the bigger the drop, the more important
    the feature.

    For each model and feature, the corrected resampled t-test (with Holm
    correction across the four features) checks whether removing the
    feature makes a real difference.

Inputs:
    - data/processed/scam_dialogue_thread_features.csv
    - data/processed/split_indices.json
    - outputs/reports/selected_hyperparams_five_models.json

Outputs:
    - outputs/reports/ablation_fold_results_five_models.csv
    - outputs/reports/ablation_results_five_models.csv
    - outputs/reports/ablation_report_five_models.md

Run:
    - python src/evaluation/ablation_five_models.py

Important:
    - Only train_row_indices from split_indices.json are used.
    - The held-out test set is not used in this script.
    - Model settings are the ones selected by grid search on all four
      features; they are not re-tuned for the reduced feature sets.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import (
    ABLATION_FOLD_RESULTS,
    ABLATION_REPORT_FIVE_MODELS,
    ABLATION_RESULTS_FIVE_MODELS,
    COMP902_FEATURE_RANKING,
    CV_FOLDS,
    CV_REPEATS,
    FEATURE_COLUMNS,
    MODEL_NAMES,
    PRIMARY_METRIC,
    PROCESSED_DATA,
    REPEATED_CV_SEED,
    SELECTED_HYPERPARAMS_FIVE_MODELS,
    SIGNIFICANCE_ALPHA,
    SPLIT_INDICES,
    TARGET_COLUMN,
)
from src.evaluation.significance_tests import corrected_resampled_ttest, holm_correction
from src.models.model_registry import build_models

FULL_SET = "none (all four features)"


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


def load_training_data():
    """
    Description:
        Loads the features and saved split, checks them, and returns the
        training rows only.

    Output:
        Returns X_train (all four features) and y_train.
    """
    if not Path(PROCESSED_DATA).exists():
        raise FileNotFoundError(f"Feature file not found: {PROCESSED_DATA}")

    features_df = pd.read_csv(PROCESSED_DATA)
    split_payload = load_json(SPLIT_INDICES, "Split index file")

    overlap = set(split_payload["train_row_indices"]) & set(split_payload["test_row_indices"])
    if overlap:
        raise ValueError(f"{len(overlap)} rows appear in both the training and test split")

    train_df = features_df.iloc[split_payload["train_row_indices"]]
    return train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]


def load_tuned_models():
    """
    Description:
        Builds the five models with the settings selected by grid search.

    Output:
        Returns a dictionary of model names and untrained tuned pipelines.
    """
    selected = load_json(SELECTED_HYPERPARAMS_FIVE_MODELS, "Selected settings file")["selected_models"]
    return build_models({name: selected[name]["best_params"] for name in MODEL_NAMES})


def run_ablation(models, X_train, y_train):
    """
    Description:
        Scores each model with all four features and with each feature
        removed, on the same 50 repeated cross-validation folds.

    Input:
        models: Dictionary of tuned pipelines.
        X_train: Training feature matrix with all four features.
        y_train: Training target labels.

    Output:
        Returns a DataFrame with one row per model, removed feature and fold.
    """
    splitter = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=REPEATED_CV_SEED
    )

    feature_sets = {FULL_SET: FEATURE_COLUMNS}
    for feature in FEATURE_COLUMNS:
        feature_sets[feature] = [column for column in FEATURE_COLUMNS if column != feature]

    rows = []

    for model_name, model in models.items():
        print(f"Running ablation for {model_name}...")

        for removed_feature, columns in feature_sets.items():
            scores = cross_val_score(
                model, X_train[columns], y_train, cv=splitter, scoring=PRIMARY_METRIC, n_jobs=-1
            )

            for fold_index, score in enumerate(scores):
                rows.append(
                    {
                        "model": model_name,
                        "removed_feature": removed_feature,
                        "fold_index": fold_index + 1,
                        PRIMARY_METRIC: score,
                    }
                )

    return pd.DataFrame(rows)


def summarise_ablation(fold_df):
    """
    Description:
        Calculates each feature's F1 drop per model, tests whether the drop
        is real, and ranks the features within each model.

    Input:
        fold_df: DataFrame of per-fold ablation results.

    Output:
        Returns a DataFrame with one row per model and removed feature.
    """
    rows = []

    for model_name in MODEL_NAMES:
        model_df = fold_df[fold_df["model"] == model_name]
        full = model_df[model_df["removed_feature"] == FULL_SET].sort_values("fold_index")[PRIMARY_METRIC].to_numpy()

        model_rows = []
        for feature in FEATURE_COLUMNS:
            reduced = model_df[model_df["removed_feature"] == feature].sort_values("fold_index")[PRIMARY_METRIC].to_numpy()
            drops = full - reduced
            result = corrected_resampled_ttest(drops)

            model_rows.append(
                {
                    "model": model_name,
                    "removed_feature": feature,
                    "f1_full": full.mean(),
                    "f1_without_feature": reduced.mean(),
                    "f1_drop": result["mean_difference"],
                    "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"],
                    "folds_with_drop": int((drops > 0).sum()),
                    "p_value": result["p_value"],
                }
            )

        adjusted = holm_correction([row["p_value"] for row in model_rows])
        ranked = sorted(model_rows, key=lambda row: row["f1_drop"], reverse=True)

        for row, p_adjusted in zip(model_rows, adjusted):
            row["p_value_holm"] = p_adjusted
            row["significant"] = p_adjusted < SIGNIFICANCE_ALPHA
            row["rank"] = ranked.index(row) + 1
            rows.append(row)

    return pd.DataFrame(rows)


def check_comp902_ranking(summary_df):
    """
    Description:
        Checks the RQ3 hypothesis for each model: turn_count ranks first and
        aggregate_money_terms ranks second, as in COMP902.

    Input:
        summary_df: DataFrame returned by summarise_ablation().

    Output:
        Returns a DataFrame with one row per model and a boolean that is True
        if the hypothesis holds for every model.
    """
    rows = []

    for model_name in MODEL_NAMES:
        ranked = (
            summary_df[summary_df["model"] == model_name]
            .sort_values("rank")["removed_feature"]
            .tolist()
        )
        rows.append(
            {
                "model": model_name,
                "ranking": " > ".join(ranked),
                "top_two_match_comp902": ranked[:2] == COMP902_FEATURE_RANKING[:2],
                "full_ranking_matches_comp902": ranked == COMP902_FEATURE_RANKING,
            }
        )

    ranking_df = pd.DataFrame(rows)
    return ranking_df, bool(ranking_df["top_two_match_comp902"].all())


def build_report(summary_df, ranking_df, hypothesis_holds):
    """
    Description:
        Builds the readable markdown ablation report.

    Output:
        Returns the report as a single string.
    """
    total_folds = CV_FOLDS * CV_REPEATS

    lines = [
        "# Ablation Report (Five Models)",
        "",
        "## Method",
        "",
        "- Each tuned model was scored with all four features, then with each feature removed in turn.",
        f"- All versions used the same {total_folds} repeated cross-validation folds on the training set only.",
        "- F1 drop = F1 with all four features minus F1 without the feature. A bigger drop means the "
        "model relies more on that feature.",
        "- Each drop was tested with the corrected resampled t-test, Holm-corrected across the four "
        f"features within each model (alpha = {SIGNIFICANCE_ALPHA}).",
        "- Model settings were not re-tuned for the reduced feature sets.",
        "- The held-out test set was not used.",
        "",
        "## F1 Drop When Each Feature Is Removed",
        "",
        "| Model | " + " | ".join(FEATURE_COLUMNS) + " |",
        "|---|" + "---:|" * len(FEATURE_COLUMNS),
    ]

    for model_name in MODEL_NAMES:
        model_df = summary_df[summary_df["model"] == model_name].set_index("removed_feature")
        cells = []
        for feature in FEATURE_COLUMNS:
            row = model_df.loc[feature]
            marker = "*" if row["significant"] else ""
            cells.append(f"{row['f1_drop']:+.4f}{marker} (#{int(row['rank'])})")
        lines.append(f"| {model_name} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "`*` = significant drop after Holm correction. `(#n)` = the feature's rank within that model.",
        "",
        "## RQ3: Is the COMP902 Ranking the Same?",
        "",
        f"COMP902 ranking: {' > '.join(COMP902_FEATURE_RANKING)}.",
        "",
        "| Model | Ranking in COMP903 | Top two match COMP902 | Full ranking matches COMP902 |",
        "|---|---|---|---|",
    ]

    for _, row in ranking_df.iterrows():
        lines.append(
            f"| {row['model']} | {row['ranking']} | "
            f"{'Yes' if row['top_two_match_comp902'] else 'No'} | "
            f"{'Yes' if row['full_ranking_matches_comp902'] else 'No'} |"
        )

    verdict = (
        "Result: turn_count ranked first and aggregate_money_terms second in all five models, "
        "so the RQ3 null hypothesis (the COMP902 ranking holds) is supported."
        if hypothesis_holds
        else "Result: the COMP902 top-two ranking did not hold in every model, so the RQ3 null "
        "hypothesis is rejected for the models marked No."
    )

    lines += [
        "",
        verdict,
        "",
        "## Reproducibility Note",
        "",
        "Generated by `ablation_five_models.py`.",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
    ]

    return "\n".join(lines) + "\n"


def main():
    """
    Description:
        Runs the ablation for all five models, summarises and tests the
        drops, checks the RQ3 hypothesis and saves all outputs.
    """
    X_train, y_train = load_training_data()
    models = load_tuned_models()

    fold_df = run_ablation(models, X_train, y_train)
    summary_df = summarise_ablation(fold_df)
    ranking_df, hypothesis_holds = check_comp902_ranking(summary_df)

    ABLATION_FOLD_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(ABLATION_FOLD_RESULTS, index=False)
    summary_df.to_csv(ABLATION_RESULTS_FIVE_MODELS, index=False)
    ABLATION_REPORT_FIVE_MODELS.write_text(
        build_report(summary_df, ranking_df, hypothesis_holds), encoding="utf-8"
    )

    print("")
    print("Feature ranking by F1 drop (training set, 50 folds)")
    print("=" * 51)
    for _, row in ranking_df.iterrows():
        print(f"{row['model']:22s} {row['ranking']}")

    print("")
    print(
        "RQ3: COMP902 top two (turn_count, aggregate_money_terms) hold in all five models: "
        f"{'YES' if hypothesis_holds else 'NO'}"
    )
    print(f"Saved report to: {ABLATION_REPORT_FIVE_MODELS}")


if __name__ == "__main__":
    main()