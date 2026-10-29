"""
calibration_five_models.py

Description:
    Calibrates the scam probabilities of the five tuned COMP903 models and
    compares how trustworthy those probabilities are (RQ2).

    Each tuned model is tested in three versions:
    - uncalibrated: the model's own probabilities
    - sigmoid: probabilities adjusted with a fitted S-shaped curve (Platt scaling)
    - isotonic: probabilities adjusted with a fitted step-shaped curve

    Every version is scored on the same 50 repeated cross-validation folds
    used for the significance tests (REPEATED_CV_SEED), on the training set
    only. Inside each fold, calibration is learned with an inner 5-fold
    split of that fold's training part, so the calibration never sees the
    fold it is scored on.

    The main measure is the Brier score (lower is better). For each model,
    the calibration method with the lowest mean Brier score is taken as its
    calibrated version. Each new model's calibrated version is then compared
    with calibrated Logistic Regression using the corrected resampled t-test
    with Holm correction.

    Finally, the COMP902 risk bands (low < 0.26, medium 0.26 to 0.54,
    high >= 0.54) are re-checked using out-of-fold probabilities from the
    best calibrated model.

Inputs:
    - data/processed/scam_dialogue_thread_features.csv
    - data/processed/split_indices.json
    - outputs/reports/selected_hyperparams_five_models.json

Outputs:
    - outputs/reports/calibration_fold_results_five_models.csv
    - outputs/reports/calibration_results_five_models.csv
    - outputs/reports/risk_band_check_five_models.csv
    - outputs/reports/calibration_report_five_models.md

Run:
    - python src/evaluation/calibration_five_models.py

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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import (
    CALIBRATION_FOLD_RESULTS,
    CALIBRATION_REPORT_FIVE_MODELS,
    CALIBRATION_RESULTS_FIVE_MODELS,
    COMP902_RISK_THRESHOLDS,
    CV_FOLDS,
    CV_REPEATS,
    FEATURE_COLUMNS,
    MODEL_NAMES,
    NEW_MODELS,
    PROCESSED_DATA,
    REFERENCE_MODEL,
    REPEATED_CV_SEED,
    RISK_BAND_CHECK,
    SELECTED_HYPERPARAMS_FIVE_MODELS,
    SIGNIFICANCE_ALPHA,
    SPLIT_INDICES,
    TARGET_COLUMN,
)
from src.evaluation.significance_tests import corrected_resampled_ttest, holm_correction
from src.models.model_registry import build_models

CALIBRATION_METHODS = ["uncalibrated", "sigmoid", "isotonic"]

SCORING = {
    "brier_score": "neg_brier_score",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


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
        Returns X_train and y_train.
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


def build_version(model, method):
    """
    Description:
        Returns the model itself (uncalibrated) or wrapped in a calibrator.
        The calibrator learns its adjustment with an inner stratified 5-fold
        split of whatever training data it is given.

    Input:
        model: Tuned pipeline from the model registry.
        method: "uncalibrated", "sigmoid" or "isotonic".

    Output:
        Returns an untrained scikit-learn estimator.
    """
    if method == "uncalibrated":
        return model

    inner_cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=REPEATED_CV_SEED)
    return CalibratedClassifierCV(estimator=model, method=method, cv=inner_cv)


def score_all_versions(models, X_train, y_train):
    """
    Description:
        Scores every model in every calibration version on the same 50
        repeated cross-validation folds.

    Input:
        models: Dictionary of tuned pipelines.
        X_train: Training feature matrix.
        y_train: Training target labels.

    Output:
        Returns a DataFrame with one row per model, method and fold.
    """
    splitter = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=REPEATED_CV_SEED
    )

    rows = []

    for model_name, model in models.items():
        for method in CALIBRATION_METHODS:
            print(f"Scoring {model_name} ({method})...")

            scores = cross_validate(
                estimator=build_version(model, method),
                X=X_train,
                y=y_train,
                cv=splitter,
                scoring=SCORING,
                n_jobs=-1,
            )

            for fold_index in range(CV_FOLDS * CV_REPEATS):
                rows.append(
                    {
                        "model": model_name,
                        "method": method,
                        "fold_index": fold_index + 1,
                        "brier_score": -scores["test_brier_score"][fold_index],
                        "f1": scores["test_f1"][fold_index],
                        "roc_auc": scores["test_roc_auc"][fold_index],
                    }
                )

    return pd.DataFrame(rows)


def summarise(fold_df):
    """
    Description:
        Averages the fold results and picks each model's best calibration
        method by mean Brier score.

    Input:
        fold_df: DataFrame of per-fold results.

    Output:
        Returns the summary DataFrame and a dictionary of model -> best method.
    """
    summary_df = (
        fold_df.groupby(["model", "method"])[["brier_score", "f1", "roc_auc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_df.columns = ["_".join(col).strip("_") for col in summary_df.columns]

    calibrated_df = summary_df[summary_df["method"] != "uncalibrated"]
    best_methods = {
        model_name: group.loc[group["brier_score_mean"].idxmin(), "method"]
        for model_name, group in calibrated_df.groupby("model")
    }

    summary_df["selected"] = summary_df.apply(
        lambda row: best_methods.get(row["model"]) == row["method"], axis=1
    )

    return summary_df, best_methods


def test_calibrated_models(fold_df, best_methods):
    """
    Description:
        Compares each new model's calibrated Brier score with calibrated
        Logistic Regression on the 50 paired folds.

    Input:
        fold_df: DataFrame of per-fold results.
        best_methods: Dictionary of model -> selected calibration method.

    Output:
        Returns a DataFrame with one row per new model.
    """
    def brier_for(model_name):
        rows = fold_df[(fold_df["model"] == model_name) & (fold_df["method"] == best_methods[model_name])]
        return rows.sort_values("fold_index")["brier_score"].to_numpy()

    reference = brier_for(REFERENCE_MODEL)
    rows = []

    for model_name in NEW_MODELS:
        differences = brier_for(model_name) - reference
        result = corrected_resampled_ttest(differences)
        rows.append(
            {
                "model": model_name,
                "method": best_methods[model_name],
                "reference_method": best_methods[REFERENCE_MODEL],
                "model_brier": brier_for(model_name).mean(),
                "reference_brier": reference.mean(),
                **result,
                "folds_model_better": int((differences < 0).sum()),
            }
        )

    tests_df = pd.DataFrame(rows)
    tests_df["p_value_holm"] = holm_correction(tests_df["p_value"].tolist())
    tests_df["significant"] = tests_df["p_value_holm"] < SIGNIFICANCE_ALPHA

    return tests_df


def check_risk_bands(models, best_methods, summary_df, X_train, y_train):
    """
    Description:
        Re-checks the COMP902 risk bands using out-of-fold calibrated
        probabilities from the model with the lowest calibrated Brier score.

    Input:
        models: Dictionary of tuned pipelines.
        best_methods: Dictionary of model -> selected calibration method.
        summary_df: Summary DataFrame from summarise().
        X_train: Training feature matrix.
        y_train: Training target labels.

    Output:
        Returns the band table, the model name and its method.
    """
    selected_df = summary_df[summary_df["selected"]]
    best_row = selected_df.loc[selected_df["brier_score_mean"].idxmin()]
    model_name, method = best_row["model"], best_row["method"]

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=REPEATED_CV_SEED)
    probabilities = cross_val_predict(
        build_version(models[model_name], method), X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]

    medium, high = COMP902_RISK_THRESHOLDS["medium"], COMP902_RISK_THRESHOLDS["high"]
    bands = np.where(probabilities >= high, "High", np.where(probabilities >= medium, "Medium", "Low"))
    y = y_train.to_numpy()

    rows = []
    for band, low_edge, high_edge in [("Low", 0.0, medium), ("Medium", medium, high), ("High", high, 1.0)]:
        in_band = bands == band
        rows.append(
            {
                "risk_band": band,
                "probability_range": f"{low_edge:.2f} to {high_edge:.2f}",
                "conversations": int(in_band.sum()),
                "scams": int(y[in_band].sum()),
                "scam_rate": float(y[in_band].mean()) if in_band.any() else float("nan"),
            }
        )

    flagged_medium_or_high = probabilities >= medium
    flagged_high = probabilities >= high
    band_df = pd.DataFrame(rows)
    band_df.attrs["recall_at_medium"] = float(flagged_medium_or_high[y == 1].mean())
    band_df.attrs["precision_at_high"] = float(y[flagged_high].mean()) if flagged_high.any() else float("nan")

    return band_df, model_name, method


def build_report(summary_df, tests_df, band_df, band_model, band_method):
    """
    Description:
        Builds the readable markdown calibration report.

    Output:
        Returns the report as a single string.
    """
    lines = [
        "# Calibration Report (Five Models)",
        "",
        "## Method",
        "",
        "- Three versions of each tuned model: uncalibrated, sigmoid and isotonic calibration.",
        f"- Scored on the same {CV_FOLDS * CV_REPEATS} repeated cross-validation folds as the "
        "significance tests, on the training set only.",
        "- Calibration was learned inside each fold with an inner 5-fold split, so it never saw "
        "the fold it was scored on.",
        "- Each model's calibrated version is the method with its lowest mean Brier score.",
        "- The held-out test set was not used.",
        "",
        "## Brier Score by Model and Method (lower is better)",
        "",
        "| Model | Uncalibrated | Sigmoid | Isotonic | Selected method |",
        "|---|---:|---:|---:|---|",
    ]

    for model_name in MODEL_NAMES:
        model_rows = summary_df[summary_df["model"] == model_name].set_index("method")
        cells = [f"{model_rows.loc[m, 'brier_score_mean']:.4f}" for m in CALIBRATION_METHODS]
        selected = model_rows[model_rows["selected"]].index[0]
        lines.append(f"| {model_name} | " + " | ".join(cells) + f" | {selected} |")

    lines += [
        "",
        f"## RQ2: Calibrated Models Compared with Calibrated {REFERENCE_MODEL}",
        "",
        f"Corrected resampled t-test on the {CV_FOLDS * CV_REPEATS} paired folds, Holm-corrected "
        f"(alpha = {SIGNIFICANCE_ALPHA}). A negative difference means a lower (better) Brier score.",
        "",
        "| Model | Method | Brier | Difference | 95% CI | Folds better | Holm p-value | Significant |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]

    for _, row in tests_df.iterrows():
        lines.append(
            f"| {row['model']} | {row['method']} | {row['model_brier']:.4f} | {row['mean_difference']:+.4f} | "
            f"{row['ci_low']:+.4f} to {row['ci_high']:+.4f} | {row['folds_model_better']}/{CV_FOLDS * CV_REPEATS} | "
            f"{row['p_value_holm']:.4f} | {'Yes' if row['significant'] else 'No'} |"
        )

    lines += [
        "",
        f"Calibrated {REFERENCE_MODEL} ({tests_df['reference_method'].iloc[0]}) Brier score: "
        f"{tests_df['reference_brier'].iloc[0]:.4f}.",
        "",
        "Note: the selected method for each model was chosen on the same folds used for this test, "
        "which can make the selected versions look slightly better than they are.",
        "",
        "## COMP902 Risk Bands Re-checked",
        "",
        f"Using out-of-fold probabilities from the best calibrated model: {band_model} ({band_method}).",
        "",
        "| Risk band | Probability range | Conversations | Scams | Scam rate |",
        "|---|---|---:|---:|---:|",
    ]

    for _, row in band_df.iterrows():
        lines.append(
            f"| {row['risk_band']} | {row['probability_range']} | {row['conversations']} | "
            f"{row['scams']} | {row['scam_rate']:.3f} |"
        )

    lines += [
        "",
        f"- Share of scams flagged Medium or High (recall at {COMP902_RISK_THRESHOLDS['medium']}): "
        f"{band_df.attrs['recall_at_medium']:.3f} (COMP902 target: at least 0.90)",
        f"- Share of High-band conversations that are scams (precision at {COMP902_RISK_THRESHOLDS['high']}): "
        f"{band_df.attrs['precision_at_high']:.3f} (COMP902 target: at least 0.90)",
        "",
        "## Reproducibility Note",
        "",
        "Generated by `calibration_five_models.py`.",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
    ]

    return "\n".join(lines) + "\n"


def main():
    """
    Description:
        Scores all calibration versions, selects each model's calibration
        method, tests the calibrated models against Logistic Regression,
        re-checks the risk bands and saves all outputs.
    """
    X_train, y_train = load_training_data()
    models = load_tuned_models()

    fold_df = score_all_versions(models, X_train, y_train)
    summary_df, best_methods = summarise(fold_df)
    tests_df = test_calibrated_models(fold_df, best_methods)
    band_df, band_model, band_method = check_risk_bands(models, best_methods, summary_df, X_train, y_train)

    CALIBRATION_FOLD_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(CALIBRATION_FOLD_RESULTS, index=False)
    summary_df.to_csv(CALIBRATION_RESULTS_FIVE_MODELS, index=False)
    band_df.to_csv(RISK_BAND_CHECK, index=False)
    CALIBRATION_REPORT_FIVE_MODELS.write_text(
        build_report(summary_df, tests_df, band_df, band_model, band_method), encoding="utf-8"
    )

    print("")
    print("Calibrated Brier scores (lower is better)")
    print("=" * 41)
    for model_name in MODEL_NAMES:
        row = summary_df[(summary_df["model"] == model_name) & summary_df["selected"]].iloc[0]
        uncal = summary_df[(summary_df["model"] == model_name) & (summary_df["method"] == "uncalibrated")].iloc[0]
        print(
            f"{model_name:22s} uncalibrated {uncal['brier_score_mean']:.4f} -> "
            f"{row['method']:8s} {row['brier_score_mean']:.4f}"
        )

    print("")
    print(f"RQ2: calibrated models vs calibrated {REFERENCE_MODEL} (Holm-corrected)")
    for _, row in tests_df.iterrows():
        print(
            f"{row['model']:22s} diff {row['mean_difference']:+.4f} | folds better "
            f"{row['folds_model_better']}/50 | Holm p {row['p_value_holm']:.4f} | "
            f"{'SIGNIFICANT' if row['significant'] else 'not significant'}"
        )

    print("")
    print(f"Risk bands re-checked with {band_model} ({band_method}): "
          f"recall at medium {band_df.attrs['recall_at_medium']:.3f}, "
          f"precision at high {band_df.attrs['precision_at_high']:.3f}")
    print(f"Saved report to: {CALIBRATION_REPORT_FIVE_MODELS}")


if __name__ == "__main__":
    main()