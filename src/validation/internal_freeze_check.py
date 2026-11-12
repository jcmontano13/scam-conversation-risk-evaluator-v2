"""
internal_freeze_check.py

Description:
    Runs the internal freeze checklist for COMP903 before final report
    writing. COMP902 used a hand-written checklist; this version checks
    itself, so a "Complete" status is backed by evidence rather than by
    memory.

    The script makes three kinds of checks:
    1. Completeness: every expected output file exists and is not empty.
    2. Consistency: numbers that appear in more than one file agree, for
       example the test metrics match the saved test predictions, and the
       significance tests match the repeated cross-validation folds.
    3. Integrity: the held-out test set was not mixed with the training
       data, the repeated-CV seed differs from the tuning seed, the unit
       tests pass, and no figure is older than the results it shows.

    It then writes a markdown checklist with a PASS/FAIL line for every
    check, the frozen headline numbers for the final report, and the
    manual items that a script cannot check.

    Nothing is trained or changed: the script only reads.

Inputs:
    - config.py paths for all COMP903 outputs (stories 2 to 20)

Outputs:
    - outputs/reports/internal_freeze_checklist_five_models.md

Run:
    - python src/validation/internal_freeze_check.py
    - python src/validation/internal_freeze_check.py --skip-tests   (skip the pytest run)

Exit code:
    - 0 if every automated check passes, 1 otherwise.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import config  # noqa: E402

TOLERANCE = 1e-6

# Every output the final report relies on, by config name.
REQUIRED_FILES = {
    "Data": ["PROCESSED_DATA", "SPLIT_INDICES"],
    "Baseline and training (stories 2-6)": [
        "BASELINE_VERIFICATION_REPORT",
        "UNIT_TEST_LOG_MODELS",
        "CV_RESULTS_FIVE_MODELS",
        "CV_SUMMARY_REPORT_FIVE_MODELS",
    ],
    "Tuning and repeated CV (stories 7-10)": [
        "GRID_SEARCH_RESULTS_FIVE_MODELS",
        "SELECTED_HYPERPARAMS_FIVE_MODELS",
        "REPEATED_CV_RESULTS",
        "SIGNIFICANCE_TESTS",
        "SIGNIFICANCE_REPORT",
    ],
    "Held-out test (story 14)": [
        "TEST_RESULTS_FIVE_MODELS",
        "TEST_PREDICTIONS_FIVE_MODELS",
        "MCNEMAR_RESULTS",
        "TEST_EVALUATION_REPORT_FIVE_MODELS",
    ],
    "Calibration (story 15)": [
        "CALIBRATION_RESULTS_FIVE_MODELS",
        "CALIBRATION_FOLD_RESULTS",
        "RISK_BAND_CHECK",
        "CALIBRATION_REPORT_FIVE_MODELS",
    ],
    "Ablation and interpretability (stories 16-17)": [
        "ABLATION_RESULTS_FIVE_MODELS",
        "ABLATION_FOLD_RESULTS",
        "ABLATION_REPORT_FIVE_MODELS",
        "RF_FEATURE_IMPORTANCE",
        "NB_CLASS_STATISTICS",
        "KNN_NEIGHBOUR_EXAMPLES",
        "INTERPRETABILITY_REPORT_FIVE_MODELS",
    ],
    "Extension: leave-one-type-out (story 18)": ["LOTO_RESULTS", "LOTO_REPORT"],
    "Documentation": ["CHALLENGES_LOG"],
}

# Each figure from story 20 and the result files it is drawn from.
FIGURE_INPUTS = {
    "model_comparison_significance": ["SIGNIFICANCE_TESTS"],
    "test_confusion_matrices": ["TEST_RESULTS_FIVE_MODELS"],
    "roc_curves_test_set": ["TEST_PREDICTIONS_FIVE_MODELS"],
    "precision_recall_curves_test_set": ["TEST_PREDICTIONS_FIVE_MODELS"],
    "reliability_curves_test_set": ["TEST_PREDICTIONS_FIVE_MODELS"],
    "calibration_brier_comparison": ["CALIBRATION_RESULTS_FIVE_MODELS"],
    "ablation_f1_drop_heatmap": ["ABLATION_RESULTS_FIVE_MODELS"],
    "generalisation_recall": ["LOTO_RESULTS"],
}

MANUAL_ITEMS = [
    "Every user story on the project board (except 22-24) is in Done",
    "All branches are merged into main and `git status` shows a clean working tree",
    "README lists every script, run command and output (story 22 will do the final sync)",
    "Challenges log reviewed: every entry is a real problem or decision, with its evidence",
    "Challenges log entry D4 says the extension was added after the significance tests (not planned in the proposal)",
    "The frozen numbers below match the numbers already quoted in Progress Reports 1 and 2",
    "Freeze tag created: `git tag freeze-v1` and `git push origin freeze-v1`",
]


class Checklist:
    """
    Description:
        Collects check results as (section, check, passed, evidence) rows.
    """

    def __init__(self):
        self.rows = []

    def add(self, section, check, passed, evidence):
        self.rows.append({"section": section, "check": check, "passed": bool(passed), "evidence": evidence})
        print(f"[{'PASS' if passed else 'FAIL'}] {section}: {check}")

    def run(self, section, check, function):
        """
        Description:
            Runs one check function. A crash (for example a missing column)
            counts as a FAIL with the error as evidence, so one broken file
            never stops the rest of the checklist.
        """
        try:
            passed, evidence = function()
        except Exception as error:  # noqa: BLE001
            passed, evidence = False, f"Check could not run: {type(error).__name__}: {error}"
        self.add(section, check, passed, evidence)

    @property
    def failed(self):
        return [row for row in self.rows if not row["passed"]]


def path_of(name):
    """
    Description:
        Returns the path for a config name, or None if config.py does not
        define it.
    """
    return getattr(config, name, None)


def relative(path):
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- Completeness


def check_files(checklist):
    for section, names in REQUIRED_FILES.items():
        for name in names:
            path = path_of(name)
            if path is None:
                checklist.add(f"Files: {section}", name, False, "Not defined in config.py")
            elif not Path(path).exists():
                checklist.add(f"Files: {section}", name, False, f"Missing: {relative(path)}")
            elif Path(path).stat().st_size == 0:
                checklist.add(f"Files: {section}", name, False, f"Empty: {relative(path)}")
            else:
                checklist.add(f"Files: {section}", name, True, relative(path))


# ----------------------------------------------------------------- Consistency


def check_split():
    """The training and test rows do not overlap, and the sizes add up."""
    split = json.loads(Path(config.SPLIT_INDICES).read_text(encoding="utf-8"))
    train, test = set(split["train_row_indices"]), set(split["test_row_indices"])
    overlap = train & test
    passed = not overlap and len(train) + len(test) == split["total_rows"]
    return passed, f"train {len(train)}, test {len(test)}, total {split['total_rows']}, overlap {len(overlap)}"


def check_test_predictions_are_test_rows():
    """The saved test predictions cover exactly the held-out test rows."""
    split = json.loads(Path(config.SPLIT_INDICES).read_text(encoding="utf-8"))
    features = pd.read_csv(config.PROCESSED_DATA)
    expected = set(features.iloc[split["test_row_indices"]]["thread_id"])
    training = set(features.iloc[split["train_row_indices"]]["thread_id"])
    predicted = set(pd.read_csv(config.TEST_PREDICTIONS_FIVE_MODELS)["thread_id"])
    passed = predicted == expected and not (predicted & training)
    return passed, (
        f"{len(predicted)} predicted conversations; {len(predicted & expected)} are test rows; "
        f"{len(predicted & training)} are training rows"
    )


def check_test_metrics_match_predictions():
    """F1, ROC-AUC, Brier score and confusion counts recompute from the predictions."""
    predictions = pd.read_csv(config.TEST_PREDICTIONS_FIVE_MODELS)
    results = pd.read_csv(config.TEST_RESULTS_FIVE_MODELS).set_index("model")
    y_true = predictions[config.TARGET_COLUMN]
    mismatches = []

    for model_name in config.MODEL_NAMES:
        y_pred = predictions[f"{model_name} | prediction"]
        y_prob = predictions[f"{model_name} | scam_probability"]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        recomputed = {
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_prob),
            "brier_score": brier_score_loss(y_true, y_prob),
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        }
        for column, value in recomputed.items():
            if abs(results.loc[model_name, column] - value) > TOLERANCE:
                mismatches.append(f"{model_name} {column}")

    passed = not mismatches
    return passed, "All 5 models x 7 values match" if passed else "Mismatch: " + ", ".join(mismatches)


def check_test_size():
    """Every model was scored on the same 320 test conversations."""
    results = pd.read_csv(config.TEST_RESULTS_FIVE_MODELS)
    totals = results[["true_negative", "false_positive", "false_negative", "true_positive"]].sum(axis=1)
    split = json.loads(Path(config.SPLIT_INDICES).read_text(encoding="utf-8"))
    expected = len(split["test_row_indices"])
    passed = set(results["model"]) == set(config.MODEL_NAMES) and (totals == expected).all()
    return passed, f"Confusion-matrix totals: {sorted(set(totals))}; expected {expected} for all 5 models"


def check_repeated_cv_folds():
    """Every model has CV_FOLDS x CV_REPEATS paired folds."""
    repeated = pd.read_csv(config.REPEATED_CV_RESULTS)
    expected = config.CV_FOLDS * config.CV_REPEATS
    counts = repeated.groupby("model")["fold_index"].nunique().reindex(config.MODEL_NAMES)
    passed = (counts == expected).all()
    return passed, f"Folds per model: {counts.fillna(0).astype(int).to_dict()}; expected {expected}"


def check_significance_matches_folds():
    """Each mean difference in the significance tests equals the mean of the paired fold differences."""
    repeated = pd.read_csv(config.REPEATED_CV_RESULTS)
    significance = pd.read_csv(config.SIGNIFICANCE_TESTS)
    mismatches = []

    for _, row in significance.iterrows():
        wide = repeated.pivot(index="fold_index", columns="model", values=row["metric"])
        expected = (wide[row["model"]] - wide[config.REFERENCE_MODEL]).mean()
        if abs(expected - row["mean_difference"]) > TOLERANCE:
            mismatches.append(f"{row['metric']} {row['model']}")

    passed = not mismatches
    return passed, (
        f"All {len(significance)} mean differences match the folds" if passed else "Mismatch: " + ", ".join(mismatches)
    )


def check_holm(file_name, path):
    """Holm-corrected p-values are never below the raw ones, and 'significant' follows alpha."""

    def check():
        data = pd.read_csv(path)
        not_lower = (data["p_value_holm"] >= data["p_value"] - TOLERANCE).all()
        flags = (data["significant"].astype(bool) == (data["p_value_holm"] < config.SIGNIFICANCE_ALPHA)).all()
        return not_lower and flags, (
            f"{file_name}: {len(data)} tests; Holm >= raw: {not_lower}; "
            f"'significant' matches Holm p < {config.SIGNIFICANCE_ALPHA}: {flags}"
        )

    return check


def check_calibration_selection():
    """Each model has exactly one selected method, and it has the lowest calibrated Brier score."""
    calibration = pd.read_csv(config.CALIBRATION_RESULTS_FIVE_MODELS)
    calibration["selected"] = calibration["selected"].astype(bool)
    problems, chosen = [], []

    for model_name in config.MODEL_NAMES:
        rows = calibration[calibration["model"] == model_name]
        selected = rows[rows["selected"]]
        calibrated = rows[rows["method"] != "uncalibrated"]
        if len(selected) != 1:
            problems.append(f"{model_name}: {len(selected)} selected")
            continue
        if selected["brier_score_mean"].iloc[0] > calibrated["brier_score_mean"].min() + TOLERANCE:
            problems.append(f"{model_name}: selected method is not the lowest Brier")
        chosen.append(f"{model_name} = {selected['method'].iloc[0]}")

    passed = not problems
    return passed, "; ".join(chosen) if passed else "; ".join(problems)


def check_ablation_shape():
    """Every model has one row per feature, ranked 1 to 4."""
    ablation = pd.read_csv(config.ABLATION_RESULTS_FIVE_MODELS)
    problems = []
    for model_name in config.MODEL_NAMES:
        rows = ablation[ablation["model"] == model_name]
        if set(rows["removed_feature"]) != set(config.FEATURE_COLUMNS):
            problems.append(f"{model_name}: features {sorted(rows['removed_feature'])}")
        if sorted(rows["rank"]) != list(range(1, len(config.FEATURE_COLUMNS) + 1)):
            problems.append(f"{model_name}: ranks {sorted(rows['rank'])}")
    passed = not problems
    return passed, f"{len(ablation)} rows (5 models x 4 features)" if passed else "; ".join(problems)


def check_loto():
    """Every scam type has a result for every model, and gap = seen recall - unseen recall."""
    loto = pd.read_csv(config.LOTO_RESULTS)
    features = pd.read_csv(config.PROCESSED_DATA)
    scam_types = set(features.loc[features[config.TARGET_COLUMN] == 1, "type"])
    complete = all(
        set(loto.loc[loto["scam_type"] == scam_type, "model"]) == set(config.MODEL_NAMES) for scam_type in scam_types
    )
    gap_ok = np.allclose(loto["recall_seen"] - loto["recall_unseen"], loto["generalisation_gap"], atol=TOLERANCE)
    return complete and gap_ok, (
        f"Scam types {sorted(scam_types)} x 5 models present: {complete}; gap formula consistent: {gap_ok}"
    )


def check_hyperparams():
    """The selected hyperparameter file names all five models."""
    text = Path(config.SELECTED_HYPERPARAMS_FIVE_MODELS).read_text(encoding="utf-8")
    json.loads(text)  # must be valid JSON
    missing = [model_name for model_name in config.MODEL_NAMES if model_name not in text]
    return not missing, "All 5 models present" if not missing else f"Missing: {missing}"


# ------------------------------------------------------------------- Integrity


def check_seeds():
    """Repeated CV uses a different seed from tuning, so it does not reuse the tuning folds (C1)."""
    passed = config.REPEATED_CV_SEED != config.SEED
    return passed, f"SEED = {config.SEED}, REPEATED_CV_SEED = {config.REPEATED_CV_SEED}"


def check_figures():
    """Every figure has a summary file and is newer than the results it shows."""
    problems = []
    for figure, inputs in FIGURE_INPUTS.items():
        png = Path(config.FIGURES_DIR) / f"{figure}.png"
        summary = Path(config.FIGURES_DIR) / f"{figure}_summary.txt"
        if not png.exists() or not summary.exists():
            problems.append(f"{figure}: missing PNG or summary")
            continue
        for name in inputs:
            source = path_of(name)
            if source and Path(source).exists() and Path(source).stat().st_mtime > png.stat().st_mtime:
                problems.append(f"{figure}: older than {Path(source).name} (re-run story 20 script)")
    passed = not problems
    return passed, f"{len(FIGURE_INPUTS)} figures with summaries, all up to date" if passed else "; ".join(problems)


def check_unit_tests():
    """The full unit test suite passes."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.strip().splitlines() if line.strip()]
    return completed.returncode == 0, lines[-1] if lines else completed.stderr.strip()[-200:]


# ---------------------------------------------------------------------- Report


def frozen_numbers():
    """
    Description:
        Builds the headline-number tables for the final report from the
        saved files, so the report quotes the same numbers the checks
        verified.
    """
    lines = []
    try:
        repeated = pd.read_csv(config.REPEATED_CV_RESULTS)
        test = pd.read_csv(config.TEST_RESULTS_FIVE_MODELS).set_index("model")
        calibration = pd.read_csv(config.CALIBRATION_RESULTS_FIVE_MODELS)
        calibration["selected"] = calibration["selected"].astype(bool)
        selected = calibration[calibration["selected"]].set_index("model")

        lines += [
            "| Model | Repeated-CV F1 | Test F1 | Test ROC-AUC | Test Brier | Missed scams | False alarms | Calibrated Brier (CV) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for model_name in config.MODEL_NAMES:
            cv_f1 = repeated.loc[repeated["model"] == model_name, "f1"].mean()
            row = test.loc[model_name]
            lines.append(
                f"| {model_name} | {cv_f1:.4f} | {row['f1']:.4f} | {row['roc_auc']:.4f} | {row['brier_score']:.4f} "
                f"| {int(row['false_negative'])} | {int(row['false_positive'])} "
                f"| {selected.loc[model_name, 'brier_score_mean']:.4f} ({selected.loc[model_name, 'method']}) |"
            )

        significance = pd.read_csv(config.SIGNIFICANCE_TESTS)
        lines += ["", f"| Metric | Model vs {config.REFERENCE_MODEL} | Mean difference | Holm p | Significant |",
                  "|---|---|---|---|---|"]
        for _, row in significance.iterrows():
            lines.append(
                f"| {row['metric']} | {row['model']} | {row['mean_difference']:+.4f} "
                f"| {row['p_value_holm']:.4f} | {'Yes' if bool(row['significant']) else 'No'} |"
            )
    except Exception as error:  # noqa: BLE001
        lines.append(f"Could not build the frozen numbers: {error}")
    return lines


def write_report(checklist, skipped_tests):
    total, failed = len(checklist.rows), len(checklist.failed)
    decision = "READY FOR FREEZE" if failed == 0 else "NOT READY - fix the failed checks and run again"

    lines = [
        "# Internal Freeze Checklist (Five Models)",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Freeze Decision",
        "",
        f"**{decision}**",
        "",
        f"Automated checks passed: {total - failed} of {total}.",
    ]
    if skipped_tests:
        lines.append("Unit tests were skipped (--skip-tests); run without the flag before the final freeze.")

    lines += ["", "## Automated Checks", "", "| Status | Section | Check | Evidence |", "|---|---|---|---|"]
    for row in checklist.rows:
        evidence = str(row["evidence"]).replace("|", "/")
        lines.append(f"| {'PASS' if row['passed'] else '**FAIL**'} | {row['section']} | {row['check']} | {evidence} |")

    lines += ["", "## Manual Checks", "", "Tick each item by hand before the freeze tag is created.", ""]
    lines += [f"- [ ] {item}" for item in MANUAL_ITEMS]

    lines += ["", "## Frozen Headline Numbers", "",
              "These are the numbers the final report should quote. They are read from the checked files above.", ""]
    lines += frozen_numbers()

    lines += [
        "",
        "## Freeze Rule",
        "",
        "After the freeze tag, no script is re-run and no result file changes. If an error is found later, "
        "it is fixed on a new branch, this checklist is run again, and the change is recorded in the challenges log.",
        "",
    ]

    output = Path(config.REPORTS_DIR) / "internal_freeze_checklist_five_models.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output, decision


def main():
    parser = argparse.ArgumentParser(description="Run the COMP903 internal freeze checklist.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running the unit tests.")
    arguments = parser.parse_args()

    checklist = Checklist()
    check_files(checklist)

    consistency = "Consistency"
    checklist.run(consistency, "Train/test split has no overlap", check_split)
    checklist.run(consistency, "Test predictions are exactly the held-out test rows", check_test_predictions_are_test_rows)
    checklist.run(consistency, "All models scored on the same test set size", check_test_size)
    checklist.run(consistency, "Test metrics recompute from saved predictions", check_test_metrics_match_predictions)
    checklist.run(consistency, "Repeated CV has 50 paired folds per model", check_repeated_cv_folds)
    checklist.run(consistency, "Significance tests match the repeated-CV folds", check_significance_matches_folds)
    checklist.run(consistency, "Holm correction applied correctly (significance tests)",
                  check_holm("significance_tests.csv", config.SIGNIFICANCE_TESTS))
    checklist.run(consistency, "Holm correction applied correctly (ablation)",
                  check_holm("ablation_results_five_models.csv", config.ABLATION_RESULTS_FIVE_MODELS))
    checklist.run(consistency, "One calibration method selected per model, with lowest Brier", check_calibration_selection)
    checklist.run(consistency, "Ablation covers 5 models x 4 features", check_ablation_shape)
    checklist.run(consistency, "Leave-one-type-out covers every scam type", check_loto)
    checklist.run(consistency, "Selected hyperparameters saved for all 5 models", check_hyperparams)

    integrity = "Integrity"
    checklist.run(integrity, "Repeated-CV seed differs from tuning seed", check_seeds)
    checklist.run(integrity, "Figures complete and up to date", check_figures)
    if not arguments.skip_tests:
        checklist.run(integrity, "Unit tests pass", check_unit_tests)

    output, decision = write_report(checklist, arguments.skip_tests)
    print(f"\n{len(checklist.rows) - len(checklist.failed)} of {len(checklist.rows)} checks passed: {decision}")
    print(f"Checklist saved to: {output}")
    sys.exit(0 if not checklist.failed else 1)


if __name__ == "__main__":
    main()
