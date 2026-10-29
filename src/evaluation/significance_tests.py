"""
significance_tests.py

Description:
    Tests whether the differences between each new model and the reference
    model (Logistic Regression) are real or could be due to chance. It uses
    the 50 paired fold results from repeated_cv_five_models.py.

    For each new model and each metric, the per-fold differences from
    Logistic Regression are tested with the corrected resampled t-test
    (Nadeau & Bengio, 2003). The correction is needed because the
    cross-validation folds share training data, which makes an ordinary
    paired t-test too optimistic.

    Three new models are compared with the same reference, so the p-values
    are adjusted with the Holm correction to keep the overall chance of a
    false positive at alpha (0.05).

    Metrics tested:
    - F1-score (RQ1): higher is better.
    - Brier score (RQ2): lower is better. These are the uncalibrated Brier
      scores from repeated cross-validation; calibrated Brier scores are
      tested after the calibration story.

Inputs:
    - outputs/reports/repeated_cv_results.csv

Outputs:
    - outputs/reports/significance_tests.csv
    - outputs/reports/significance_report.md

Run:
    - python src/evaluation/significance_tests.py
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import (
    CV_FOLDS,
    CV_REPEATS,
    NEW_MODELS,
    REFERENCE_MODEL,
    REPEATED_CV_RESULTS,
    SIGNIFICANCE_ALPHA,
    SIGNIFICANCE_REPORT,
    SIGNIFICANCE_TESTS,
)

# Metric name -> (label, True if higher is better)
TESTED_METRICS = {
    "f1": ("F1-score", True),
    "brier_score": ("Brier score (before calibration)", False),
}

# In k-fold cross-validation each validation fold is 1/k of the data and the
# training part is (k-1)/k, so the test-to-train size ratio is 1/(k-1).
TEST_TRAIN_RATIO = 1 / (CV_FOLDS - 1)


def corrected_resampled_ttest(differences, test_train_ratio=TEST_TRAIN_RATIO):
    """
    Description:
        Runs the corrected resampled t-test of Nadeau and Bengio (2003) on
        paired per-fold differences. The variance is inflated by the
        test-to-train ratio to account for overlapping training sets.

    Input:
        differences: Array of per-fold differences (model minus reference).
        test_train_ratio: Size of a validation fold divided by the size of
                          its training part.

    Output:
        Returns a dictionary with the mean difference, t statistic, degrees
        of freedom, two-sided p-value and 95% confidence interval.
    """
    differences = np.asarray(differences, dtype=float)
    k = len(differences)

    mean_difference = differences.mean()
    variance = differences.var(ddof=1)
    corrected_se = np.sqrt((1 / k + test_train_ratio) * variance)
    degrees_of_freedom = k - 1

    if corrected_se == 0:
        t_statistic = 0.0
        p_value = 1.0
    else:
        t_statistic = mean_difference / corrected_se
        p_value = 2 * stats.t.sf(abs(t_statistic), degrees_of_freedom)

    t_critical = stats.t.ppf(0.975, degrees_of_freedom)

    return {
        "mean_difference": mean_difference,
        "t_statistic": t_statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "ci_low": mean_difference - t_critical * corrected_se,
        "ci_high": mean_difference + t_critical * corrected_se,
    }


def holm_correction(p_values):
    """
    Description:
        Adjusts a set of p-values with the Holm step-down method, which
        controls the chance of at least one false positive across all the
        comparisons.

    Input:
        p_values: List of raw p-values.

    Output:
        Returns a list of adjusted p-values in the original order.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)

    adjusted = np.empty(m)
    running_max = 0.0

    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        running_max = max(running_max, value)
        adjusted[index] = running_max

    return adjusted.tolist()


def load_repeated_cv_results(file_path):
    """
    Description:
        Loads the repeated cross-validation results and checks they are
        complete and paired.

    Input:
        file_path: Path to repeated_cv_results.csv.

    Output:
        Returns a pandas DataFrame with one row per model per fold.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"Repeated CV results not found: {file_path}. "
            "Run src/models/repeated_cv_five_models.py first."
        )

    results_df = pd.read_csv(file_path)

    expected_folds = CV_FOLDS * CV_REPEATS
    for model_name in [REFERENCE_MODEL] + NEW_MODELS:
        count = (results_df["model"] == model_name).sum()
        if count != expected_folds:
            raise ValueError(f"{model_name} has {count} folds, expected {expected_folds}")

    if (results_df.groupby("fold_index")["fold_signature"].nunique() != 1).any():
        raise ValueError("Models were not scored on identical folds; results are not paired")

    return results_df


def run_tests(results_df):
    """
    Description:
        Compares every new model with the reference model on each tested
        metric, then applies the Holm correction within each metric.

    Input:
        results_df: DataFrame of per-fold repeated CV results.

    Output:
        Returns a DataFrame with one row per model per metric.
    """
    pivots = {
        metric: results_df.pivot(index="fold_index", columns="model", values=metric)
        for metric in TESTED_METRICS
    }

    rows = []

    for metric, (label, higher_is_better) in TESTED_METRICS.items():
        table = pivots[metric]
        metric_rows = []

        for model_name in NEW_MODELS:
            differences = table[model_name] - table[REFERENCE_MODEL]
            result = corrected_resampled_ttest(differences)

            better = result["mean_difference"] > 0 if higher_is_better else result["mean_difference"] < 0

            metric_rows.append(
                {
                    "metric": metric,
                    "metric_label": label,
                    "model": model_name,
                    "reference_model": REFERENCE_MODEL,
                    "model_mean": table[model_name].mean(),
                    "reference_mean": table[REFERENCE_MODEL].mean(),
                    **result,
                    "direction": "better than reference" if better else "worse than reference",
                    "folds_model_better": int(((differences > 0) if higher_is_better else (differences < 0)).sum()),
                }
            )

        adjusted = holm_correction([row["p_value"] for row in metric_rows])

        for row, p_adjusted in zip(metric_rows, adjusted):
            row["p_value_holm"] = p_adjusted
            row["significant"] = p_adjusted < SIGNIFICANCE_ALPHA
            rows.append(row)

    return pd.DataFrame(rows)


def build_report(tests_df):
    """
    Description:
        Builds a readable markdown report of the significance tests.

    Input:
        tests_df: DataFrame returned by run_tests().

    Output:
        Returns the report as a single string.
    """
    total_folds = CV_FOLDS * CV_REPEATS

    lines = [
        "# Significance Test Report",
        "",
        "## Description",
        "",
        f"This report tests whether each new model differs from {REFERENCE_MODEL} by more than "
        "would be expected by chance.",
        "",
        "## Method",
        "",
        f"- Data: {total_folds} paired folds from repeated stratified cross-validation "
        f"({CV_REPEATS} repeats x {CV_FOLDS} folds) on the training set only.",
        "- Test: corrected resampled t-test (Nadeau & Bengio, 2003), which widens the "
        "uncertainty to allow for folds sharing training data.",
        "- Multiple comparisons: Holm correction across the three new models, applied "
        "separately for each metric.",
        f"- Significance level: alpha = {SIGNIFICANCE_ALPHA}. Tests are two-sided.",
        "- The held-out test set was not used.",
        "",
    ]

    for metric, (label, higher_is_better) in TESTED_METRICS.items():
        metric_df = tests_df[tests_df["metric"] == metric]
        reference_mean = metric_df["reference_mean"].iloc[0]

        lines += [
            f"## {label}",
            "",
            f"{REFERENCE_MODEL} mean: {reference_mean:.4f}. "
            f"{'Higher' if higher_is_better else 'Lower'} is better.",
            "",
            "| Model | Mean | Difference | 95% CI of difference | Folds better | p-value | Holm p-value | Significant |",
            "|---|---:|---:|---|---:|---:|---:|---|",
        ]

        for _, row in metric_df.iterrows():
            lines.append(
                f"| {row['model']} | {row['model_mean']:.4f} | {row['mean_difference']:+.4f} | "
                f"{row['ci_low']:+.4f} to {row['ci_high']:+.4f} | "
                f"{row['folds_model_better']}/{total_folds} | {row['p_value']:.4f} | "
                f"{row['p_value_holm']:.4f} | {'Yes' if row['significant'] else 'No'} |"
            )

        significant = metric_df[metric_df["significant"]]
        if significant.empty:
            summary = (
                f"Result: no new model differed significantly from {REFERENCE_MODEL} on this "
                f"metric after Holm correction, so the differences cannot be distinguished from chance."
            )
        else:
            parts = [f"{row['model']} ({row['direction']})" for _, row in significant.iterrows()]
            summary = (
                f"Result: after Holm correction, the following differed significantly from "
                f"{REFERENCE_MODEL} on this metric: " + "; ".join(parts) + "."
            )

        lines += ["", summary, ""]

    lines += [
        "## Reading This Report",
        "",
        "- **Difference** is the new model's mean minus the reference mean.",
        "- **95% CI** is the range the true difference is likely to fall in. If it crosses zero, "
        "the two models cannot be told apart with confidence.",
        "- **Folds better** counts the folds in which the new model beat the reference.",
        "- **Holm p-value** is the p-value adjusted for testing three models; a value below "
        f"{SIGNIFICANCE_ALPHA} means the difference is unlikely to be due to chance.",
        "",
        "## Reproducibility Note",
        "",
        "Generated from `repeated_cv_results.csv` by `significance_tests.py`.",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
    ]

    return "\n".join(lines) + "\n"


def print_summary(tests_df):
    """
    Description:
        Prints the key test results to the terminal.

    Input:
        tests_df: DataFrame returned by run_tests().

    Output:
        Displays the summary in the terminal.
    """
    print(f"Significance tests against {REFERENCE_MODEL} (corrected resampled t-test, Holm, alpha={SIGNIFICANCE_ALPHA})")
    print("=" * 96)

    for _, row in tests_df.iterrows():
        print(
            f"{row['metric']:12s} {row['model']:22s} diff {row['mean_difference']:+.4f} "
            f"| 95% CI {row['ci_low']:+.4f} to {row['ci_high']:+.4f} "
            f"| p {row['p_value']:.4f} | Holm p {row['p_value_holm']:.4f} "
            f"| {'SIGNIFICANT' if row['significant'] else 'not significant'}"
        )

    print("")
    print(f"Saved test results to: {SIGNIFICANCE_TESTS}")
    print(f"Saved report to: {SIGNIFICANCE_REPORT}")


def main():
    """
    Description:
        Loads the repeated CV results, runs the significance tests and saves
        the results and report.

    Input:
        No direct function input. Reads paths and settings from config.py.

    Output:
        Creates significance_tests.csv and significance_report.md.
    """
    results_df = load_repeated_cv_results(REPEATED_CV_RESULTS)

    tests_df = run_tests(results_df)

    SIGNIFICANCE_TESTS.parent.mkdir(parents=True, exist_ok=True)
    tests_df.to_csv(SIGNIFICANCE_TESTS, index=False)
    SIGNIFICANCE_REPORT.write_text(build_report(tests_df), encoding="utf-8")

    print_summary(tests_df)


if __name__ == "__main__":
    main()