"""
test_significance.py

Description:
    Runs automated unit checks on the statistical functions in
    significance_tests.py: the corrected resampled t-test and the Holm
    correction. The tests use small hand-made inputs with known answers,
    so they do not depend on the project data files.

Run:
    - pytest tests/test_significance.py -v
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.evaluation.significance_tests import corrected_resampled_ttest, holm_correction


def test_holm_correction_matches_hand_calculation():
    # Sorted: 0.01 x 3 = 0.03, 0.03 x 2 = 0.06, 0.04 x 1 = 0.04 -> kept at 0.06 (step-down)
    adjusted = holm_correction([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_holm_correction_never_exceeds_one():
    adjusted = holm_correction([0.6, 0.7, 0.9])
    assert max(adjusted) <= 1.0


def test_holm_adjusted_values_are_never_smaller_than_raw():
    raw = [0.002, 0.03, 0.2]
    adjusted = holm_correction(raw)
    assert all(a >= r for a, r in zip(adjusted, raw))


def test_identical_scores_give_no_difference():
    result = corrected_resampled_ttest(np.zeros(50))
    assert result["mean_difference"] == 0
    assert result["p_value"] == 1.0


def test_corrected_test_matches_hand_calculation():
    differences = np.array([0.02, 0.01, 0.03, 0.00, 0.02])
    k = len(differences)
    expected_t = differences.mean() / np.sqrt((1 / k + 0.25) * differences.var(ddof=1))

    result = corrected_resampled_ttest(differences, test_train_ratio=0.25)

    assert np.isclose(result["t_statistic"], expected_t)
    assert result["degrees_of_freedom"] == k - 1


def test_correction_is_more_cautious_than_ordinary_t_test():
    rng = np.random.default_rng(0)
    differences = rng.normal(loc=0.004, scale=0.015, size=50)

    corrected_p = corrected_resampled_ttest(differences)["p_value"]
    ordinary_p = stats.ttest_1samp(differences, 0).pvalue

    assert corrected_p > ordinary_p


def test_confidence_interval_contains_mean_difference():
    differences = np.array([0.01, -0.02, 0.03, 0.00, 0.01, 0.02])
    result = corrected_resampled_ttest(differences)
    assert result["ci_low"] < result["mean_difference"] < result["ci_high"]