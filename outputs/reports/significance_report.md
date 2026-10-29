# Significance Test Report

## Description

This report tests whether each new model differs from Logistic Regression by more than would be expected by chance.

## Method

- Data: 50 paired folds from repeated stratified cross-validation (10 repeats x 5 folds) on the training set only.
- Test: corrected resampled t-test (Nadeau & Bengio, 2003), which widens the uncertainty to allow for folds sharing training data.
- Multiple comparisons: Holm correction across the three new models, applied separately for each metric.
- Significance level: alpha = 0.05. Tests are two-sided.
- The held-out test set was not used.

## F1-score

Logistic Regression mean: 0.8889. Higher is better.

| Model | Mean | Difference | 95% CI of difference | Folds better | p-value | Holm p-value | Significant |
|---|---:|---:|---|---:|---:|---:|---|
| Gaussian Naive Bayes | 0.8809 | -0.0079 | -0.0264 to +0.0105 | 16/50 | 0.3920 | 1.0000 | No |
| Random Forest | 0.8929 | +0.0040 | -0.0091 to +0.0171 | 26/50 | 0.5450 | 1.0000 | No |
| k-Nearest Neighbours | 0.8913 | +0.0024 | -0.0094 to +0.0142 | 25/50 | 0.6865 | 1.0000 | No |

Result: no new model differed significantly from Logistic Regression on this metric after Holm correction, so the differences cannot be distinguished from chance.

## Brier score (before calibration)

Logistic Regression mean: 0.0844. Lower is better.

| Model | Mean | Difference | 95% CI of difference | Folds better | p-value | Holm p-value | Significant |
|---|---:|---:|---|---:|---:|---:|---|
| Gaussian Naive Bayes | 0.0925 | +0.0080 | -0.0001 to +0.0162 | 9/50 | 0.0540 | 0.0540 | No |
| Random Forest | 0.0788 | -0.0057 | -0.0102 to -0.0011 | 44/50 | 0.0153 | 0.0306 | Yes |
| k-Nearest Neighbours | 0.0778 | -0.0066 | -0.0106 to -0.0026 | 48/50 | 0.0017 | 0.0050 | Yes |

Result: after Holm correction, the following differed significantly from Logistic Regression on this metric: Random Forest (better than reference); k-Nearest Neighbours (better than reference).

## Reading This Report

- **Difference** is the new model's mean minus the reference mean.
- **95% CI** is the range the true difference is likely to fall in. If it crosses zero, the two models cannot be told apart with confidence.
- **Folds better** counts the folds in which the new model beat the reference.
- **Holm p-value** is the p-value adjusted for testing three models; a value below 0.05 means the difference is unlikely to be due to chance.

## Reproducibility Note

Generated from `repeated_cv_results.csv` by `significance_tests.py`.

Generated at: 2026-10-29T23:35:13
